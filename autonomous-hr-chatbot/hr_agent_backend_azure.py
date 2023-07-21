# load core modules
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
# load agents and tools modules
import pandas as pd
from azure.storage.filedatalake import DataLakeServiceClient
from io import StringIO
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain import LLMMathChain
import os 

PINECONE_API_KEY        = os.environ['PINECONE_API_KEY']
PINECONE_API_ENV        = os.environ['PINECONE_API_ENV']
AZURE_OPENAI_API_KEY    = os.environ['OPENAI_API_KEY']
AZURE_OPENAI_API_BASE   = os.environ['AZURE_OPENAI_API_BASE']
DEPLOYMENT_NAME         = os.environ['DEPLOYMENT_NAME']
AZURE_STORAGE_URL       = os.environ['AZURE_STORAGE_URL']
AZURE_STORAGE_ACCOUNT_KEYS = os.environ['AZURE_STORAGE_ACCOUNT_KEYS']
AZURE_STORAGE_ACCOUNT_NAME = os.environ['AZURE_STORAGE_ACCOUNT_NAME']

# initialize pinecone client and connect to pinecone index
pinecone.init(
        api_key=PINECONE_API_KEY,  
        environment=PINECONE_API_ENV  
) 

index_name = 'tk-policy'
index = pinecone.Index(index_name) # connect to pinecone index


# initialize embeddings object; for use with user query/input
embed = OpenAIEmbeddings(
                deployment=DEPLOYMENT_NAME,
                model="text-embedding-ada-002",
                openai_api_key=AZURE_OPENAI_API_KEY,
                openai_api_base=AZURE_OPENAI_API_BASE,
                openai_api_type="azure",
            )

# initialize langchain vectorstore(pinecone) object
text_field = 'text' # key of dict that stores the text metadata in the index
vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

#initialize LLM object
llm = AzureChatOpenAI(    
    deployment_name=DEPLOYMENT_NAME, 
    model_name="gpt-35-turbo", 
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_version = '2023-03-15-preview', 
    openai_api_base=AZURE_OPENAI_API_BASE,
    openai_api_type='azure'
    )

# initialize vectorstore retriever object
employee_policy = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

# create employee data tool 
client = DataLakeServiceClient( # authenticate to azure datalake
                               account_url = AZURE_STORAGE_URL,
                               credential = AZURE_STORAGE_ACCOUNT_KEYS
                              )
# azure data lake boilerplate to load from file system.  
file = client.get_file_system_client(AZURE_STORAGE_ACCOUNT_NAME) \
             .get_file_client("employee_data/employee_data.csv") \
             .download_file() \
             .readall() \
             .decode('utf-8') 

csv_file = StringIO(file) 
df = pd.read_csv(csv_file) # load employee_data.csv as dataframe
python = PythonAstREPLTool(locals={"df": df}) # set access of python_repl tool to the dataframe

csv_file = StringIO(file)
df = pd.read_csv(csv_file) # load employee_data.csv as dataframe
python = PythonAstREPLTool(locals={"df": df}) # set access of python_repl tool to the dataframe

# create calculator tool
calculator = LLMMathChain.from_llm(llm=llm, verbose=True)

# create variables for f strings embedded in the prompts
user = 'Alexander Verdad' # set user
df_columns = df.columns.to_list() # print column names of df

# prep the (tk policy) vectordb retriever, the python_repl(with df access) and langchain calculator as tools for the agent
tools = [
    Tool(
        name = "Timekeeping Policies",
        func=employee_policy.run,
        description="""
        Useful for when you need to answer questions about employee timekeeping policies.

        <user>: What is the policy on unused vacation leave?
        <assistant>: I need to check the timekeeping policies to answer this question.
        <assistant>: Action: Timekeeping Policies
        <assistant>: Action Input: Vacation Leave Policy - Unused Leave
        ...
        """
    ),
    Tool(
        name = "Employee Data",
        func=python.run,
        description = f"""
        Useful for when you need to answer questions about employee data stored in pandas dataframe 'df'. 
        Run python pandas operations on 'df' to help you get the right answer.
        'df' has the following columns: {df_columns}
        
        <user>: How many Sick Leave do I have left?
        <assistant>: df[df['name'] == '{user}']['sick_leave']
        <assistant>: You have n sick leaves left.              
        """
    ),
    Tool(
        name = "Calculator",
        func=calculator.run,
        description = f"""
        Useful when you need to do math operations or arithmetic.
        """
    )
]

# change the value of the prefix argument in the initialize_agent function. This will overwrite the default prompt template of the zero shot agent type
agent_kwargs = {'prefix': f'You are friendly HR assistant. You are tasked to assist the current user: {user} on questions related to HR. You have access to the following tools:'}


# initialize the LLM agent
agent = initialize_agent(tools, 
                         llm, 
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                         verbose=True, 
                         agent_kwargs=agent_kwargs
                         )
# define q and a function for frontend
def get_response(user_input):
    response = agent.run(user_input)
    return response