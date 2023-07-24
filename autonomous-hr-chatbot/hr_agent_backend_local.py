# load core modules
import pinecone ## Pinecone is vactorstore. We use it to store vactor.
from langchain.embeddings.openai import OpenAIEmbeddings ## We use openai embedding to embad input string. It check the similarity in vactor string.
from langchain.vectorstores import Pinecone ## this is pinecone from langchain. This is for use in langchain pinecone code. 
from langchain.chat_models import ChatOpenAI ## ChatOpenAI use to create the Large Language Model. 
from langchain.chains import RetrievalQA ## This is retriver that use to retrive a correct vactor from vactor store

import pandas as pd ## pandas use to read dataframe
from io import StringIO
from langchain.tools.python.tool import PythonAstREPLTool ## This is python repl that use to get result from Dataframe immediately. 
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor, AgentType ## AgentExecutor is use to call agent, call tool, get the result and also passing all that information back into the Agent.
from langchain import LLMMathChain ## This is a langchain object that use for mathematical calculation.
from langchain.memory import ConversationBufferMemory ## This create a memory
from langchain import OpenAI, LLMChain 


import os

########### This are important API keys
# initialize pinecone client and connect to pinecone index
# PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
# PINECONE_API_ENV = os.environ['PINECONE_API_ENV']
OPENAI_API_KEY   = os.environ['OPENAI_API_KEY']
PINECONE_API_ENV = 'northamerica-northeast1-gcp'
PINECONE_API_KEY = 'e80c9a1c-d3ae-424f-a61d-f2f7b6aad746'
############


#### Initializing pinecone 
pinecone.init(
        api_key=PINECONE_API_KEY,  
        environment=PINECONE_API_ENV  
) 

index_name = 'tk-policy' # This is our index name in out vactorstore
index = pinecone.Index(index_name) # connect to pinecone index

# initialize embeddings object; for use with user query/input
embed = OpenAIEmbeddings(
                model = 'text-embedding-ada-002',
                openai_api_key=OPENAI_API_KEY,
            )

# initialize langchain vectorstore(pinecone) object
text_field = 'text' # key of dict that stores the text metadata in the index
vectorstore = Pinecone(
    index, embed.embed_query, text_field
)


#### This is out Large Language Model by openai
llm = ChatOpenAI(    
    openai_api_key=OPENAI_API_KEY, 
    model_name="gpt-3.5-turbo", 
    temperature=0.0
    )


# initialize vectorstore retriever object. This function will use to retriev vactor from vactorstore
employee_policy = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

df = pd.read_csv("employee_data.csv") # load employee_data.csv as dataframe
python = PythonAstREPLTool(locals={"df": df}) # set access of python_repl tool to the dataframe

# create calculator tool
calculator = LLMMathChain.from_llm(llm=llm, verbose=True)

# create variables for f strings embedded in the prompts


user = 'Alexander Verdad' # set user

df_columns = df.columns.to_list() # print column names of df

# prep the (tk policy) vectordb retriever, the python_repl(with df access) and langchain calculator as tools for the agent
## This is out 3 tools. "Employee Policies" use to answer query from policy documents. "Employee Database" use to answer from employee csv file
## and Calculator use for matematical calculation. "description" is importent, it is prompt. this prompt use to give instruction to openai when to 
## use which tool.


tools = [
    Tool(
        name = "Employee Policies",
        func = employee_policy.run,
        description="""
        Useful for when you need to answer questions about employee policies, rules and regulations. You should give priority to using it.

        <user>: What is the policy on unused vacation leave?
        <assistant>: I need to check the policies to answer this question.
        <assistant>: Action: Employee Policies
        <assistant>: Action Input: Vacation Leave Policy - Unused Leave
        ...
        """
    ),
    Tool(
        name = "Employee Database",
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


prefix = """You are friendly HR assistant. You are tasked to answer questions related to HR. You have access to the following tools:"""
suffix = """Begin!"
{chat_history}
Question: {input}
{agent_scratchpad}"""

# ZeroShotAgent use for create agen that has no mamory. It will work only on current action. 
prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)
memory = ConversationBufferMemory(memory_key="chat_history") ### This class use to create a memory 

llm_chain = LLMChain(llm=llm, prompt=prompt) ## It create a chain of prompt and LLM and create output.


agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True) # ZeroShotAgent use for create agen that has no mamory. It will work only on current action.

## This agent chain will manage to call agent, call tool, get the result and also passing all that information back into the mamory. We give mamory to it.
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)

# define q and a function for frontend
def get_response(user_input):
    try:
        response = agent_chain.run(user_input) ### get the response from agent chain
    
    except Exception as e:
        response = str(e)
        if response.startswith("Could not parse LLM output: `"):
            response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
            return response
        else:
            raise Exception(str(e))

    return response
