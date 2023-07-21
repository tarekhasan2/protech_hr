# HR Chatbot built using ChatGPT, LangChain, Pinecone and Streamlit


#### How to use this repo

1. Install python 3.10. [Windows](https://www.tomshardware.com/how-to/install-python-on-windows-10-and-11#:~:text=1.,and%20download%20the%20Windows%20installer.&text=2.,is%20added%20to%20your%20path.), [Mac](https://www.codingforentrepreneurs.com/guides/install-python-on-macos/) 
2. Clone the repo to a local directory.
3. Install 'pipenv' with - `pip install pipenv`
4. Navigate to the local directory where is `Pipfile` and run this command in your terminal to install all prerequisite modules - `pipenv install`
5. activate the virtual env with - `pipenv shell`
6. go to repository by - `cd autonomous-hr-chatbot` 
7. Run `streamlit run hr_agent_frontent.py` in your terminal


---
### Tech Stack
---

[Azure OpenAI Service](https://azure.microsoft.com/en-us/products/cognitive-services/openai-service) - the OpenAI service offering for Azure customers.  
[LangChain](https://python.langchain.com/docs/get_started/introduction.html) - development frame work for building apps around LLMs.    
[Pinecone](https://www.pinecone.io/) - the vector database for storing the embeddings.  
[Streamlit](https://streamlit.io/) - used for the front end. Lightweight framework for deploying python web apps.  
[Azure Data Lake](https://azure.microsoft.com/en-us/solutions/data-lake) - for landing the employee data csv files. Any other cloud storage should work just as well (blob, S3 etc).    
[Azure Data Factory](https://azure.microsoft.com/en-ca/products/data-factory/) - used to create the data pipeline.  
[SAP HCM](https://www.sap.com/sea/products/hcm/what-is-sap-hr.html) - the source system for employee data.   

### Video Demo 
---

[Youtube Link](https://www.youtube.com/watch?v=id7XRcEIBvg&ab_channel=StephenBonifacio)
