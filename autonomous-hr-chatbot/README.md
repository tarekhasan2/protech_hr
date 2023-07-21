# HR Chatbot built using ChatGPT, LangChain, Pinecone and Streamlit


---
#### HOW TO DEPLOY THE APP IN AZURE
---

1. You need some keys to deploy this app:
	1. Pinecone api key and api env: You get it in Pinecone](https://www.pinecone.io/). Just create a free account.
	2. Azure openai key, base, deployment name: you will get this in Azure OpenAI Service](https://azure.microsoft.com/en-us/products/cognitive-services/openai-service). 
	3. Azure storage url, account keys and account name. You will get this in Azure Data Lake](https://azure.microsoft.com/en-us/solutions/data-lake). Before do that upload employee_data.csv file to Azure Data Lake Storage.  

2. Now put all the keys in Dockerfile line[9-16].
Docker container is ready to deploy. 

3. To deploy this app please follw the instruction from "Part 3: Deploy your Dockerized App to Azure": https://towardsdatascience.com/beginner-guide-to-streamlit-deployment-on-azure-f6618eee1ba9

If you failed to deploy or need any help please contact me.
