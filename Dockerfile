FROM mambaorg/micromamba:0.15.3
USER root
RUN mkdir /opt/azure_deployment
RUN chmod -R 777 /opt/azure_deployment
WORKDIR /opt/azure_deployment
USER micromamba
COPY environment.yml environment.yml
RUN micromamba install -y -n base -f environment.yml && micromamba clean --all --yes
ENV PINECONE_API_KEY=['YOUR PINECONE_API_KEY']
ENV PINECONE_API_ENV=['YOUR PINECONE_API_ENV']
ENV AZURE_OPENAI_API_KEY=['YOUR AZURE OPENAI KEY']
ENV AZURE_OPENAI_API_BASE=['YOUR AZURE_OPENAI_API_BASE']
ENV DEPLOYMENT_NAME=['YOUR AZURE DEPLOYMENT_NAME']
ENV AZURE_STORAGE_URL=['YOUR AZURE_STORAGE_URL']
ENV AZURE_STORAGE_ACCOUNT_KEYS=['YOUR AZURE_STORAGE_ACCOUNT_KEYS']
ENV AZURE_STORAGE_ACCOUNT_NAME=['YOUR AZURE_STORAGE_ACCOUNT_NAME']
COPY run.sh run.sh
COPY autonomous-hr-chatbot autonomous-hr-chatbot
USER root
RUN chmod a+x run.sh
CMD ["./run.sh"]