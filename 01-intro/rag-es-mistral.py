#%% [markdown]
# # RAG system using ElasticSearch and Mistral LLM

#%% [markdown]
# Importing the libraries
import requests, os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import OpenAI
# from mistralai.client import MistralClient
# from mistralai.models.chat_completion import ChatMessage

#%% [markdown]
# Download the json data
docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)

documents = []

for course in docs_response.json():
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)
        
#%% [markdown]
# Index the data using elasticsearch

# Connect to the Elasticsearch client
es_client = Elasticsearch('http://localhost:9200')

index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"} 
        }
    }
}
index_name = "course-questions"


# %% [markdown]
# Load Nvidia api key and initialize the client

# Load environment variables from .env file
load_dotenv(dotenv_path='.env')

# Get the API key from the environment variable
# Access the API key
api_key_nvidia = os.environ["NVIDIA_API_KEY"]

client_nvidia = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = api_key_nvidia
)

def llm_nvidia(query):
    response = client_nvidia.chat.completions.create(
        model="meta/llama3-70b-instruct",
        messages=[{"role":"user","content": query}],
        temperature=0.5,
        top_p=1,
        max_tokens=1024,
    )
    
    return response.choices[0].message.content
# %% [markdown]
# # Query the model and get the response
# Zero-shot query

query = "The course has already started, can I still enroll?"
print(f'LLM Answer: {llm_nvidia(query)}')

# %%
# Remove later
def llm_nvidia_stream(query):
    response = client_nvidia.chat.completions.create(
        model="meta/llama3-70b-instruct",
        messages=[{"role":"user","content": query}],
        temperature=0.5,
        top_p=1,
        max_tokens=1024,
        stream=True
    )
    
    print(f'LLM Response: {response}')
    
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            answer = chunk.choices[0].delta.content
    
    return answer
