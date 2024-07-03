#%% [markdown]
# # Using Ollama with OpenAI's API and Elastic Search in Docker
# To run the Llama3 model in a docker container on localhost:11434, you can use Ollama locally.
# Check the docker-compose.yml file for more details.

#%%
# # Import the required libraries
import requests, json
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm
from openai import OpenAI # Remember to install openai using !pip install openai

url_data = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/01-intro/documents.json'
r_data = requests.get(url_data)

# Assuming you want to save the content of the URL to a file named 'documents.json' in the current directory
with open('documents.json', 'wb') as f:
    f.write(r_data.content)
    
# %%
# # Load json data and index by text and keyword fields
with open('documents.json', 'rt') as f_in:
    docs_raw = json.load(f_in)
    
documents = []
for course_dict in docs_raw:
    course_name = course_dict['course']
    for doc in course_dict['documents']:
        doc['course'] = course_name
        documents.append(doc)
        
#%%
# Using Elastic Search
from elasticsearch import Elasticsearch

es_client = Elasticsearch('http://localhost:9200')
print(es_client.info())

# %%
# Index the documents
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
es_client.indices.delete(index=index_name, ignore=[400, 404]) # Maybe this works.
es_client.indices.create(index=index_name, body=index_settings)

for doc in tqdm(documents):
    es_client.index(index=index_name, body=doc)

# %%
# Query using elasticsearch
def elastic_search(query):
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"], #^3 to give priority to the question field
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "data-engineering-zoomcamp"
                    }
                }
            }
        }
    }

    response = es_client.search(index=index_name, body=search_query)
    result_docs = []
    
    for hit in response['hits']['hits']:
        result_docs.append(hit['_source'])
    
    return result_docs

# %%
# Define the build prompt method
def build_prompt(query, search_results):
    prompt_template = """
    You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
    Use only the facts from the CONTEXT when answering the QUESTION.

    QUESTION: {question}

    CONTEXT: 
    {context}
    """.strip()

    context = ""
    
    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

#%%
# Define OpenAI API client
client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)

#%%
# Define the LLM api call
def llm(prompt):
    response = client.chat.completions.create(
        model='llama3',
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

#%%
# Define the RAG method
def rag(query):
    search_results = elastic_search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer

#%%
# Test the RAG method
query = "I just found out about this course, can I still join?"
print(f"Answer from Llama3 with finetuning:\n {rag(query)}")


#%% [markdown]
# Useful links to use Ollama and ElasticSearch in Docker:
# - [Ollama](https://hub.docker.com/r/ollama/ollama)
# - [ElasticSearch](https://hub.docker.com/_/elasticsearch)
# - [Podman Compose](https://podman-desktop.io/docs/compose/running-compose)

# %%
