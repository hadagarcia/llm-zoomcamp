#%% [markdown]
# # Homework

# Importing libraries
import requests, os
import tiktoken 
from dotenv import load_dotenv

# Load the data

docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)
        
#%%
# Index the data using elasticsearch
from elasticsearch import Elasticsearch

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

# First delete the index if it exists (if we want to avoid an error while running multiple times)
if es_client.indices.exists(index=index_name):
    es_client.indices.delete(index=index_name, ignore=[400, 404])

es_client.indices.create(index=index_name, body=index_settings)

for doc in documents:
    es_client.index(index=index_name, body=doc)

# %%
# Query using elasticsearch
def elastic_search(query, filter=''):
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^4", "text"], #^4 to give priority to the question field
                        "type": "best_fields"
                    }
                }
            }
        }
    }
    
    if filter:
        print(f"Filtering by: {filter}")
        search_query['query']['bool']['filter'] = {
            "term": {
                "course": filter
            }
        }

    es_response = es_client.search(index=index_name, body=search_query)
    es_result_docs = []
    
    for hit in es_response['hits']['hits']:
        es_result_docs.append(hit['_source'])
        
        # Homework: Q3 - Searching: score for top record
        print(f"Score: Filter:[{filter}] - {hit['_score']}\n")
    
    return es_result_docs

query = "How do I execute a command in a running docker container?"
filter = "machine-learning-zoomcamp"

results_no_filter = elastic_search(query)

# Homework: Q4 - Filtering
results_filter = elastic_search(query, filter)

# %%
# Building a prompt

def build_prompt(query, results):
    
    context = ""
    context_template = """
        Q: {question}
        A: {text} \n\n
        """.strip()
        
    prompt_template = """
        You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
        Use only the facts from the CONTEXT when answering the QUESTION.

        QUESTION: {question}

        CONTEXT:
        {context}
        """.strip()
        
    for doc in results:
        context += context_template.format(question=doc['question'], text=doc['text'])
        
    prompt = prompt_template.format(question=query, context=context).strip()
    
    # Homework: Q5 - Prompt length
    print(f"Prompt length: {len(prompt)}")
    
    return prompt

built_prompt = build_prompt(query, results_filter)

# %%
# Calculating tokens for the prompt

encoding = tiktoken.encoding_for_model("gpt-4o")
tokens = encoding.encode(built_prompt)

# Homework: Q6 - Tokens
print(f"Tokens: {len(tokens)}")


# %% Bonus: Generating the answer

# Load Mistral api key and initialize the client
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Load environment variables from .env file and create Mistral client
load_dotenv(dotenv_path='.env')
api_key = os.environ["MISTRAL_API_KEY"]
model_small = "mistral-small-latest"

client = MistralClient(api_key=api_key)

def llm(prompt):
    # Generate the response
    response = client.chat(
        model=model_small,
        messages=[ChatMessage(role="user", content=prompt)]
    )
    
    return response.choices[0].message.content

def generate_answer(query, results):
    prompt = build_prompt(query, results)
    answer = llm(prompt)
    
    return answer

print(f"Question: {query}\nAnswer: {generate_answer(query, results_filter)}")

#%% [markdown]
# Some other useful links:
# - [How to Use Elasticsearch in Python](https://dylancastillo.co/elasticsearch-python/)


