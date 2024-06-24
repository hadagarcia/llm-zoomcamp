#%% [markdown]
# # Retrieval and search (clean version)

#%%
# # Importing the libraries
import requests, json, os
from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# %%
# # Download the data
url = 'https://raw.githubusercontent.com/alexeygrigorev/minsearch/main/minsearch.py'
r = requests.get(url)

# Assuming you want to save the content of the URL to a file named 'minsearch.py' in the current directory
with open('minsearch.py', 'wb') as f:
   f.write(r.content)

# Import the minsearch module
import minsearch

url_data = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/01-intro/documents.json'
r_data = requests.get(url_data)

# Assuming you want to save the content of the URL to a file named 'minsearch.py' in the current directory
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

index = minsearch.Index(
    text_fields=["question", "text", "section"],
    keyword_fields=["course"]
)

index.fit(documents)

# %%
# # Load Mistral api key and initialize the client

# Load environment variables from .env file
load_dotenv(dotenv_path='.env')

# Get the API key from the environment variable
# Access the API key
api_key = os.environ["MISTRAL_API_KEY"]

model_large = "mistral-large-latest"
model_small = "mistral-small-latest"

client = MistralClient(api_key=api_key)

#%% 
# # Define methods

# Query the indexed data
def search(query):
    boost = {'question': 3.0, 'section': 0.5} # To give proprity to certain fields
    filter = {"course": "data-engineering-zoomcamp"}
    
    results = index.search(
        query=query,
        filter_dict=filter,
        boost_dict=boost,
        num_results=5
    )
    return results

# Build the prompt
def build_prompt(query, search_results):
    
    context = "";
    prompt_template = """
    You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database. 
    Use only the facts from the CONTEXT when answering the QUESTION.
    If the CONTEXT doesn't contain the answer, output NONE

    QUESTION: {question}
    CONTEXT: {context}
    """.strip()

    for doc in search_results:
        context = context + f"\nsection: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
    
    # Homework: Q4 - Filtering - 3rd document
    print(f"3rd doc: {search_results[2]}")
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def llm(prompt):
    
    # Generate the response
    response = client.chat(
    model=model_small,
    messages=[ChatMessage(role="user", content=prompt)]
    )
    
    return response.choices[0].message.content

# %%
# All together
def rag(query): 
    
    results = search(query)
    prompt = build_prompt(query, results)
    answer = llm(prompt)
    
    return answer

# %%
# Full RAG implementation
query = "The course has already started, can I still enroll?"
print(rag(query))

#%% [markdown]
# # Replacing MinSearch with ElasticSearch

#%%
from elasticsearch import Elasticsearch

es_client = Elasticsearch('http://localhost:9200')
print(es_client.info())

# %%
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

# %%
from tqdm.auto import tqdm
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

    es_response = es_client.search(index=index_name, body=search_query)
    es_result_docs = []
    
    for hit in es_response['hits']['hits']:
        es_result_docs.append(hit['_source'])
        
        # Homework: Q3 - Searching: score for top record
        # Assuming 'hit' is a dictionary containing the '_source' key, which is another dictionary containing the '_score' key
        print(f"Score:  {hit['_score']}\n")
    
    return es_result_docs


# %%

# Full RAG implementation using ElasticSearch
def rag_elastic_search(query): 
    
    results = elastic_search(query)
    prompt = build_prompt(query, results)
    
    # Homework: Q5 - Prompt length
    print(f"Prompt length: {len(prompt)}")
    
    answer = llm(prompt)
    
    return answer

print(rag_elastic_search(query))
# %%
