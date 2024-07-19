#%% [markdown]
# # Homework 3 - Vector search with Elasticsearch

#%% [markdown]
# ## Q1: Getting the embeddings model
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')

# %% [markdown]
# Create embeddings for this user question
user_question = "I just discovered the course. Can I still join it?"
embedded_user_question = embedding_model.encode(user_question)

# %% [markdown]
# ## First value in resulting vector
print(embedded_user_question[0])

# %% [markdown]
# ## Prepare the documents
import requests 

base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
relative_url = '03-vector-search/eval/documents-with-ids.json'
docs_url = f'{base_url}/{relative_url}?raw=1'
docs_response = requests.get(docs_url)
documents = docs_response.json()

# %% [markdown]
# ### Filter documents by course
subset_documents =  [doc for doc in documents if doc['course'] == 'machine-learning-zoomcamp']
print(len(subset_documents))

# %% [markdown]
# ## Q2: Create embeddings for the documents

embeddings = []

for doc in subset_documents:
    qa_text = f'{doc['question']} {doc['text']}'
    embeddings.append(embedding_model.encode(qa_text))

# %%
import numpy as np
X = np.array(embeddings)

# %% [markdown]
# ### What's the shape of X?
print(X.shape)

# %% [markdown]
# ## Q3: Search - Highest score

# Make sure X is normalized
# Check by computing a dot product of a vector with itself, should return something close to 1.0
print(np.dot(X[0], X[0]))

# %% [markdown]
# ### Cosine Similarity by dot product of X and v (user question)
v = np.array(embedded_user_question)
print(v.shape) # Single vector: (768,)

# %% [markdown]
# ### What's the highest score in the results?
scores = X.dot(v) # Cosine similarity

highest_score = np.max(scores)
highest_score_index = np.argmax(scores)

print(f"Highest score is {highest_score} at index {highest_score_index}")

# %% [markdown]
# ### Vector Search

# %%
# Let's implement our own vector search
class VectorSearchEngine():
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings

    def search(self, v_query, num_results=10):
        scores = self.embeddings.dot(v_query) # Cosine similarity
        highest_score = np.max(scores)
        idx = np.argsort(-scores)[:num_results] # Sort in descending order and get top num_results
        return [self.documents[i] for i in idx], highest_score # Return the documents with the highest scores

# %%
search_engine = VectorSearchEngine(documents=subset_documents, embeddings=X)
top_5 = search_engine.search(v, num_results=5)

# %% [markdown]
# ### Top 5 documents
for doc in top_5:
    print(f'{doc}\n')

# %% [markdown]
# ## Q4: Hit-rate for our search engine

# First load the ground truth dataset (csv)
import pandas as pd

base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
relative_url = '03-vector-search/eval/ground-truth-data.csv'
ground_truth_url = f'{base_url}/{relative_url}?raw=1'

df_ground_truth = pd.read_csv(ground_truth_url)
df_ground_truth = df_ground_truth[df_ground_truth.course == 'machine-learning-zoomcamp']
ground_truth = df_ground_truth.to_dict(orient='records')

# > **Note:**
# What is ground_truth data? it has sets of 5 questions generated for each document in documents-with-ids.json
# this way we can test the search engine by comparing the results with the ground truth
# The ground truth data sets of 5 questions, were generated with an LLM.

print(ground_truth[0])

# %% [markdown]
# ## Hit-rate for ground truth

# %% [markdown]
# Hit Rate function definition
def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)

# %%
relevance_total = []

from tqdm import tqdm
for q in tqdm(ground_truth):
    doc_id = q['document']
    # using VectorSearchEngine instead elastic_search
    embedded_question = embedding_model.encode(q['question'])
    results = search_engine.search(embedded_question, num_results=5) 
    relevance = [d['id'] == doc_id for d in results]
    relevance_total.append(relevance)

# %% [markdown]
print(f'Hit rate for VectorSearch engine: {hit_rate(relevance_total)}')

# %% [markdown]
# ## Q5: Indexing with Elasticsearch

# %% [markdown]
# Setup Elasticsearch connection
from elasticsearch import Elasticsearch

es_client = Elasticsearch('http://localhost:9200')
es_client.info()


# %%
# Create index
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
            "course": {"type": "keyword"},
            "id": {"type": "keyword"},
            "question_vector": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine"
            },
            "text_vector": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine"
            },
            "question_text_vector": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine"
            },
        }
    }
}

index_name = "course-questions"

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=index_settings)

#%% [markdown]
# ### Adding embeddings to subset_documents
for doc in tqdm(subset_documents):
    question = doc['question']
    text = doc['text']
    qt = question + ' ' + text

    doc['question_vector'] = embedding_model.encode(question)
    doc['text_vector'] = embedding_model.encode(text)
    doc['question_text_vector'] = embedding_model.encode(qt)

# %% [markdown]
# ### Indexing the documents
for doc in tqdm(subset_documents):
    es_client.index(index=index_name, document=doc)
    
 
# %% [markdown]
# Defining elasticsearch query
def elastic_search_knn(field, vector, course):
    knn = {
        "field": field,
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000,
        "filter": {
            "term": {
                "course": course
            }
        }
    }

    search_query = {
        "knn": knn,
        "_source": ["text", "section", "question", "course", "id"]
    }

    es_results = es_client.search(
        index=index_name,
        body=search_query
    )
    
    result_docs = []
    
    for hit in es_results['hits']['hits']:
        result_docs.append(hit['_source'])
        
        # print(f"Score:  {hit['_score']} - Id: {hit['_source']['id']}\n")

    return result_docs

# %% [markdown]
# ### Search with Elasticsearch
search_knn_results = []
question = "I just discovered the course. Can I still join it?"
course="machine-learning-zoomcamp"
search_knn_results = elastic_search_knn('question_vector', embedding_model.encode(question), course)

# %% [markdown]
# ### Higher result score
for result in search_knn_results:
    if result['id'] == 'ee58a693':
        print(result)


#%% [markdown]
# ## Q6: Hit-rate for Elasticsearch

#%%
# Let's add embeddings to ground truth to avoid re-embedding
embeddings_ground_truth = ground_truth.copy()
for doc in embeddings_ground_truth:
    doc['question_vector'] = embedding_model.encode(doc['question'])
 

#%% [markdown]
# ### Search with Elasticsearch for ground truth
es_relevance_total = []

for q in tqdm(ground_truth):
    doc_id = q['document']

    # es_knn_results = elastic_search_knn('question_vector', embedding_model.encode(q['question']), 'machine-learning-zoomcamp')
    es_knn_results = elastic_search_knn('question_vector', q['question_vector'], 'machine-learning-zoomcamp') 
    es_knn_relevance = [d['id'] == doc_id for d in es_knn_results]
    es_relevance_total.append(es_knn_relevance)

# %% [markdown]
# Hit rate for Elasticsearch
print(f'Hit rate for Vector Elasticsearch engine: {hit_rate(es_relevance_total)}')



# %%
