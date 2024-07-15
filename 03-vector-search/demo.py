#%% [markdown]
# # Vector search

#%%
# ## Prepare the data
import json
with open('../utilities/documents.json', 'rt') as f_in:
    docs_raw = json.load(f_in)
    
# %%
documents = []

for course_dict in docs_raw:
    for doc in course_dict['documents']:
        doc['course'] = course_dict['course']
        documents.append(doc)

documents[1]
# %%
# ## Create embeddings using pretrained model
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')

# %%
len(model.encode("This is a simple sentence"))

# %%
# ## Create dense vectors using pre-trained model

operations = []

for doc in documents:
    doc['text_vector'] = model.encode(doc['text']).tolist()
    operations.append(doc)
    
operations[0]

# %% [markdown]
# ## Setup Elasticsearch connection

from elasticsearch import Elasticsearch
es_client = Elasticsearch('http://localhost:9200') 

es_client.info()

# %% [markdown]
# ## Create mappings and index

#%% [markdown]
# ### First we need to verify all embeddings have the same dimension (optional)
all_dims = [len(doc['text_vector']) for doc in documents]
if all(dim == all_dims[0] for dim in all_dims):
    print("All embeddings have the same dimension:", all_dims[0])
else:
    print("There are embeddings with varying dimensions.")

#%% [markdown]
# ### Now that we know the dimension of the embeddings, we can create the index
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
            "course": {"type": "keyword"} ,
            "text_vector": {"type": "dense_vector", "dims": 768, "index": True, "similarity": "cosine"},
        }
    }
}

# %%
index_name = "course-questions"

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=index_settings)
# %%

#%% [markdown]
# ### Add documents to the index
for doc in operations:
    try:
        es_client.index(index=index_name, document=doc)
    except Exception as e:
        print(e)
# %% [markdown]
# ## Create end user query

# User's query also needs to be encoded
search_term = "windows or mac?"
vector_search_term = model.encode(search_term)

# %%
query = {
    "field": "text_vector",
    "query_vector": vector_search_term,
    "k": 5, # Find the top 5 documents that are closest to the query vector
    "num_candidates": 10000, 
}
# %%
# knn (K-nearest neighbors) search: Here specifies the vector query body
# source: specifies which fields from the matching documents should be returned

res = es_client.search(index=index_name, knn=query, source=["text", "section", "question", "course"])
res["hits"]["hits"]

# %% [markdown]
# Hits are not filtered. We could filter by section = "General course-related questions"
# Let's perform with a hybrid approach

# %%[markdown]
# ##  Perform Keyword search with Semantic Search (Hybrid/Advanced Search)

knn_query = {
    "field": "text_vector",
    "query_vector": vector_search_term,
    "k": 5,
    "num_candidates": 10000
}

response = es_client.search(
    index=index_name,
    query={
        "match": {"section": "General course-related questions"},
    },
    knn=knn_query,
    size=5,
    source=["text", "section", "question", "course"]
)

response["hits"]["hits"]
# %%
