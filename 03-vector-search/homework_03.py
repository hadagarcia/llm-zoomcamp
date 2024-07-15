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

# Let's implement our own vector search
class VectorSearchEngine():
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings

    def search(self, v_query, num_results=10):
        scores = self.embeddings.dot(v_query) # Cosine similarity
        idx = np.argsort(-scores)[:num_results] # Sort in descending order and get top num_results
        return [self.documents[i] for i in idx] # Return the documents with the highest scores

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

print(ground_truth[0])
# %%
