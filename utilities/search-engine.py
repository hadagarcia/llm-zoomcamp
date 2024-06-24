#%% [markdown]
# # Build your own search engine

#%%
# Importing the libraries

import requests
import pandas as pd

# %%
# Download the data

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()
documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

# %%
# Creating the DataFrame        

df = pd.DataFrame(documents, columns=['course', 'section', 'question', 'text'])
df.head()

# %% [markdown]
# Vector spaces
# - Turn docs into vectors

# %% [markdown]
# Continue later ...
# https://github.com/alexeygrigorev/build-your-own-search-engine

# %% [markdown]
# # References
# - [Latent semantic analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis)
# - [Introduction to Information Retrieval book](https://nlp.stanford.edu/IR-book/information-retrieval-book.html)
# - [Building a Search Engine with LSA](https://towardsdatascience.com/building-a-search-engine-with-latent-semantic-analysis-12b6a4d6833d)
# - [Mining of Massive Datasets book](http://www.mmds.org/)
