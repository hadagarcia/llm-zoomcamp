#%% [markdown]
### Homework RAG Evaluation

# %% [markdown]
# ### Load the data

import pandas as pd

base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
relative_url = '04-monitoring/data/results-gpt4o-mini.csv'
url = f'{base_url}/{relative_url}?raw=1'
df = pd.read_csv(url)

# %%
# We'll use only the first 300 rows

df = df.iloc[:300]

# %% [markdown]
# #### Q1. Getting the embeddings model

# %%
model_name = 'multi-qa-mpnet-base-dot-v1'

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer(model_name)


# %%
# Create embeddings for the first LLM answer:
answer_llm = df['answer_llm'].iloc[0]
embedding_llm = embedding_model.encode(answer_llm)
# %% [markdown]
# #### Embedding for the first LLM answer:
embedding_llm[0]

# %% [markdown]
# #### Q2. Computing the dot product

# %%
# Create embeddings for each answer pair (answer_llm, answer_orig) and compute
# the dot product between them.

def compute_doc_product(record):
    answer_orig = record['answer_orig']
    answer_llm = record['answer_llm']
    
    v_llm = embedding_model.encode(answer_llm)
    v_orig = embedding_model.encode(answer_orig)
    
    return v_llm.dot(v_orig)

evaluations = []
results_gpt4o_mini = df.to_dict(orient='records')
from tqdm.auto import tqdm

# %%
for record in tqdm(results_gpt4o_mini):
    dot = compute_doc_product(record)
    evaluations.append(dot)
    
# %%
df['dot'] = evaluations
df['dot'].describe()
# %%
