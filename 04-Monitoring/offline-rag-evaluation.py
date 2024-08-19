#%% [markdown]
### Offline RAG Evaluation

# %% [markdown]
# ### Load the data

import requests 

base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
relative_url = '03-vector-search/eval/documents-with-ids.json'
docs_url = f'{base_url}/{relative_url}?raw=1'
docs_response = requests.get(docs_url)
documents = docs_response.json()

# %%
documents[10]
# %% [markdown]
# ### Load the ground truth csv data

# %%
import pandas as pd

# %%
base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
relative_url = '03-vector-search/eval/ground-truth-data.csv'
ground_truth_url = f'{base_url}/{relative_url}?raw=1'

# %%
df_ground_truth = pd.read_csv(ground_truth_url)
df_ground_truth = df_ground_truth[df_ground_truth.course == 'machine-learning-zoomcamp']
ground_truth = df_ground_truth.to_dict(orient='records')

# %%
ground_truth[10]

# %%
doc_idx = {d['id']: d for d in documents}
doc_idx['5170565b']['text']

# %% [markdown]
# ### Index data and create embeddings using pretrained model

# %%
from sentence_transformers import SentenceTransformer

# %%
model_name = 'multi-qa-MiniLM-L6-cos-v1'
model = SentenceTransformer(model_name)

# %%
from elasticsearch import Elasticsearch

es_client = Elasticsearch('http://localhost:9200') 

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
            "course": {"type": "keyword"},
            "id": {"type": "keyword"},
            "question_text_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },
        }
    }
}

index_name = "course-questions"

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=index_settings)

# %%
from tqdm.auto import tqdm

# %%
# Index data and create embeddings using pretrained model
for doc in tqdm(documents):
    question = doc['question']
    text = doc['text']
    doc['question_text_vector'] = model.encode(question + ' ' + text)

    es_client.index(index=index_name, document=doc)

# %% [markdown]
# ### Retrieval using Elasticsearch

def elastic_search_knn(field, vector, course):
    # k-nearest neighbors (k-NN) search query
    knn = {
        "field": field,
        "query_vector": vector,
        "k": 3, #I'll use 3 for now instead of 5
        "num_candidates": 10000,
        "filter": {
            "term": {
                "course": course
            }
        }
    }
    
    search_query = {
        "knn": knn,
        "_source": ['text', 'question', 'section', 'id', 'course']
    }
    
    es_results = es_client.search(
        index=index_name,
        body=search_query
    )
    
    result_docs = []
    
    for hit in es_results['hits']['hits']:
        result_docs.append(hit['_source'])
        
    return result_docs


# %%
# Define the function to search for the closest question_text_vector to the query vector
def question_text_vector_knn(q):
    question = q['question']
    course = q['course']
    v_q = model.encode(question)
    
    return elastic_search_knn('question_text_vector', v_q, course)

# %% [markdown]
# #### Let's test the question_text_vector_knn function
question_text_vector_knn(dict(
    question='Are sessions recorded if I miss one?',
    course='machine-learning-zoomcamp'
))

# %% [markdown]
# ### The RAG flow
from openai import OpenAI

# %% [markdown]
# #### Define the openAI client using Ollama Container
client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama'
)

# %% [markdown]
# Let's build the prompt
def build_prompt(query, search_results):
    prompt_template = """
    You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
    Use only the facts from the CONTEXT when answering the QUESTION.

    QUESTION: {question}

    CONTEXT: 
    {context}
    """.strip()
    
    context = ''
    
    for doc in search_results:
        context += f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    print(f'built prompt: {prompt}')
    return prompt

# %% [markdown]
# #### Let's define the LLM function - Using the newest Llama 3.1 model
def llm(prompt, model='llama3.1:latest'):
    response = client.chat.completions.create(
        model = model,
        messages = [
            {'role': 'user', 'content': prompt}
        ]
    )
    return response.choices[0].message.content

# %% [markdown]
# #### Let's define the RAG function

def rag(query: dict, model='llama3.1:latest') -> str:
    search_results = question_text_vector_knn(query)
    prompt = build_prompt(query['question'], search_results)
    answer = llm(prompt, model)    

    return answer

# %% [markdown]
# #### Let's test the RAG function with the ground truth data

# %%
ground_truth[10]

# %%
rag(ground_truth[10])
# %%
# Comparing to the original answer
doc_idx['5170565b']['text']

# %% [markdown]
# ## Cosine Similarity metric

# %%
answer_original = 'Everything is recorded, so you won’t miss anything. You will be able to ask your questions for office hours in advance and we will cover them during the live stream. Also, you can always ask questions in Slack.'
answer_llm = 'Yes, sessions are recorded. Everything is recorded, so if you miss a session, you can catch up by watching the recording later. Additionally, office hours - live sessions where questions will be answered - are also recorded.'

v_original = model.encode(answer_original)
v_llm = model.encode(answer_llm)

v_llm.dot(v_original)

# %% [markdown]
# > **Note:**
# The part where the gpt-4o model is evaluated, I can skip it, since our instructor
# has already provided the results. So, I'll get the evaluation results from a csv file.


# %% [markdown]
# #### But let's try to evaluate the Llama 3.1 8b model using the ground truth data and ThreadPoolExecutor to speed up the process

# %% 
from tqdm.auto import tqdm

from concurrent.futures import ThreadPoolExecutor

pool = ThreadPoolExecutor(max_workers=6)

def map_progress(pool, seq, f):
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        for future in futures:
            result = future.result()
            results.append(result)

    return results

def process_record(rec):
    # model = 'gpt-3.5-turbo'
    # using the newest Llama 3.1 model instead
    answer_llm = rag(rec)
    
    doc_id = rec['document']
    original_doc = doc_idx[doc_id]
    answer_orig = original_doc['text']

    return {
        'answer_llm': answer_llm,
        'answer_orig': answer_orig,
        'document': doc_id,
        'question': rec['question'],
        'course': rec['course'],
    }

# %%
process_record(ground_truth[10])
# %%
# It's too slow... I need to find out first how to run in GPU
# results_llama318b = map_progress(pool, ground_truth, process_record)

# %% [markdown]
# ### Cosine similarity
# A->Q->A' cosine similarity
#
# A -> Q -> A'
#
# cosine(A, A')
# ### gpt-4o

# %% [markdown]
import pandas as pd

base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
relative_url = '04-monitoring/data/results-gpt4o.csv'
gpt_4o_results_url = f'{base_url}/{relative_url}?raw=1'

df_4o_results = pd.read_csv(gpt_4o_results_url)
df_4o_results.head()

# %%
df_4o_results = df_4o_results[df_4o_results.course == 'machine-learning-zoomcamp']
results_gpt4o = df_4o_results.to_dict(orient='records')

# %% [markdown]
# #### Compute the cosine similarity between the original and generated answers

# %%
def compute_similarity(record):
    answer_orig = record['answer_orig']
    answer_llm = record['answer_llm']
    
    v_llm = model.encode(answer_llm)
    v_orig = model.encode(answer_orig)
    
    return v_llm.dot(v_orig)
# %% [markdown]
# #### Similarity 
# %%
similarity = []

for record in tqdm(results_gpt4o):
    sim = compute_similarity(record)
    similarity.append(sim)
    
# %%
df_4o_results['cosine'] = similarity
df_4o_results['cosine'].describe()

# %%
import seaborn as sns

# %% [markdown]
# ### gpt-3.5-turbo

# %%
base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
relative_url = '04-monitoring/data/results-gpt35.csv'
gpt_35_results_url = f'{base_url}/{relative_url}?raw=1'

df_35_results = pd.read_csv(gpt_35_results_url)
df_35_results.head()

# %%
df_35_results = df_35_results[df_35_results.course == 'machine-learning-zoomcamp']
results_gpt35 = df_35_results.to_dict(orient='records')

# %%
similarity_35 = []

for record in tqdm(results_gpt35):
    sim = compute_similarity(record)
    similarity_35.append(sim)
    
# %%
df_35_results['cosine'] = similarity_35
df_35_results['cosine'].describe()

# %% [markdown]
# ### Let's graph the cosine similarity for the 2 models.
import matplotlib.pyplot as plt

# %%
sns.displot(df_4o_results['cosine'], label='gpt-4o')
sns.displot(df_35_results['cosine'], label='gpt-3.5-turbo')

plt.title('RAG LLM performance')
plt.xlabel("A->Q->A' Cosine Similarity")
plt.legend()

# %%
