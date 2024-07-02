#%% [markdown]
# # Using Ollama with OpenAI's API. 
# Llama3 model is running in docker container on localhost:11434, so it can be used locally.

#%%
# # Importing the libraries
import requests, json, os

#%%
# Get the minsearch library
url = 'https://raw.githubusercontent.com/alexeygrigorev/minsearch/main/minsearch.py'
r = requests.get(url)
# Assuming you want to save the content of the URL to a file named 'minsearch.py' in the current directory
with open('minsearch.py', 'wb') as f:
   f.write(r.content)

# %%
# Download the data
import minsearch

docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

index = minsearch.Index(
    text_fields=["question", "text", "section"],
    keyword_fields=["course"]
)

index.fit(documents)

# %%
# Define the search query
def search(query):
    boost = {'question': 3.0, 'section': 0.5}

    results = index.search(
        query=query,
        filter_dict={'course': 'data-engineering-zoomcamp'},
        boost_dict=boost,
        num_results=5
    )

    return results

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
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)

#%%
# Define the LLM api call

def llm_llama3(prompt):
    response = client.chat.completions.create(
        model='llama3',
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

def llm_phi3(prompt):
    response = client.chat.completions.create(
        model='phi3',
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content
# %%
# Test the LLM method
llm_llama3('write that this is a test')

# %%
# Ask Llama3 our question without finetuning.
llm_llama3('I just found out about this course, can I still join?')

#%%
# Define the RAG method
def rag(query):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm_llama3(prompt)
    return answer

#%%
# Test the RAG method
query = "I just found out about this course, can I still join?"
print(f"Answer from Llama3 with finetuning: {rag(query)}")