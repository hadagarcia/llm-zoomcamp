#%% [markdown]
# # # Retrieval and search (Rustic version)

#%%
# Importing the libraries
import requests

# %%
# Download the data
url = 'https://raw.githubusercontent.com/alexeygrigorev/minsearch/main/minsearch.py'
r = requests.get(url)
# Assuming you want to save the content of the URL to a file named 'minsearch.py' in the current directory
with open('minsearch.py', 'wb') as f:
   f.write(r.content)

url_data = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/01-intro/documents.json'
r_data = requests.get(url_data)

# Assuming you want to save the content of the URL to a file named 'minsearch.py' in the current directory
with open('documents.json', 'wb') as f:
    f.write(r_data.content)
    
# %%
# Importing the libraries and json data. Also index by text and keyword fields
import minsearch, json

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

question = "The course has already started, can I still enroll?"
boost = {'question': 3.0, 'section': 0.5} # To give proprity to certain fields

index.fit(documents)
results = index.search(
    query=question,
    filter_dict={"course": "data-engineering-zoomcamp"},
    boost_dict=boost,
    num_results=5
)

#%%
# Load Mistral api key 
from dotenv import load_dotenv
import os

# os.path.abspath('.env')

# Load environment variables from .env file
load_dotenv(dotenv_path='.env')

# Get the API key from the environment variable
# Access the API key
api_key = os.getenv('MISTRAL_API_KEY')

#%%
# Create a client and Test a message using Claude API
from anthropic import Anthropic
client = Anthropic(api_key=api_key)

message = client.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hola, Claude"}
    ]
)
print(message.content)

# %%
# Create a client using MistralAI API

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

api_key = os.environ["MISTRAL_API_KEY"]
model_large = "mistral-large-latest"
model_small = "mistral-small-latest"

client = MistralClient(api_key=api_key)

#%% Test the chat completion API
chat_response = client.chat(
    model=model_small,
    messages=[ChatMessage(role="user", content=question)]
)

print(chat_response.choices[0].message.content)

# %%
context = ""

for doc in results:
    context = context + f"\nsection: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
    
prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database. 
Use only the facts from the CONTEXT when answering the QUESTION.
If the CONTEXT doesn't contain the answer, output NONE

QUESTION: {question}
CONTEXT: {context}
""".strip()

# %%
prompt = prompt_template.format(question=question, context=context).strip()

# %%
# Finally put all together and send the promot to the API
chat_response = client.chat(
    model=model_small,
    messages=[ChatMessage(role="user", content=prompt)]
)

print(chat_response.choices[0].message.content)
# %%
