# %% [markdown]
# ### Homework 5: Orchestration
# #### Parse FAQ LLM

# %%
# ### Imports
import io
import requests
import docx
from tqdm.auto import tqdm
import hashlib
from datetime import datetime
from elasticsearch import Elasticsearch

# %%
# ### Prepare methods to read the FAQ document
def clean_line(line):
    line = line.strip()
    line = line.strip('\uFEFF')
    return line

def read_faq(file_id):
    url = f'https://docs.google.com/document/d/{file_id}/export?format=docx'
    
    response = requests.get(url)
    response.raise_for_status()
    
    with io.BytesIO(response.content) as f_in:
        doc = docx.Document(f_in)

    questions = []

    question_heading_style = 'heading 2'
    section_heading_style = 'heading 1'
    
    heading_id = ''
    section_title = ''
    question_title = ''
    answer_text_so_far = ''
     
    for p in doc.paragraphs:
        style = p.style.name.lower()
        p_text = clean_line(p.text)
    
        if len(p_text) == 0:
            continue
    
        if style == section_heading_style:
            section_title = p_text
            continue
    
        if style == question_heading_style:
            answer_text_so_far = answer_text_so_far.strip()
            if answer_text_so_far != '' and section_title != '' and question_title != '':
                questions.append({
                    'text': answer_text_so_far,
                    'section': section_title,
                    'question': question_title,
                })
                answer_text_so_far = ''
    
            question_title = p_text
            continue
        
        answer_text_so_far += '\n' + p_text
    
    answer_text_so_far = answer_text_so_far.strip()
    if answer_text_so_far != '' and section_title != '' and question_title != '':
        questions.append({
            'text': answer_text_so_far,
            'section': section_title,
            'question': question_title,
        })

    return questions

# %%
# ### FAQ Documents
faq_documents = {
    'llm-zoomcamp': '1qZjwHkvP0lXHiE4zdbWyUXSVfmVGzougDD6N37bat3E',
}


# %% [markdown]
# ### Q2. Reading the documents
data = []

for course, file_id in tqdm(faq_documents.items()):
    print(course)
    course_documents = read_faq(file_id)
    data.append({'course': course, 'documents': course_documents})

# %% [markdown]
# ### Mage Pipeline - Ingest
# ![Pipeline - ingest](https://github.com/hadagarcia/llm-zoomcamp/blob/main/images/Module5/Q2_ReadingDocuments.png)

# %% [markdown]
# #### Transform data

# %%
# Method definitions
def generate_document_id(doc):
    combined = f"{doc['course']}-{doc['question']}-{doc['text'][:10]}"
    hash_object = hashlib.md5(combined.encode())
    hash_hex = hash_object.hexdigest()
    document_id = hash_hex[:8]
    return document_id

# %% [markdown]
# ### Q3. Transform data
transformed_documents = []
print(type(data))

# %%      
for course_dict in data:
    for doc in tqdm(course_dict['documents']):
        doc['course'] = course_dict['course']
        # previously we used just "id" for document ID
        doc['document_id'] = generate_document_id(doc)
        transformed_documents.append(doc)

print(f'Chunking: number of questions: {len(transformed_documents)}')
print(f'Chunking: example question: {transformed_documents[0]}')

# In Mage didn't need to add the extra loop, data was a dictionary already

# %% [markdown]
# ### Mage Pipeline - Chunking
# ![Pipeline - ingest](https://github.com/hadagarcia/llm-zoomcamp/blob/main/images/Module5/Q3_ChunkingDocuments.png)

# %% [markdown]
# ### Q4. Export

# %%
# Create Elasticsearch client
es_client = Elasticsearch('http://localhost:9200')
es_client.info()

# %%
index_name_prefix = 'documents'
current_time = datetime.now().strftime("%Y%m%d_%M%S")
index_name = f"{index_name_prefix}_{current_time}"
print(f'index name:  {index_name}')

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
            "document_id": {"type": "keyword"}
        }
    }
}

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=index_settings)

# %%
# Initialize a variable to store the last document
last_document = None

for document in tqdm(transformed_documents):
    # Update the last_document variable
    last_document = document

    es_client.index(index=index_name, document=document)
    
# Print the last document after the loop
if last_document:
    print(f'Last document indexed: {last_document["document_id"]}')

# %% [markdown]
# ### Mage Pipeline - Export
# ![Pipeline - ingest](https://github.com/hadagarcia/llm-zoomcamp/blob/main/images/Module5/Q4_ExportToElasticsearch.png)
