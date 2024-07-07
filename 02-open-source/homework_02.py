#%% [markdown]
# # Homework 02: Using Ollama with OpenAI's API.

#%% [markdown]
# Q1 - Running Ollama with Docker
# - I run the docker-compose.yaml file with the command `podman compose --file docker-compose.yaml up --detach`
#     I used Podman instead of Docker, but the commands are similar. The podman compose command is used for multi-container.
#     in this case I'm running Ollama and Elasticsearch.
# - To check the Ollama version I run the command "ollama -v" in the terminal in Podman terminal.

#%% [markdown]
# Q2 - Downloading an LLM
# - I run the command `ollama pull gemma:2b` to download the LLM model from the Podman terminal.
# - In the Podman terminal searched for the downloaded model and verified the folder structure.
# - Found the metadata in the folder `models/manifests/registry.ollama.ai/library/gemma/2b`.

#%%
# Define OpenAI API client
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)

#%% [markdown]
# Q3 - Running the LLM
query = "10 * 10"

def llm_gemma(prompt):
    response = client.chat.completions.create(
        model='gemma:2b',
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

llm_gemma(query)

#%% [markdown]
# Q4 - Downloading the weights
# - Modified the docker-compose.yaml file to change the volume to the local folder.
# - Opened a Podman terminal to run these commands.
# - Run the command `podman compose --file docker-compose.yaml up --detach` to start the container.
# - Run the command `ollama pull gemma:2b` to download the weights to the local folder.
# - Run the command `du -h` to check the size of the local 'ollama_files/models/' folder.
# - See images: 
#      images\Q4_DownloadingWeightsUsingDockerComposeYamlFile.png
#      images\Q4_DownloadingWeights_Size.png

#%% [markdown]
# Q5 - Adding the weights
# - Created the Dockerfile (C:\MisProyectos\Road to AI Projects\llm-zoomcamp\Dockerfile)

#%% [markdown]
# Q6 - Serving the ollama-gemma2b image with Podman (Docker)
# - I used the user interfice provided by Podman to build and run the image.
#    See images: 
#       images\BuildingImageFileWithLocalWeights.png
#       images\Image_ollama-gemma2b_ImageSuccess.png

prompt = "What's the formula for energy?"

def llm_gemma_from_image(prompt):
    response = client.chat.completions.create(
        model='gemma:2b',
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    # return response.choices[0].message.content
    return response # I need to see the full response to check the tokens

#%%
# Test the new image
print(f'Full Response: {llm_gemma_from_image(prompt)}')

# How many completion tokens did you get in response?
# usage=CompletionUsage(completion_tokens=304, prompt_tokens=0, total_tokens=304)
