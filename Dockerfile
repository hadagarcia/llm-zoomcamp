# Use the ollama base image
FROM ollama/ollama

# Copy the weights directory from the host to the container
COPY ollama_files/models /root/.ollama/models