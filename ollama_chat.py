import os
from pathlib import Path


ssl_cert_file = os.environ.get("SSL_CERT_FILE")
if ssl_cert_file and not Path(ssl_cert_file).is_file():
    os.environ.pop("SSL_CERT_FILE", None)

#-----------------Example of a simple chat with Ollama--------------------------------#
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize the model
chat = ChatOllama(model="llama3", temperature=0)

# Create a conversation
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Explain quantum computing in one sentence.")
]

response = chat.invoke(messages)
print(response.content)



#-----------------Example of Retrieval-Augmented Generation (RAG) with Ollama-----------------#
from langchain_ollama import OllamaLLM
# 2. Initialize Ollama with a local model (e.g., llama3, mistral)
# Temperature is set to 0 for deterministic output
model = OllamaLLM(model="llama3", temperature=0)

# 3. Define the prompt
prompt = "The sky is blue because of Rayleigh scattering. Explain this in one sentence."

# 4. Invoke the model
response = model.invoke(prompt)

# 5. Print the response
print(response)