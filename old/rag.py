# pip install -q langchain langchain-groq langchain-community sentence-transformers chromadb pandas openpyxl python-dotenv langchain-huggingface langchain-chroma
# Note: you may need to restart the kernel to use updated packages.
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env
load_dotenv(override=True)

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("WARNING: GROQ_API_KEY not found in .env file. Please set it.")
else:
    print("Groq API Key found.")

huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
if not huggingface_api_key:
    print("WARNING: HUGGINGFACE_API_KEY not found in .env file. Please set it.")
else:
    print("Huggingface API Key found.")
# Groq API Key found.
# Huggingface API Key found.
# 1. Load Data
file_path = "feedback.xlsx"

if os.path.exists(file_path):
    df = pd.read_excel(file_path)
    print("Data loaded. First 5 rows:")
    # display(df.head())
else:
    print(f"File not found: {file_path}. Please make sure the file is in the same directory.")
# Data loaded. First 5 rows:
# z_last_digits	student_writeup
# 0	607947	As i mentioned the first class, I'm an archite...
# 1	571257	I expect to learn basic ideas of Artificial In...
# 2	607605	I will develop high practical skills in data a...
# 3	236137	My primary goal for this course is to develop ...
# 4	561111	I want to gain practical knowledge on how to i...
# 2. Prepare Vector Database
# Convert DataFrame rows to Documents
documents = []

# Iterate over rows and convert to text format
for index, row in df.iterrows():
    # Convert the entire row to a string representation
    content = "\n".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
    
    doc = Document(page_content=content, metadata={"row": index, "source": file_path})
    documents.append(doc)

print(f"Processed {len(documents)} documents.")
# Processed 37 documents.
# Create a Vector Database
# Initialize Embeddings (using a model from HuggingFace)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create ChromaDB Vector Store

"""This creates a temporary in-memory store. The memory will be cleared when the kernel is restarted, 
if persist_directory is not specified."""

"""The from_documents method requires a list of document objects along with the 
embedding model and the name of the collection as parameters, where collection_name is optional."""

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="student_feedback",
    persist_directory="./chroma_db"
)

print("Vector Database created.")
# Vector Database created.
# 3. Setup an Agent using LangChain
# This function will get the documents from the vector store
@tool
def get_all_feedback() -> str:
    """Get all student feedback to analyze topics and themes"""
    docs = vectorstore.get()['documents']
    if isinstance(docs, list) and len(docs) > 0:
        response_content = "\n\n---\n\n".join(docs[:20])  # Get first 20 docs
    else:
        # Fallback: search with broad query
        docs = vectorstore.similarity_search("learning goals interests", k=30)
        response_content = "\n\n---\n\n".join([d.page_content for d in docs])
    return f"Total feedback entries: {len(docs)}\n\n{response_content}"

@tool  
def search_feedback(query: str) -> str:
    """Search for specific feedback on a topic"""
    docs = vectorstore.similarity_search(query, k=10)
    response_content = "\n\n".join([d.page_content for d in docs])
    return response_content
# Create the Agent

# Initialize the LLM
llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct", api_key=groq_api_key)

# Initialize the tools
tools = [get_all_feedback, search_feedback]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can answer questions about student feedback."),
    ("placeholder", "{agent_scratchpad}"),
    ("human", "{input}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({
    "input": "Using the complete student feedback data, provide: 1) A summary of the main topics and interests students have about this course, 2) Identify and list the top 5 most common themes/topics students mentioned, 3) For each topic, briefly explain what students want to learn"
}, verbose=True)
final_answer = result['output']

print(final_answer)