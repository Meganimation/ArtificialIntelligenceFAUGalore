# pip install -q langchain langchain-ollama
# In one terminal, start server:
# ollama serve
# In another terminal, run your bot:
# python3 offlineRobot.py
# Note: you may need to restart the kernel to use updated packages.
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# This script runs entirely locally - no API keys needed
# Make sure you have Ollama installed and running locally
print("Using fully private local models - no external API calls")

# Initialize the LLM - using local Ollama (fully private, no external API)
llm = ChatOllama(
    model="llama3.2",  # You can change this to any model you have pulled in Ollama
    temperature=0.7
)

print("\n" + "="*60)
print("Local Private Chatbot")
print("="*60)
print("Chat with a locally-running AI model.")
print("Type 'exit', 'quit', or 'q' to end the session.")
print("="*60 + "\n")

# Store conversation history
conversation_history = []

while True:
    user_input = input("\nYou: ").strip()
    
    if user_input.lower() in ['exit', 'quit', 'q']:
        print("\nGoodbye!")
        break
    
    if not user_input:
        print("Please enter a message.")
        continue
    
    try:
        # Add user message to history
        conversation_history.append(HumanMessage(content=user_input))
        
        # Get response from the model
        response = llm.invoke(conversation_history)
        
        # Add AI response to history
        conversation_history.append(AIMessage(content=response.content))
        
        print(f"\nAssistant: {response.content}")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Please try again.")