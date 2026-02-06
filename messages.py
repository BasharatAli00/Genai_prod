from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# 1. Setup the model (Llama 3.2 3B is the perfect balance for 8GB RAM)
llm = ChatOllama(model="llama3.2:3b", temperature=0.7)

print("--- AI Chatbot Started (Type 'exit' to stop) ---")

# 2. Simple Chat Loop
messages = [
    SystemMessage(content="You are a helpful and funny AI assistant.")
]

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Add user message to history
    messages.append(HumanMessage(content=user_input))

    # Get AI response
    response = llm.invoke(messages)
    
    # Print and save AI response to history
    print(f"AI: {response.content}")
    messages.append(response)
