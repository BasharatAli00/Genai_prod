from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama

templates = ChatPromptTemplate.from_messages([
    ("system", "you are helpful {domain} expert"),
    MessagesPlaceholder(variable_name='chat_hist'),
    ("human", "{query}")
])
llm = ChatOllama(model="llama3.2:3b", temperature=0.7)

chat_hist = []
user_input=input("query")

# Logic to read the file and convert strings to Message Objects
with open('messgae.txt', 'r') as f:
    for line in f:
        content = line.strip()
        if not content:
            continue
            
        # If your file literally contains the text "HumanMessage(content=...)"
        # we need to extract the text inside the quotes.
        if "HumanMessage" in content:
            # Simple extraction: getting text between double quotes
            text = content.split('"')[1] 
            chat_hist.append(HumanMessage(content=text))
        elif "AIMessage" in content:
            text = content.split('"')[1]
            chat_hist.append(AIMessage(content=text))

# CRITICAL: Always ensure chat_hist is passed, even if empty []
promt = templates.invoke({
    'domain': 'customer care',
    'chat_hist': chat_hist, 
    'query': user_input
})
# user_input=("query")
response=llm.invoke(promt)
chat_hist.append(response.content)
print(response.content)
