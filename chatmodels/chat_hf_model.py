from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

endpoint = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    max_new_tokens=128,
    temperature=0.7,
)

chat = ChatHuggingFace(llm=endpoint)

print(chat.invoke("What is the biggest animal in the world?").content)
