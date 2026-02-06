from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

# UPDATED: Using a model confirmed to exist in your list
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

response = llm.invoke("Write a five line poem on apple")
print(response.content)