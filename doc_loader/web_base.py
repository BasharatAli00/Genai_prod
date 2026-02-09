import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate   

from dotenv import load_dotenv
import os

load_dotenv()


url = 'https://docs.docker.com/get-started/docker-overview/'

# Define a common browser User-Agent
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
}

# Initialize the loader with headers
loader = WebBaseLoader(
    web_path=url,
    # header_template=headers
)

doc = loader.load()
doc=doc[0].page_content

model=ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv('DEEPSEEK_API_KEY'), # Or set DEEPSEEK_API_KEY environment variable
    temperature=0,
    max_tokens=500,
    timeout=None,
    max_retries=2,
)

parser=StrOutputParser()
promt=PromptTemplate(
    template="anser the following question {question}  using the following documents {documents}",
    input_variables=["question","documents"]
)
chain=promt|model|parser
result =chain.invoke({"question":"what is docker?","documents":doc})
print(result)


# print(f"Documents loaded: {len(doc)}")
# print(doc[0].page_content) # Printing first 500 chars to verify