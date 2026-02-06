from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv
import os

load_dotenv()

model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv('DEEPSEEK_API_KEY'), # Or set DEEPSEEK_API_KEY environment variable
    temperature=0,
    max_tokens=500,
    timeout=None,
    max_retries=2,
)

endpoint = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    max_new_tokens=128,
    temperature=0.7,
)

model1 = ChatHuggingFace(llm=endpoint)
template=PromptTemplate(
    template="give me the scientific name of this animal {frog}",
    input_variables=["frog"]
)

parser=StrOutputParser()

chain=template|model|parser
result=chain.invoke({"frog":"frog"})
print(result)