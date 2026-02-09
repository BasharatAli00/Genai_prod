from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel, RunnableBranch
from pydantic import BaseModel,Field
from typing import Literal
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda


from dotenv import load_dotenv
import os

load_dotenv()
class feedback(BaseModel):
    sentiment:Literal["positive","negative"]=Field(description="sentiment of the given text")

model1 = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv('DEEPSEEK_API_KEY'), # Or set DEEPSEEK_API_KEY environment variable
    temperature=0,
    max_tokens=500,
    timeout=None,
    max_retries=2,
)
parser=StrOutputParser()
  
promt1=PromptTemplate(
    template="Generate a  joke on this {topic}",
    input_variables=["topic"]
)
promt2=PromptTemplate(
    template="genrate a two line poem on this  \n{topic}",
    input_variables=["topic"]
)

parallel_chain=RunnableParallel({
    "joke":promt1|model1|parser,
    "poem":promt2|model1|parser
})

result=parallel_chain.invoke({"topic":"cricket"})
print(result.get("joke"))
print(result.get("poem"))
