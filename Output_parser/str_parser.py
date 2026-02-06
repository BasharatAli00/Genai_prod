from langchain_core.prompts import PromptTemplate
# from models_v3 import model  
from langchain_core.output_parsers import StrOutputParser 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
from typing import TypedDict,Annotated, Optional, Literal
from pydantic import BaseModel, Field
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


#hugging face models

endpoint = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    max_new_tokens=128,
    temperature=0.7,
)

chat = ChatHuggingFace(llm=endpoint)


template1=PromptTemplate(
    template="write a short note on this {topic}",
    input_variables=['topic']
)

template2=PromptTemplate(
    template='write a 3 lines on this text \n{text}',
    input_variables=['text']
)
promt1=template1.invoke({'topic': 'population'})
result1=chat.invoke(promt1)

promt2=template2.invoke({'text': result1.content})

result2=chat.invoke(promt2)
print(result2.content)  

# parser=StrOutputParser()

# chain=template1 | chat | parser | template2 | chat | parser   

# result=chain.invoke({'topic': 'population'})
# print(result)