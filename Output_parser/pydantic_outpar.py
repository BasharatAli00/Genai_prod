from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel,Field
import os


load_dotenv()

endpoint = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    max_new_tokens=128,
    temperature=0.3,  # LOWER temperature helps JSON
)




#deepseek

model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv('DEEPSEEK_API_KEY'), # Or set DEEPSEEK_API_KEY environment variable
    temperature=0,
    max_tokens=500,
    timeout=None,
    max_retries=2,
)



#huggingface

endpoint = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    max_new_tokens=128,
    temperature=0.3,  # LOWER temperature helps JSON
)

chat = ChatHuggingFace(llm=endpoint)
class person(BaseModel):
    name:str=Field(...,description="name of the person")
    age:int=Field(...,description="age of the person",gt=18)
    job:str=Field(...,description="job of the person")

parser=PydanticOutputParser(pydantic_object=person)

template=PromptTemplate(
    template="""
    You must respond ONLY in valid JSON.

    Give me the name of the person ,age and job of {person}

    {format_instructions}
    """,
    input_variables=['person'],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

chain=template|model|parser
final_result=chain.invoke({"person": "Lionel Messi"})
print(final_result)





