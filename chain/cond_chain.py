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

#model2
endpoint = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    max_new_tokens=128,
    temperature=0.7,
)
model2 = ChatHuggingFace(llm=endpoint)


parser1=StrOutputParser()
parser2=PydanticOutputParser(pydantic_object=feedback)




#template1
template1=PromptTemplate(
    template="classify the sentiment of the given text {text} positive or negative\n {format_instructions}",
    input_variables=["text"],
    partial_variables={
        "format_instructions": parser2.get_format_instructions()
    }
)

template2=PromptTemplate(
    template="write a appropriate response for the given feedback {feedback}",
    input_variables=["feedback"]
)


template3=PromptTemplate(
    template="write a appropriate response for the given feedback {feedback}",
    input_variables=["feedback"]
)

classifier_chain=template1|model1|parser2


branch_chain=RunnableBranch(
    (lambda x: x.sentiment == "positive", template2|model1|parser1),
    (lambda x: x.sentiment == "negative", template3|model1|parser1),
    RunnableLambda(lambda x: "Invalid sentiment")   
)

main_chain=classifier_chain | branch_chain
result=main_chain.invoke({"text":"avatar is worst movie ever made "})
print(result)


