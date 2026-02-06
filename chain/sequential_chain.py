from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv
import os

load_dotenv()

model1 = ChatDeepSeek(
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
model2 = ChatHuggingFace(llm=endpoint)



#template
template1=PromptTemplate(
    template="write a precise and brief note on this {topic}",
    input_variables=["topic"]
)


#template2
template2=PromptTemplate(
    template="summarize the given text {text} and give a summrize in bullet points",
    input_variables=["text"]
)


parser=StrOutputParser()

chain=template1|model1|parser | template2|model2 | parser
result=chain.invoke({"topic":"Generative ai"})
chain.get_graph().print_ascii()