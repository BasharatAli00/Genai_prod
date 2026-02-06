from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel


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

template=PromptTemplate(
    template="write a precise and brief note on this {topic}",
    input_variables=["topic"]
)
template2=PromptTemplate(
    template="give me the answer questions on this text {topic}",
    
    input_variables=["topic"]
)


template3=PromptTemplate(
    template="merge the the notes and the quiz into single document {text} and {quiz}",
    input_variables=["text","quiz"]
)
parser=StrOutputParser()

parrallel_chain=RunnableParallel({'text':template | model2 | parser, 
         'quiz':template2 | model1 | parser})


note = """Generative AI transforms how we create content by enabling machines to produce original text, images, and code.
It relies on large language models trained on vast datasets to understand context and nuance.
This technology is revolutionizing industries from software development to creative arts."""

merge_chain=template3 | model1 | parser

final_chain=parrallel_chain | merge_chain

result=final_chain.invoke({"topic":note})
print(result)
final_chain.get_graph().print_ascii()

