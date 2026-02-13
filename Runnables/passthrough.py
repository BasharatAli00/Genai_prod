from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel, RunnableBranch

from langchain_core.runnables import RunnableLambda, RunnablePassthrough


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
parser=StrOutputParser()

promt1=PromptTemplate(
    template="Generate a cold email on this {topic}",
    input_variables=["topic"]
)
promt2=PromptTemplate(
    template="genrate a two line summary of this email  \n{topic}",
    input_variables=["topic"]
)

email_chain=promt1|model1|parser

passthrough_chain=RunnableParallel({
    'email': RunnablePassthrough(),
    'summary': promt2|model1|parser
})

final_chain=email_chain|passthrough_chain
result=final_chain.invoke({"topic":"ai engineer"})
print(result)







