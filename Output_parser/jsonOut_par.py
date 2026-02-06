from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

endpoint = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    max_new_tokens=128,
    temperature=0.3,  # LOWER temperature helps JSON
)

chat = ChatHuggingFace(llm=endpoint)

parser = JsonOutputParser()

prompt = PromptTemplate(
    template="""
You must respond ONLY in valid JSON.

Write a 5-line fact  about {topic}

{format_instructions}
""",
    input_variables=['topic'],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

chain = prompt | chat | parser

result = chain.invoke({"topic": "allien"})
print(result)
