from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import HumanMessage
import os

os.environ['HF_HOME'] = 'D:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 200,
    }
)

# Wrap the pipeline in ChatHuggingFace
chat_model = ChatHuggingFace(llm=llm)

content="how many planet are in the solar system?"

# This will now include the actual model response
result = chat_model.invoke(content)
print(result.content.strip())