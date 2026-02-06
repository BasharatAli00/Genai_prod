from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from chatmodels.hf_model_local import chat_model   
from typing import TypedDict,Annotated, Optional, Literal
from pydantic import BaseModel, Field
import os

load_dotenv()
class tweets(BaseModel):
    summary : str=Field(description='write a summary for a tweet')
    # sentiments : Annotated[str, 'Return sentiments for a tweet either bullish, bearish, neutral']
    sentiments : Literal['bull', 'bear'] =Field(description='Return sentiments for a tweet either bullish, bearish, neutral')
    name: Optional[str]=Field(description='write down the name of tweet author')




llm = ChatOllama(model="llama3.2:3b", temperature=0.7)

# chat_hsi=[
#     SystemMessage(content="you  are helpful assitant"),
#     # HumanMessage(content="tell me about ollam. In just two sentences")
# ]

# while True:
#     user_input=input("you:")
#     chat_hsi.append(HumanMessage(content=user_input))
#     if user_input.lower() =='exit':
#         break
#     response=llm.invoke(chat_hsi)
#     chat_hsi.append(AIMessage(content=response.content))
#     print("AI: ", response.content)


# structural_output=model.with_structured_output(tweets)

# Initialize the model


# Simple invocation
# response = structural_output.invoke("""Bitcoin sitting at $91,075 while the rest of the world panics about Greenland tariffs. 🇬🇱 Whether it's a 'TACO' bounce or a bear trap, I’m just here for the volatility. Diamond hands only. 💎🙌 #BTC #CryptoDegen #Greenland""")
# print(response)
# print("\n")
print(llm.invoke("where is Pakistabn"))
# print(response.summary)
# print(response.sentiments)
# res=llm.invoke("where is Pakistabn")
# print(res.content)
