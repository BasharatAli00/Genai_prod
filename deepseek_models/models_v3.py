from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
from typing import TypedDict,Annotated, Optional, Literal
from pydantic import BaseModel, Field
import os

load_dotenv()
class tweets(BaseModel):
    summary : str=Field(description='write a summary for a tweet')
    # sentiments : Annotated[str, 'Return sentiments for a tweet either bullish, bearish, neutral']
    sentiments : Literal['bull', 'bear'] =Field(description='Return sentiments for a tweet either bullish, bearish, neutral')
    name: Optional[str]=Field(description='write down the name of tweet author')



#model

model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv('DEEPSEEK_API_KEY'), # Or set DEEPSEEK_API_KEY environment variable
    temperature=0,
    max_tokens=500,
    timeout=None,
    max_retries=2,
)

structural_output=model.with_structured_output(tweets)

# Initialize the model


# Simple invocation
response = structural_output.invoke("""Bitcoin sitting at $91,075 while the rest of the world panics about Greenland tariffs. 🇬🇱 Whether it's a 'TACO' bounce or a bear trap, I’m just here for the volatility. Diamond hands only. 💎🙌 #BTC #CryptoDegen #Greenland""")
# print(response)
print("\n")
print(response.summary)
print(response.sentiments)


# print(f"Summary: {response['summary']}")
# print(f"Sentiment: {response['sentiments']}")