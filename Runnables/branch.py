from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch
from transformers import pipeline
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
import os   
load_dotenv()

# 1. Setup the raw pipeline
pipe = pipeline(
    "sentiment-analysis", 
    model="StephanAkkerman/FinTwitBERT-sentiment"
)

# 2. Fix the Model Wrapper
# We extract just the LABEL string from the Hugging Face list: [{'label': 'BULLISH', ...}]
model = RunnableLambda(lambda x: pipe(x)[0]['label'])
model1=ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv('DEEPSEEK_API_KEY'), # Or set DEEPSEEK_API_KEY environment variable
    temperature=0,
    max_tokens=500,
    timeout=None,
    max_retries=2,
)
parser = StrOutputParser()

promt1 = PromptTemplate(
    template="Analyze the sentiment of this topic: {topic}",
    input_variables=["topic"]
)
promt2 = PromptTemplate(
    template="write a appropriate response for the given feedback {feedback}",
    input_variables=["feedback"]
)
promt3 = PromptTemplate(
    template="write a appropriate response for the given feedback {feedback}",
    input_variables=["feedback"]
)

# 3. Fix the Chain
# We use .text to get the string from the prompt, then pass it to our model
tweet = promt1 | (lambda x: x.text) | model | parser


# Run it
branch_chain=RunnableBranch(
    (lambda x: x == "BULLISH", promt2|model1|parser),
    (lambda x: x == "BEARISH", promt3|model1|parser),
    RunnableLambda(lambda x: "Invalid sentiment")   
)
final_chain=tweet|branch_chain
print(final_chain.invoke({"topic": "bitcoin is going to the moon"}))

