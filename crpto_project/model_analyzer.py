import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from transformers import pipeline

load_dotenv()

# --- MODEL INITIALIZATION ---
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # Note: 2.5-flash is not a standard version, using 1.5-flash
    api_key=os.getenv('GOOGLE_API_KEY'),
    temperature=0,
    max_tokens=500,
    max_output_tokens=700,
    max_retries=2,
)

parser = StrOutputParser()

prompt_template = PromptTemplate(
    template="""
# ROLE
Expert FinTwit Linguist.

# TASK
Write ONE tweet about {coin} that is exactly 3 short sentences long. 

# STRUCTURE (CRITICAL)
-write a complex bearish tweet that is exactly 3 short sentences long.
# CONSTRAINTS
- Total length: Under 140 characters.
- NO preamble. Output ONLY the tweet.

# COIN
{coin}
""",
    input_variables=["coin"]
)
generation_chain = prompt_template | model | parser

# --- 2. SENTIMENT PIPELINE ---
sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model="StephanAkkerman/FinTwitBERT-sentiment"
)

bertweet = pipeline(
    "sentiment-analysis",
    model="finiteautomata/bertweet-base-sentiment-analysis"
)


# --- 3. AUDITOR CHAIN ---
auditor_prompt = PromptTemplate(
    template="""
# ROLE
You are an expert NLP Auditor and Crypto Market Analyst.

# INPUT DATA
- Original Tweet: "{tweet}"
- Sentiment Model label: {sentiment_label}
- Sentiment Score: {sentiment_score}

# TASK
Evaluate if the Sentiment Model classified the tweet correctly.

# EVALUATION CRITERIA
- REASONING: Does the tweet contain "FinTwit" jargon (e.g., liquidity grab, order block) that the model interpreted correctly?
- BIAS: Did the model get distracted by the "Bearish" start and miss the "Bullish" ending (Recency/Primacy bias)?
- VERDICT: Is the prediction [CORRECT], [WRONG], or [PARTIALLY CORRECT]?

# OUTPUT FORMAT
Verdict: [Your Verdict]
Reason: (5-7 sentences explaining why the model succeeded or failed based on the linguistic nuances of the tweet.)
""",
    input_variables=["tweet", "sentiment_label", "sentiment_score"]
)
auditor_chain = auditor_prompt | model | parser

# --- LOOP THROUGH COINS ---
coins_to_test = ["$BTC", "$ETH", "$SOL", "$PENGUIN", "$LINK"]

for coin in coins_to_test:
    print(f"\n{'='*20} TESTING COIN: {coin} {'='*20}")
    
    # 1. Generate Tweet
    tweet = generation_chain.invoke({"coin": coin})
    print(f"GENERATED TWEET:\n{tweet}\n")
    
    # 2. Analyze Sentiment
    sentiment_results = sentiment_pipeline(tweet)
    sentiment_label = sentiment_results[0]['label']
    sentiment_score = sentiment_results[0]['score']
    print(f"SENTIMENT PREDICTION by stephan: {sentiment_label} ({sentiment_score:.4f})\n")
    
    # 3. Audit Result
    audit_report = auditor_chain.invoke({
        "tweet": tweet, 
        "sentiment_label": sentiment_label, 
        "sentiment_score": sentiment_score
    })
    print(f"AUDITOR ANALYSIS:\n{audit_report}")
    print("-" * 50)

    # 4. Analyze Sentiment

    print("pertweet scores ")
    sentiment_results = bertweet(tweet)
    sentiment_label = sentiment_results[0]['label']
    sentiment_score = sentiment_results[0]['score']
    print(f"SENTIMENT PREDICTION by bertweet: {sentiment_label} ({sentiment_score:.4f})\n")
    