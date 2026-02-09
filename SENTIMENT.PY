from transformers import pipeline


model=pipeline("sentiment-analysis",
model="StephanAkkerman/FinTwitBERT-sentiment"
)

tweet="""$BTC holding the range high after the impulse move is constructive. Momentum remains strong and dips are getting bought quickly, suggesting demand is still in control. As long as we hold this level, continuation toward higher resistance looks likely. Let’s see if volume confirms on the next push"""

print(model(tweet))


from transformers import pipeline

bertweet = pipeline(
    "sentiment-analysis",
    model="finiteautomata/bertweet-base-sentiment-analysis"
)

print(bertweet("""$BTC holding the range high after the impulse move is constructive. Momentum remains strong and dips are getting bought quickly, suggesting demand is still in control. As long as we hold this level, continuation toward higher resistance looks likely. Let’s see if volume confirms on the next push"""))


