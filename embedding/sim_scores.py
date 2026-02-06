from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
countries = [
    "Pakistan is known for its rich culture, diverse landscapes, and strong traditions.",
    "Japan is famous for its advanced technology, discipline, and unique blend of modern and traditional life.",
    "France is renowned for its art, fashion, cuisine, and historical landmarks like the Eiffel Tower.",
    "Brazil is popular for its vibrant festivals, football culture, and the Amazon rainforest."
]


query = 'pakistan is known for what'


doc_emb = model.encode(countries)
query_emb=model.encode(query)


scores = cosine_similarity([query_emb], doc_emb)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]
print(query)
print(countries[index])
print("similarity score is:", score)
