from sentence_transformers import SentenceTransformer
sentences = ["This is", "hi there"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)

print(len(embeddings[1]))
# print(embeddings)

