from sentence_transformers import SentenceTransformer, util

# 1. Initialize the model
model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")

# 2. Your 'Database' (Corpus)
politicians = [
    "Imran Khan is the founder of PTI and 22nd PM. Led 1992 World Cup win.",
    "Nawaz Sharif is the leader of PML-N and served as PM three times.",
    "Shehbaz Sharif is the current PM, known for Punjab Speed.",
    "Zulfikar Ali Bhutto founded the PPP and the nuclear program.",
    "Muhammad Ali Jinnah is the founder of Pakistan and first Governor-General."
]

# 3. Encode the corpus into embeddings
corpus_embeddings = model.encode(politicians, convert_to_tensor=True)

# 4. Your Query
query = "Who is the leader of PTI?"
query_embedding = model.encode(query, convert_to_tensor=True)

# 5. Compute Cosine Similarity
# util.cos_sim returns a matrix of scores between 0 and 1
cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)

# 6. Find the best match
best_result_idx = cosine_scores.argmax().item()
best_score = cosine_scores[0][best_result_idx].item()

print(f"Query: {query}")
print(f"Top Match: {politicians[best_result_idx]}")
print(f"Similarity Score: {best_score:.4f}")