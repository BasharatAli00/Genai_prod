from sentence_transformers import SentenceTransformer, util
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from sentence_transformers import SentenceTransformer, util
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# 1. Initialize the model


# 2. Your 'Database' (Corpus)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- STEP 2: Prepare your 5 Politician Documents ---
doc1 = Document(
    page_content="Imran Khan is the founder of Pakistan Tehreek-e-Insaf (PTI) and served as the 22nd Prime Minister. Before politics, he was an international cricket star, leading Pakistan to its only World Cup victory in 1992. His political narrative centers on 'Riyasat-e-Madina,' aiming to create a welfare state based on justice and meritocracy. He has a massive following among the youth and overseas Pakistanis. His tenure focused on anti-corruption, 'Ehsaas' poverty alleviation, and navigating economic challenges.",
    metadata={"source": "biography", "topic": "politics", "person": "Imran Khan", "period": "Modern"}
)

# Document 2: Nawaz Sharif
doc2 = Document(
    page_content="Mian Muhammad Nawaz Sharif is a veteran politician and the leader of the Pakistan Muslim League-Nawaz (PML-N). He holds the unique distinction of being elected Prime Minister of Pakistan three times. Known for his pro-business stance, his legacy is tied to massive infrastructure development, including the national motorway network and the start of the China-Pakistan Economic Corridor (CPEC). He remains a central figure in Pakistani politics, representing a strong voter base particularly in the Punjab province.",
    metadata={"source": "biography", "topic": "politics", "person": "Nawaz Sharif", "period": "Modern"}
)

# Document 3: Shehbaz Sharif
doc3 = Document(
    page_content="Mian Muhammad Shehbaz Sharif is the current Prime Minister of Pakistan and the brother of Nawaz Sharif. He gained a reputation as a highly efficient administrator during his multiple terms as the Chief Minister of Punjab. His governance style, often called 'Punjab Speed,' focused on rapid completion of public transport projects like the Metro Bus and Orange Line, as well as energy plants. As Prime Minister, he has led coalition governments through significant economic stabilization efforts and flood relief.",
    metadata={"source": "biography", "topic": "politics", "person": "Shehbaz Sharif", "period": "Modern"}
)

# Document 4: Zulfikar Ali Bhutto
doc4 = Document(
    page_content="Zulfikar Ali Bhutto was a charismatic leader who founded the Pakistan People's Party (PPP) with the slogan 'Roti, Kapra, aur Makaan.' He served as both President and Prime Minister in the 1970s. Bhutto is credited with giving Pakistan its first consensus-based Constitution in 1973 and initiating the country's nuclear program. He was a champion of Third World diplomacy and socialist reforms. Despite his controversial exit from power, he remains a symbol of democratic struggle for millions in Pakistan.",
    metadata={"source": "biography", "topic": "history", "person": "Zulfikar Ali Bhutto", "period": "1970s"}
)

# Document 5: Quaid-e-Azam Muhammad Ali Jinnah
doc5 = Document(
    page_content="Muhammad Ali Jinnah, revered as Quaid-e-Azam (Great Leader), was the founder and first Governor-General of Pakistan. A brilliant lawyer and statesman, he led the All-India Muslim League in the struggle for a separate homeland for Muslims in the subcontinent. His unwavering commitment to the 'Two-Nation Theory' and his 14 Points formed the basis for the creation of Pakistan in 1947. He envisioned a democratic state where all citizens, regardless of religion, would have equal rights and freedom of worship.",
    metadata={"source": "biography", "topic": "history", "person": "Muhammad Ali Jinnah", "period": "Founding"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./.chroma_db",
    collection_name="politicians"
)


# Check if the store is empty
if vectorstore._collection.count() == 0:
    print("Database is empty. Adding documents...")
    vectorstore.add_documents(docs)
else:
    print(f"Database already has {vectorstore._collection.count()} documents.")

# 2. NOW GET WILL SHOW THE 5 IDs
data = vectorstore.get(include=["embeddings","documents","metadatas"])
print(f"Total IDs found: {len(data['ids'])}")



# query="who is the founder of pakistan?"

# # results=vectorstore.similarity_search(query,k=1)
# # print(results[0].page_content)
# # print(results[0].metadata)
# result=vectorstore.similarity_search_with_score(query="",
# filter={"person":"Muhammad Ali Jinnah"},
# k=1)
# print(result)
retriver=vectorstore.as_retriever(search_kwargs={"k":1})
response=retriver.invoke("who is cricket captain of pakistan?")
print(response)
