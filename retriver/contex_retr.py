from langchain_classic.retrievers import MultiQueryRetriever    , ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_community.vectorstores import FAISS
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
load_dotenv()

# 1. Initialize the model


model=ChatDeepSeek(model_name="deepseek-chat",api_key=os.getenv("DEEPSEEK_API_KEY"))

# 2. Your 'Database' (Corpus)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- STEP 2: Prepare your 5 Politician Documents ---
doc1 = Document(
    page_content="Imran Khan is the founder of Pakistan Tehreek-e-Insaf (PTI) and served as the 22nd Prime Minister. Before politics, wasim akram play a bes t role in an engaging movie which is base on true events.",
    metadata={"source": "biography", "topic": "politics", "person": "Imran Khan", "period": "Modern"}
)

# Document 2: Nawaz Sharif
doc2 = Document(
    page_content="Mian Muhammad Nawaz Sharif is a veteran politician and the leader of the Pakistan Muslim League-Nawaz (PML-N).  football is widely watch all over the world. 11 players are in each team.",
    metadata={"source": "biography", "topic": "politics", "person": "Nawaz Sharif", "period": "Modern"}
)

# Document 3: Shehbaz Sharif
doc3 = Document(
    page_content="Mian Muhammad Shehbaz Sharif is the current Prime Minister of Pakistan and the brother of Nawaz Sharif. He gained a reputation as a highly efficient administrator during his multiple terms as the Chief Minister of Punjab. His governance style, often called 'Punjab Speed,' focused on rapid completion of public transport projects like the Metro Bus and Orange Line, as well as energy plants. As Prime Minister, he has led coalition governments through significant economic stabilization efforts and flood relief.",
    metadata={"source": "biography", "topic": "politics", "person": "Shehbaz Sharif", "period": "Modern"}
)



docs = [doc1, doc2, doc3]

vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embeddings,
)



base_retriver=vectorstore.as_retriever(search_kwargs={"k":3})

compressor=LLMChainExtractor.from_llm(llm=model)

retriever = ContextualCompressionRetriever(
    base_retriever=base_retriver,
    base_compressor=compressor,  # Fixed: added the underscore
)

results=retriever.invoke("who is the founder of pti?")

for i , doc in enumerate(results):
    print(f"Document {i+1}")
    print(doc.page_content)
    print(doc.metadata)
    print("\n")


