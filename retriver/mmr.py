from langchain_classic.retrievers import MultiQueryRetriever
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

doc6=Document(
    page_content="imran khan is a Pakistani politician who served as the 22nd Prime Minister of Pakistan from August 2018 to April 2022. He is the founder of Pakistan Tehreek-e-Insaf (PTI), a political party he established in 1996. Before entering politics, Khan was a world-renowned cricketer who led Pakistan to victory in the 1992 Cricket World Cup. He is widely regarded as one of the greatest all-rounders in the history of cricket. Khan's political career has been marked by his anti-corruption stance and his promise to create a welfare state based on Islamic principles. He has also been a vocal critic of Western foreign policy and has advocated for a more independent foreign policy for Pakistan. Despite facing numerous challenges, including legal battles and political opposition, Khan remains a popular figure in Pakistan, particularly among the youth and overseas Pakistanis.   ",
    metadata={"source": "biography", "topic": "history", "person": "imran khan", "period": "Founding"}
)

docs = [doc1, doc2, doc3, doc4, doc5,doc6]

vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embeddings,
)



mmr=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3},
    lambda_mult=0.5
)
for i , doc in enumerate(mmr.invoke("who is the founder pti?")):
    print(f"Document {i+1}")
    print(doc.page_content)
    print(doc.metadata)
    print("\n")