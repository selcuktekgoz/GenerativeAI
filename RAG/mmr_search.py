from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings

import os
from dotenv import load_dotenv

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(api_key=openai_key)

# mmr en alakasızı 4. sıraya koydu. mmr' da benzerlikle-çeşitlilik arasında trade-off var. varsayılan lambda: 0.5 olduğundan çeşitlilik oluşturmaya çalışıyor.
# en benzemez olanı çeşitlilik için dahil etti.


documents = [
    "DNA örnekleri laboratuvarda analiz edildi",
    "Farklı türler arasında genetik varyasyon gözlemlendi",
    "Bilim insanları yeni bir gen keşfetti",
    "Genom dizileme teknikleri hızla gelişiyor",
    "Kalıtsal hastalıkların genetik temelleri araştırılıyor",
    "İzmit'te bugün hava çok güzel.",
    # alakasız kontrol cümlesi ekle. buna yakın benzerliktekileri eleyip filtreleyelim. dil modeline bu alakasızları göndermeyelim.
]


query = "Kalıtsal hastalıkların tedavisi için Dna genom örnekleri araştırılıyor."

vectorstore = Chroma.from_texts(
    documents, embeddings
)  # dökümana from_texts çevirecek.page content meta data ekleyecek.

# Method1 - directly from the vectorstore
relevant_documents_vs = vectorstore.max_marginal_relevance_search(query)

# Method2 - using a retriever
retriever = vectorstore.as_retriever(
    search_type="mmr"
)  # aracı ile vector db ye gitmek. deault: cos sim.
relevant_documents_rt = retriever.get_relevant_documents(query)

print("Doğrudan MMR ile Elde Edilen Dokümanlar:")
print("*" * 100)
for doc in relevant_documents_vs:
    print(doc.page_content)
print("-" * 90)
print("Retriever Üzerinden Elde Edilen Dokümanlar:")
print("*" * 100)
for doc in relevant_documents_rt:
    print(doc.page_content)
