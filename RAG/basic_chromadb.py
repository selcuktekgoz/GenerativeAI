import chromadb

client = (
    chromadb.HttpClient()
)  # yerelde çalışan sunucumuza api çağrısı yapacağız ( > chroma run )

collection_status = False  # collectionları tekrar oluşturmamak için kontrol değişkeni

current_collections = (
    client.list_collections()
)  # sunucuda mevcut bulunan tüm indekslerin listesi

for collection in current_collections:
    if collection.name == "new_collection":
        collection_status = True
        break

if collection_status:
    my_collection = client.get_collection(
        "new_collection"
    )  # new_collection var ise db den al getir
else:
    my_collection = client.create_collection(
        "new_collection"
    )  # dosya ilk kez çalıştırılıyorsa

    my_collection.add(
        documents=[
            "DNA örnekleri laboratuvarda analiz edildi",
            "Farklı türler arasında genetik varyasyon gözlemlendi",
            "Bilim insanları yeni bir gen keşfetti",
            "Genom dizileme teknikleri hızla gelişiyor",
            "Kalıtsal hastalıkların genetik temelleri araştırılıyor",
            # alakasız kontrol cümlemiz. buna yakın benzerliktekileri eleyip filtreleyelim. dil modeline bu alakasızları göndermeyelim.
        ],
        metadatas=[
            {"source": "source1"},
            {"source": "source2"},
            {"source": "source3"},
            {"source": "source4"},
            {"source": "source5"},
        ],
        ids=[
            "doc1",
            "doc2",
            "doc3",
            "doc4",
            "doc5",
        ],  # uniq olacak. yeniden sıralama, hibrit arama, çoklu getirmelerde kullanabiliriz
    )

results = my_collection.query(
    query_texts=[
        "Kalıtsal hastalıkların tedavisi için Dna genom örnekleri araştırılıyor."
    ],
    n_results=5,
)  # vektör karşılaştırması

retrieved_docs = results["documents"][0]  # dokümanlar
retrieved_distances = results["distances"][0]  # benzerlik skorları

for i, doc in enumerate(retrieved_docs):
    print(f"{doc}: {retrieved_distances[i]}")


# DNA örnekleri laboratuvarda analiz edildi: 0.4704855296224231
# Kalıtsal hastalıkların genetik temelleri araştırılıyor: 0.6984487642679768
# Farklı türler arasında genetik varyasyon gözlemlendi: 0.9425879874949435
# Genom dizileme teknikleri hızla gelişiyor: 1.02044565578035
# Bilim insanları yeni bir gen keşfetti: 1.1978877461845492

# Before running this file, first run the "chroma run" command from another terminal. Once server is up you can use this
# python can also run in-memory with no server running: chromadb.PersistentClient()

print("*" * 100)


client = chromadb.HttpClient()
collection = client.create_collection("thenewest_collection")

# Add docs to the collection. Can also update and delete. Row-based API coming soon!
collection.add(
    documents=[
        "You are not alone",
        "This is document2",
        "Seasons happen for some reason",
    ],  # we embed for you, or bring your own
    metadatas=[
        {"source": "notion"},
        {"source": "google-docs"},
        {"source": "txt file"},
    ],  # filter on arbitrary metadata!
    ids=["doc1", "doc2", "doc3"],  # must be unique for each doc
)

results = collection.query(
    query_texts=["Earth has seasons due to certain factors"],
    n_results=1,
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)

print(results)
print("*" * 100)
print(f"Distances: {results['distances']}")
