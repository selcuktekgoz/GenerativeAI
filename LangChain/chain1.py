# Create Stuff Documents Chain
from langchain_openai import ChatOpenAI
from langchain_core.documents import (
    Document,
)  # veri modelidir. Document.page_content | Document.meta_data
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

import os
from dotenv import load_dotenv

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=openai_key)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Bu belgelerde adı geçen kişilerin en sevdiği yemeği kişi:yemek şeklinde bir sözlük formatında yaz:\n\n{context}",
        )
    ]  # birden çok değer alabileceği için liste içinde system,human vb.
)


docs = [
    Document(
        page_content="Ayşe'nin en sevdiği yemek mantıdır, ancak köfteyi de çok sever."
    ),
    Document(page_content="Ali pizzayı sever ama makarnayı daha çok sever."),
    Document(
        page_content="Zeynep her zaman 'Favori yemeğim yok' dese de, aslında lahmacun en sevdiği yemektir."
    ),
]


chain_1 = create_stuff_documents_chain(llm, prompt)

sozluk = chain_1.invoke({"context": docs})

print(sozluk)
