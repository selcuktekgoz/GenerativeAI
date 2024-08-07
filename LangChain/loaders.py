# #WebBaseLoader ile URL'den içerik yüklemek

from langchain_community.document_loaders import WebBaseLoader

url = "https://blog.openzeka.com/ai/rag-retrieval-augmented-generation-nedir/"

loader = WebBaseLoader(url)

data = loader.load()

with open("github/GenerativeAI/LangChain/data/url_data.txt", "w") as file:
    file.write(data[0].page_content)

print(data[0].metadata)


# PyPDFLoader ile PDF dosyasından içerik yüklemek
from langchain_community.document_loaders import PyPDFLoader

file = "github/GenerativeAI/LangChain/data/773880.pdf"

loader = PyPDFLoader(file, extract_images=False)

data = loader.load()

with open("github/GenerativeAI/LangChain/data/pdf_data.txt", "w") as file:
    for page in data:
        file.write(page.page_content)

print(data[41].page_content, data[41].metadata)


# UnstructuredExcelLoader ile excel dosyasından içerik yüklemek

from langchain_community.document_loaders import UnstructuredExcelLoader

filepath = "github/GenerativeAI/LangChain/data/titanic.xlsx"

loader = UnstructuredExcelLoader(filepath, mode="elements")  # tabloyu htmle çeviriyor.

data = loader.load()

with open("github/GenerativeAI/LangChain/data/titanic.html", "w") as file:
    file.write(data[0].metadata["text_as_html"])
