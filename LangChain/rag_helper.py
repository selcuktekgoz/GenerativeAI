from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS  # by facebook
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

my_key_openai = os.getenv("OPENAI_API_KEY")
my_key_google = os.getenv("GOOGLE_API_KEY")
my_key_cohere = os.getenv("cohere_apikey")
my_key_hf = os.getenv("hf_access_token")

llm_gemini = ChatGoogleGenerativeAI(google_api_key=my_key_google, model="gemini-pro")

# embeddings = OpenAIEmbeddings(api_key=my_key_openai)
# embeddings = CohereEmbeddings(cohere_api_key=my_key_cohere, model="embed-multilingual-v3.0") #embed-english-v3.0

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=my_key_hf, model_name="sentence-transformers/all-MiniLM-l6-v2"
)


def ask_gemini(prompt):

    response = llm_gemini.invoke(prompt)

    return response.content


def rag_with_url(target_url, prompt):

    # yükle
    loader = WebBaseLoader(target_url)

    data = loader.load()

    # bölümle
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, length_function=len
    )

    splitted_documents = text_splitter.split_documents(data)

    # bölünlenmiş dökümanları vektöre çevirip vektör db'de sakla
    # dökümanı embeddings modeline göre vektöre çeviriyor
    vectorstore = FAISS.from_documents(splitted_documents, embeddings)
    retriever = vectorstore.as_retriever()

    # promptu vektöre çevirerek , cos similarity kullanarak en alakalı 4 metodu getirecek
    relevant_documents = retriever.get_relevant_documents(prompt)

    # tek parça string olacak
    context_data = " ".join(document.page_content for document in relevant_documents)

    # meta-prompt
    final_prompt = f"""
    Soru: {prompt}

    Kullanılabilir Bilgiler:
    {context_data}
    
    Yanıtınızı Şu Şartlarda Veriniz:
    
    Veri Sınırlamaları: Yanıtınızı yalnızca yukarıda verilen bilgiler doğrultusunda oluşturun. Sağlanan bilgiler dışında herhangi bir dış kaynağa başvurmaktan veya varsayımda bulunmaktan kaçının.
    Netlik ve Kesinlik: Yanıtınızın açık, net ve doğrudan olması önemlidir. Bilgilerin doğruluğunu ve eksiksizliğini sağlamak için sadece verilen verileri kullanarak yanıtınızı şekillendirin.
    İlgililik: Yanıtınızın soruyla doğrudan ilgili ve soruyu en iyi şekilde yanıtlayacak şekilde olmasına dikkat edin.
    """

    response = ask_gemini(final_prompt)

    return response


def rag_with_pdf(filepath, prompt):

    loader = PyPDFLoader(filepath)

    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, length_function=len
    )

    splitted_documents = text_splitter.split_documents(data)

    vectorstore = FAISS.from_documents(splitted_documents, embeddings)
    retriever = vectorstore.as_retriever()

    relevant_documents = retriever.get_relevant_documents(prompt)

    context_data = " ".join(document.page_content for document in relevant_documents)

    final_prompt = f"""
    Soru: {prompt}

    Kullanılabilir Bilgiler:
    {context_data}

    Yanıtınızı Şu Şartlarda Veriniz:

    Veri Sınırlamaları: Yanıtınızı yalnızca yukarıda verilen bilgiler doğrultusunda oluşturun. Sağlanan bilgiler dışında herhangi bir dış kaynağa başvurmaktan veya varsayımda bulunmaktan kaçının.
    Netlik ve Kesinlik: Yanıtınızın açık, net ve doğrudan olması önemlidir. Bilgilerin doğruluğunu ve eksiksizliğini sağlamak için sadece verilen verileri kullanarak yanıtınızı şekillendirin.
    İlgililik: Yanıtınızın soruyla doğrudan ilgili ve soruyu en iyi şekilde yanıtlayacak şekilde olmasına dikkat edin.
    """

    response = ask_gemini(final_prompt)

    return response, relevant_documents
