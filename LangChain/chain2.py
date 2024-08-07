from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional
from langchain.chains.openai_functions import create_openai_fn_runnable

import os
from dotenv import load_dotenv

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")


class Araba(BaseModel):
    """Bir araba hakkında tanımlayıcı bilgiler"""

    marka: str = Field(..., description="Arabanın markası")  # ... zorunlu alan
    model: str = Field(..., description="Arabanın modeli")
    yil: int = Field(..., description="Arabanın üretim yılı")
    km: Optional[int] = Field(None, description="Arabanın km.si")


class Kullanici(BaseModel):
    """Bir kullanıcı hakkında tanımlayıcı bilgiler"""

    isim: str = Field(..., description="Kullanıcının ismi")
    yas: int = Field(..., description="Kullanıcının yaşı")
    ehliyet_sinifi: Optional[str] = Field(
        None, description="Kullanıcının ehliyet sınıfı"
    )


llm = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=openai_key)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sen varlıkları kaydetmek konusunda uzmansın.",
        ),
        (
            "human",
            "Verdiğim girdideki varlıkları kaydetmek için gerekli fonksiyonlara çağrı yaparak sonuçları sözlük yapısında döndür: {input}",
        ),
    ]
)

chain_2 = create_openai_fn_runnable([Araba, Kullanici], llm, prompt)

sonuc_1 = chain_2.invoke({"input": "Ali'nin 2015 model beyaz bir Ford Fiesta'sı var."})

sonuc_2 = chain_2.invoke(
    {
        "input": "Selçuk 39 yaşında, ehliyetini 10 yıl önce aldı ve B sınıfı ehliyeti var."
    }
)


print(sonuc_1)
print(sonuc_2)
