import os
from openai import OpenAI
from dotenv import load_dotenv

path = "GenerativeAI/TextGen/.env"
load_dotenv(dotenv_path=path, verbose=True)
my_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=my_key)

response = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    temperature=0,
    max_tokens=512,
    messages=[
        {
            "role": "system",
            "content": "Sen Büyük Dil Modelleri konusunda bir uzmansın.",
        },
        {
            "role": "user",
            "content": "Büyük Dil Modellerini kullanarak neler yapabiliriz?",
        },
    ],
)


print(response.choices[0].message.content)


### STATS ###
# usage=CompletionUsage(completion_tokens=492, prompt_tokens=42, total_tokens=534)

### AI RESPONSE ###
# Büyük Dil Modelleri, doğal dil işleme alanında birçok farklı görevde kullanılabilir. İşte bazı örnekler:
# 1. Metin oluşturma: Büyük Dil Modelleri, metin oluşturma görevlerinde kullanılabilir. Örneğin, makaleler, blog yazıları, hikayeler gibi metinler oluşturmak için kullanılabilir.
# 2. Metin sınıflandırma: Metin sınıflandırma görevlerinde, metinlerin belirli kategorilere veya etiketlere göre sınıflandırılması için büyük dil modelleri kullanılabilir. Örneğin, duygu analizi, spam tespiti gibi görevlerde kullanılabilir.
# 3. Metin çevirisi: Büyük Dil Modelleri, metin çevirisi görevlerinde kullanılabilir. Özellikle çoklu dil desteği sağlayan modeller, farklı diller arasında metin çevirisi yapabilir.
# 4. Soru-cevap sistemleri: Büyük Dil Modelleri, soru-cevap sistemleri oluşturmak için kullanılabilir. Kullanıcıların sorularını anlayarak doğru cevapları üretebilir.
# 5. Konuşma tanıma ve sentezleme: Büyük Dil Modelleri, konuşma tanıma ve sentezleme görevlerinde de kullanılabilir. Konuşma metinlerini anlayarak doğru çıkarımlar yapabilir veya metinleri sesli olarak sentezleyebilir.
# 6. Metin özetleme: Büyük Dil Modelleri, metin özetleme görevlerinde kullanılabilir. Uzun metinleri özetleyerek ana fikirleri veya önemli bilgileri vurgulayabilir.
# Bu sadece bazı örneklerdir, büyük dil modelleri birçok farklı görevde kullanılabilir ve doğal dil işleme alanında geniş bir uygulama yelpazesine sahiptir.
