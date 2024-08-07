import streamlit as st
import basic_rag_helper

st.set_page_config(page_title="LangChain ile Bellek Genişletme", layout="wide")
st.title("LangChain ile Bellek Genişletme: URL")
st.divider()

col_input, col_rag = st.columns([1, 2])

with col_input:
    target_url = st.text_input(label="Web Adresini Giriniz:")
    st.divider()
    prompt = st.text_input(label="Sorunuzu Giriniz:", key="url_prompt")
    st.divider()
    submit_btn = st.button(label="Sor", key="url_button")
    st.divider()

    if submit_btn:

        with col_rag:
            with st.spinner("Yanıt Hazırlanıyor..."):
                st.success("YANIT:")
                st.markdown(
                    basic_rag_helper.rag_with_url(target_url=target_url, prompt=prompt)
                )
                st.divider()


st.title("LangChain ile Bellek Genişletme: PDF")
st.divider()

col_input, col_rag = st.columns([1, 2])

with col_input:
    selected_file = st.file_uploader(label="İşlenecek Dosyayı Seçiniz", type=["pdf"])
    st.divider()
    prompt = st.text_input(label="Sorunuzu Giriniz:", key="pdf_prompt")
    st.divider()
    submit_btn = st.button(label="Sor", key="pdf_button")
    st.divider()

if submit_btn:

    with col_rag:
        with st.spinner("Yanıt Hazırlanıyor..."):
            st.success("YANIT:")
            response, relevant_documents = basic_rag_helper.rag_with_pdf(
                filepath=f"./data/{selected_file.name}", prompt=prompt
            )
            st.markdown(response)
            st.divider()
            for doc in relevant_documents:
                st.caption(doc.page_content)
                st.markdown(f"Kaynak: {doc.metadata}")
                st.divider()
