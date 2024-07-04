# https://www.skysports.com/football/news/11667/13160228/erik-ten-hag-manchester-united-boss-extends-contract-until-2026
# https://www.nytimes.com/athletic/5616278/2024/07/04/manchester-united-transfers-christopher-vivell/
# https://www.juvefc.com/juventus-to-present-their-new-jersey-design-without-sponsors/

import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os;

load_dotenv('.env')
API_KEY = os.environ['API_KEY']
llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=API_KEY, temperature=1, max_output_tokens=1000)

st.title("News Report Research")
st.sidebar.title("News Article URLs")

if "urls" not in st.session_state:
    st.session_state.urls = ["", "", ""]
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None

for i in range(3):
    st.session_state.urls[i] = st.sidebar.text_input(f"URL {i+1}", value=st.session_state.urls[i])

process_URL_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

if process_URL_clicked:
    loader = UnstructuredURLLoader(urls=st.session_state.urls)
    main_placeholder.text("Loading data...")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=1000,
    )
    main_placeholder.text("Splitting data...")
    docs = text_splitter.split_documents(data)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
    vector_index = FAISS.from_documents(docs, embeddings)
    main_placeholder.write("Saving vector index...")
    time.sleep(2)
    vector_index.save_local("faiss_index")

    st.session_state.embeddings = embeddings
    st.session_state.vector_index = vector_index
    main_placeholder.text("Done!")

query = main_placeholder.text_input("Ask your question.")
if query and st.session_state.embeddings and st.session_state.vector_index:
    vector_store = FAISS.load_local("faiss_index", st.session_state.embeddings, allow_dangerous_deserialization=True)
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(), input_key="query", return_source_documents=True)
    result = chain({"query": query})
    st.header("Answer")
    st.write(result["result"])
else:
    st.write("Please process URLs first.")
