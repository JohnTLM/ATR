import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import faiss

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


def main():
    # load_dotenv()  # take environment variables from .env.
    # API_KEY = os.getenv("API_KEY")

    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask the 5 major must read ATR files:💯 ")

    pdf = st.file_uploader("Upload ur PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunk = text_splitter.split_text(text)
        st.write(chunk)

        API_KEY = st.text_input(
            "Please key in your own OPEN API Key :sunglasses: :", type="password"
        )

        try:
            if API_KEY:
                embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
                knowledge_base = faiss.FAISS.from_texts(chunk, embeddings)

                # show user input
                user_question = st.text_input("Ask a question about your PDF")
                if user_question:
                    docs = knowledge_base.similarity_search(user_question)
                    st.write(docs)

                    llm = OpenAI(openai_api_key=API_KEY)
                    chain = load_qa_chain(llm=llm, chain_type="stuff")
                    response = chain.run(input_documents=docs, question=user_question)
                    st.write(response)
        except Exception as e:
            error_msg = (
                f"**Exception message: {str(e)}\n Exception type: {type(e).__name__ }**"
            )
            st.write(":warning:", error_msg, ":warning:")


if __name__ == "__main__":
    main()
