import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
import os
import re
import csv
import time
from dotenv import load_dotenv
import openai
import tempfile
import uuid

# 환경 변수 로드 및 API 키 설정
time.sleep(1)
openai.api_key = st.secrets["openai"]["API_KEY"]
api_key = openai.api_key

# 배경 및 커스텀 CSS 적용
st.markdown("""
    <style>
    body {
        background-color: #f7f7f8;
    }
    .user-msg {
        background-color: #fbe8ed;
        border: 2px solid #dc143c;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .chat-history {
        background-color: white;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# PDF 처리 클래스
class PDFProcessor:
    @staticmethod
    def pdf_to_documents(pdf_path: str) -> list[Document]:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        for d in documents:
            d.metadata['file_path'] = pdf_path
        return documents

    @staticmethod
    def chunk_documents(documents: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        return splitter.split_documents(documents)

    @staticmethod
    def save_to_vector_store(documents: list[Document], index_name: str) -> bool:
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
            vector_store = FAISS.from_documents(documents, embeddings)
            vector_store.save_local(index_name)
            return True
        except Exception as e:
            st.error(f"벡터 저장 중 오류 발생: {e}")
            return False

# RAG 시스템 클래스
class RAGSystem:
    def __init__(self, api_key: str, index_name: str):
        self.api_key = api_key
        self.index_name = index_name

    def get_rag_chain(self) -> Runnable:
        template = """질문: {question}\n\n컨텍스트: {context}\n\n답변:"""
        prompt = PromptTemplate.from_template(template)
        model = ChatOpenAI(model="gpt-4o", openai_api_key=self.api_key)
        return prompt | model | StrOutputParser()

    @st.cache_resource
    def get_vector_db(_self, index_name):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
        return FAISS.load_local(index_name, embeddings, allow_dangerous_deserialization=True)

    def process_question(self, question: str) -> str:
        vector_db = self.get_vector_db(self.index_name)
        retriever = vector_db.as_retriever(search_kwargs={"k": 5})
        context_docs = retriever.invoke(question)
        chain = self.get_rag_chain()
        return chain.invoke({"question": question, "context": context_docs})

# 메인 앱 실행 함수
def main():
    st.set_page_config(page_title="디지털경영 챗봇", layout="wide")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_questions" not in st.session_state:
        st.session_state.user_questions = []

    st.title("💬 디지털경영전공 AI 챗봇")
    st.caption("PDF 기반 학과 정보 상담 도우미")

    left_column, mid_column, right_column = st.columns([1.2, 2.5, 1.3])

    # 중앙 채팅 영역
    with mid_column:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-msg">🧑‍🎓 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                with st.chat_message("assistant"):
                    st.markdown(f"🤖 {msg['content']}")

        prompt = st.chat_input("질문을 입력해 주세요.")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.user_questions.append(prompt)

            rag_system = RAGSystem(api_key, "faiss_index_department")
            with st.spinner("답변 생성 중..."):
                answer = rag_system.process_question(prompt)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.experimental_rerun()

    # 오른쪽 히스토리 및 다운로드
    with right_column:
        st.subheader("📝 질문 히스토리")
        if st.session_state.user_questions:
            with st.expander("지금까지 질문한 목록"):
                for i, q in enumerate(st.session_state.user_questions, 1):
                    st.markdown(f"{i}. {q}")

        if st.session_state.messages:
            chat_log = "역할,내용\n"
            for m in st.session_state.messages:
                role = "사용자" if m["role"] == "user" else "챗봇"
                content = m["content"].replace("\n", " ").replace(",", " ")
                chat_log += f"{role},{content}\n"

            st.download_button(
                label="⬇️ 질문-응답 기록 다운로드 (CSV)",
                data=chat_log.encode("utf-8-sig"),
                file_name="chat_history.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
