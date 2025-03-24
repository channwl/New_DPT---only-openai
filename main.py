import streamlit as st
st.set_page_config(page_title="디지털경영 챗봇", layout="wide")

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

# PDF 인덱스 생성 스크립트 (한 번만 실행)
def generate_faiss_index():
    pdf_dir = "data/"
    all_documents = []

    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        st.warning("data/ 폴더가 생성되었습니다. PDF 파일을 여기에 넣고 다시 실행하세요.")
        return

    pdf_files = [file for file in os.listdir(pdf_dir) if file.endswith(".pdf")]
    if not pdf_files:
        st.error("data/ 폴더에 PDF 파일이 없습니다. PDF를 추가한 후 다시 실행하세요.")
        return

    for file_name in pdf_files:
        docs = PDFProcessor.pdf_to_documents(os.path.join(pdf_dir, file_name))
        all_documents.extend(docs)

    chunks = PDFProcessor.chunk_documents(all_documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index_internal")
    st.success(f"{len(pdf_files)}개의 PDF 파일로 인덱스 생성 완료!")

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

# RAG 시스템 클래스 (사전 인덱스 불러오기)
class RAGSystem:
    def __init__(self, api_key: str):
        self.api_key = api_key

    @st.cache_resource
    def get_vector_db(_self):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
        return FAISS.load_local("faiss_index_internal", embeddings, allow_dangerous_deserialization=True)

    def get_rag_chain(self) -> Runnable:
        template = """질문: {question}\n\n컨텍스트: {context}\n\n답변 (마지막에 '추가로 궁금하신 점이 있다면 편하게 물어봐 주세요 😊'로 마무리하세요):"""
        prompt = PromptTemplate.from_template(template)
        model = ChatOpenAI(model="gpt-4o", openai_api_key=self.api_key)
        return prompt | model | StrOutputParser()

    def process_question(self, question: str) -> str:
        vector_db = self.get_vector_db()
        retriever = vector_db.as_retriever(search_kwargs={"k": 5})
        context_docs = retriever.invoke(question)
        chain = self.get_rag_chain()
        return chain.invoke({"question": question, "context": context_docs})

# 피드백 저장 함수
def save_feedback(feedback_text, rating):
    if feedback_text.strip() != "" or rating > 0:
        with open("feedback_log.csv", mode="a", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), rating, feedback_text])
        return True
    return False

# 메인 앱 실행 함수
def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_questions" not in st.session_state:
        st.session_state.user_questions = []

    st.title("🎓 디지털경영전공 AI 챗봇")
    st.caption("PDF 자료 기반으로 빠르게 학과 정보를 알려드려요!")

    if st.button("📥 (관리자) 인덱스 다시 생성하기"):
        generate_faiss_index()

    left_column, mid_column, right_column = st.columns([1.2, 2.5, 1.3])

    with left_column:
        st.subheader("📚 사용 가이드")
        st.markdown("""1. 질문을 입력해 주세요.<br>2. 답변을 확인하고 추가 질문도 가능해요.<br>3. 필요 시 오른쪽에서 피드백 남겨주세요!""", unsafe_allow_html=True)
        st.subheader("📂 PDF 목록")
        pdf_files = [file for file in os.listdir("data/") if file.endswith(".pdf")]
        if pdf_files:
            for file in pdf_files:
                st.markdown(f"✅ {file}")
        else:
            st.info("현재 등록된 PDF 파일이 없습니다.")

    with mid_column:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'''
                    <div style="
                        background-color: #e9f5ff;
                        padding: 12px;
                        border-radius: 20px;
                        margin-bottom: 10px;
                        max-width: 65%;
                        text-align: left;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    ">
                    💬 <b>질문:</b> {msg["content"]}
                    </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                    <div style="
                        background-color: #f8f8f8;
                        padding: 12px;
                        border-radius: 20px;
                        margin-bottom: 10px;
                        margin-left: auto;
                        max-width: 65%;
                        text-align: left;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    ">
                    🤖 <b>답변:</b> {msg["content"]}
                    </div>
                ''', unsafe_allow_html=True)

        prompt = st.chat_input("궁금한 점을 입력해 주세요!")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.user_questions.append(prompt)

            rag_system = RAGSystem(api_key)
            with st.spinner("질문을 이해하는 중이에요! 잠시만 기다려주세요 😊"):
                answer = rag_system.process_question(prompt)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()

    with right_column:
        st.subheader("📝 질문 히스토리")
        if st.session_state.user_questions:
            with st.expander("지금까지 질문한 내용"):
                for i, q in enumerate(st.session_state.user_questions, 1):
                    st.markdown(f"{i}. {q}")

        st.subheader("🌟 만족도 피드백")
        rating = st.slider("별점으로 만족도를 남겨주세요", min_value=0, max_value=5, step=1, format="%d⭐")
        feedback_input = st.text_area("개발자에게 전하고 싶은 말을 적어주세요!")
        if st.button("피드백 제출"):
            if save_feedback(feedback_input, rating):
                st.success("피드백이 제출되었습니다! 감사합니다.")
            else:
                st.warning("별점 또는 의견을 입력해 주세요.")

        if st.session_state.messages:
            chat_log = "역할,내용\n"
            for m in st.session_state.messages:
                role = "사용자" if m["role"] == "user" else "챗봇"
                content = m["content"].replace("\n", " ").replace(",", " ")
                chat_log += f"{role},{content}\n"

            st.download_button(
                label="⬇️ 대화 기록 다운로드 (CSV)",
                data=chat_log.encode("utf-8-sig"),
                file_name="chat_history.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
