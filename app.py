import argparse
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.memory import ConversationSummaryMemory
from typing import List
import os
import time

# OpenAI API 키 로드
time.sleep(1)
try:
    api_key = st.secrets["openai"]["API_KEY"]
except:
    api_key = os.environ.get("OPENAI_API_KEY")  # 콘솔용

# PDF 처리 클래스
class PDFProcessor:
    @staticmethod
    def pdf_to_documents(pdf_path: str) -> List[Document]:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        for d in documents:
            d.metadata['file_path'] = pdf_path
        return documents

    @staticmethod
    def chunk_documents(documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        return splitter.split_documents(documents)

# FAISS 인덱스 생성 함수
def generate_faiss_index():
    pdf_dir = "data/"
    all_documents = []

    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        print("data/ 폴더가 생성되었습니다. PDF 파일을 여기에 넣고 다시 실행해주세요.")
        return

    pdf_files = [file for file in os.listdir(pdf_dir) if file.endswith(".pdf")]
    if not pdf_files:
        print("data/ 폴더에 PDF 파일이 없습니다. PDF를 추가한 후 다시 실행해주세요.")
        return

    for file_name in pdf_files:
        docs = PDFProcessor.pdf_to_documents(os.path.join(pdf_dir, file_name))
        all_documents.extend(docs)

    chunks = PDFProcessor.chunk_documents(all_documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index_internal")
    print(f"{len(pdf_files)}개의 PDF 파일로 인덱스 생성 완료!")

# RAG 시스템
class RAGSystem:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = ChatOpenAI(model="gpt-4o", openai_api_key=self.api_key)
        self.memory = ConversationSummaryMemory(llm=self.llm)
        self.rag_chain = self.get_rag_chain()

    def get_vector_db(self):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=self.api_key)
        return FAISS.load_local("faiss_index_internal", embeddings, allow_dangerous_deserialization=True)

    def get_rag_chain(self) -> Runnable:
        template = """
        아래 컨텍스트와 대화 기록을 바탕으로 질문에 답변해 주세요:

        - 간결하고 명확하게 최대 4문장 이내로 작성
        - 핵심 내용은 **굵게** 표시
        - 불릿 포인트로 정리
        - 추가 질문 유도
        - 이해가 어렵거나 불확실하면 “잘 모르겠습니다.”라고 답변

        대화 요약: {history}
        문서 컨텍스트: {context}
        질문: {question}

        답변:
        """
        prompt = PromptTemplate.from_template(template)
        return prompt | self.llm | StrOutputParser()

    def process_question(self, question: str) -> str:
        vector_db = self.get_vector_db()
        retriever = vector_db.as_retriever(search_kwargs={"k": 10})
        docs = retriever.invoke(question)

        history = self.memory.chat_memory.messages

        answer = self.rag_chain.invoke({
            "question": question,
            "context": docs,
            "history": history,
        })

        self.memory.save_context({"input": question}, {"output": answer})
        return answer

# 콘솔 모드
def run_console_mode():
    print("디지털경영전공 챗봇 (콘솔 모드)")
    print("PDF 기반 학과 질문에 답변합니다. 'exit' 입력 시 종료됩니다.")
    rag = RAGSystem(api_key)

    while True:
        user_input = input("질문: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            print("종료합니다.")
            break
        answer = rag.process_question(user_input)
        print(f"답변: {answer}\n")

# 웹앱 메인 (Streamlit)
def run_web_mode():
    st.set_page_config(page_title="디지털경영전공 챗봇", layout="wide")
    st.title("🎓 디지털경영전공 챗봇")
    st.caption("학과에 대한 다양한 질문에 친절하게 답변해드립니다.")

    if st.button("📥 채팅 시작 !"):
        generate_faiss_index()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    prompt = st.chat_input("궁금한 점을 입력해 주세요.")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        rag = RAGSystem(api_key)

        with st.spinner("질문을 이해하는 중입니다. 잠시만 기다려주세요."):
            answer = rag.process_question(prompt)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

    for msg in st.session_state.messages:
        role, content = msg["role"], msg["content"]
        color = "#731034" if role == "user" else "#f8f8f8"
        prefix = "💬 질문" if role == "user" else "🤖 답변"
        st.markdown(f"""
        <div style='background-color: {color}; padding: 10px; border-radius: 15px; margin-bottom: 10px; max-width: 70%; color: {"white" if role == "user" else "black"};'>
        <b>{prefix}:</b> {content}
        </div>
        """, unsafe_allow_html=True)

# 실행 모드 선택
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["console", "web"], default="web", help="실행 모드 선택: console 또는 web")
    args = parser.parse_args()

    if args.mode == "console":
        run_console_mode()
    else:
        run_web_mode()
