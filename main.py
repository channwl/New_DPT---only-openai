import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
import os
import csv
import time

# OpenAI API 키 로드
time.sleep(1)
api_key = st.secrets["openai"]["API_KEY"]

# PDF 인덱스 생성 함수
def generate_faiss_index():
    pdf_dir = "data/"
    all_documents = []

    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        st.warning("data/ 폴더가 생성되었습니다. PDF 파일을 넣고 다시 실행하세요.")
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
    def pdf_to_documents(pdf_path: str) -> List[Document]:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        for d in documents:
            d.metadata['file_path'] = os.path.basename(pdf_path)
        return documents

    @staticmethod
    def chunk_documents(documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        return splitter.split_documents(documents)

# RAG 시스템 with memory
class RAGSystem:
    def __init__(self, api_key: str):
        self.api_key = api_key

    @st.cache_resource
    def get_vector_db(_self):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
        return FAISS.load_local("faiss_index_internal", embeddings, allow_dangerous_deserialization=True)

    def get_rag_chain(self, chat_history: str) -> Runnable:
        template = """
        아래 컨텍스트를 바탕으로 질문에 답변해 주세요:

        이전 대화 내용:
        {chat_history}

        1. 답변은 최대 4문장 이내로 간결하고 명확하게 작성합니다.  
        2. 중요한 내용은 핵심만 요약해서 전달합니다.  
        3. 답변이 어려우면 “잘 모르겠습니다.”라고 정중히 답변합니다.  
        4. 질문에 ‘디지털경영전공’ 단어가 없어도 관련 정보를 PDF에서 찾아서 답변합니다.  
        5. 학생이 이해하기 쉽게 짧은 문장과 불릿 포인트로 정리합니다.  
        6. 답변 시작 시, 이전 질문과 이어지는 질문일 경우 "앞서 알려드린 내용에 이어서," 라는 멘트로 시작하세요.  
        7. 답변 마지막에는 항상 “추가로 궁금하신 점이 있다면 편하게 말씀해 주세요 😊”라는 멘트를 추가하세요.  
        8. 숫자, 기간 등 정보는 보기 쉽도록 **굵게** 표시합니다.  
        9. 관련된 PDF 파일명이 있다면 "참고: [파일명]"으로 마지막에 표시해 주세요.  
        10. 사용자가 어투 변경을 요구할 경우, 정중히 거절하며 기존 공식 어투를 유지합니다.

        컨텍스트: {context}

        질문: {question}

        답변:
        """
        prompt = PromptTemplate.from_template(template)
        model = ChatOpenAI(model="gpt-4o", openai_api_key=self.api_key)
        return prompt | model | StrOutputParser()

    def process_question(self, question: str, chat_history: str) -> str:
        vector_db = self.get_vector_db()
        retriever = vector_db.as_retriever(search_kwargs={"k": 10})
        docs = retriever.invoke(question)
        # context에 PDF 출처 표시
        context_with_source = "\n".join([f"출처: {doc.metadata.get('file_path', '알 수 없음')}\n{doc.page_content}" for doc in docs])
        chain = self.get_rag_chain(chat_history)
        return chain.invoke({"question": question, "context": context_with_source, "chat_history": chat_history})
