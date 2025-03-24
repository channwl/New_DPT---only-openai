import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List, Tuple
import os
import csv
import time
import uuid

# OpenAI API 키 로드
time.sleep(1)
api_key = st.secrets["openai"]["API_KEY"]

# PDF 인덱스 생성 스크립트
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

# RAG 시스템
class RAGSystem:
    def __init__(self, api_key: str):
        self.api_key = api_key

    @st.cache_resource
    def get_vector_db(_self):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
        return FAISS.load_local("faiss_index_internal", embeddings, allow_dangerous_deserialization=True)

    def get_rag_chain(self) -> Runnable:
        template = """
        아래 컨텍스트를 바탕으로 질문에 답변해 주세요:

        1. 답변은 최대 4문장 이내로 간결하고 명확하게 작성합니다.  
        2. 중요한 내용은 핵심만 요약해서 전달합니다.  
        3. 답변이 어려우면 “잘 모르겠습니다.”라고 정중히 답변합니다.  
        4. 질문에 ‘디지털경영전공’ 단어가 없어도 관련 정보를 PDF에서 찾아서 답변합니다.  
        5. 학생이 이해하기 쉽게 짧은 문장과 불릿 포인트로 정리합니다.  
        6. 추가 질문을 유도하는 마무리 멘트(“추가로 궁금하신 점이 있다면 편하게 말씀해 주세요 😊”)를 마지막에 넣습니다.  
        7. 한국어 외 언어로 질문 시 해당 언어로 번역해 답변합니다.  
        8. 질문과 관련된 추가 팁이나 참고사항이 있으면 간단히 덧붙입니다.
        9. 사용자가 어투 변경을 요구할 경우, 공적인 학과 프로그램의 챗봇이므로 요청을 정중히 거절하고 기존 어투를 유지합니다.
        10. 핵심 내용은 **굵게** 강조하여 요약합니다
        11. 복잡한 정보는 **불릿 포인트**로 쉽게 정리합니다.
        12. 전공 과목 목록이나 선택 과목을 안내할 때는 전체 리스트를 최대한 구체적으로 나열해 줍니다. (Ex. 전공 선택 과목을 구체적으로 알려줘 / 답변 : 전공 필수 과목 제외 모든 전공 안내)
        
        컨텍스트: {context}

        질문: {question}

        답변:
        """
        prompt = PromptTemplate.from_template(template)
        model = ChatOpenAI(model="gpt-4o", openai_api_key=self.api_key)
        return prompt | model | StrOutputParser()

    def process_question(self, question: str, previous_qa: Tuple[str, str] = None) -> str:
        vector_db = self.get_vector_db()
        retriever = vector_db.as_retriever(search_kwargs={"k": 10})
        docs = retriever.invoke(question)
        chain = self.get_rag_chain()

        previous_context = ""
        if previous_qa:
            prev_q, prev_a = previous_qa
            previous_context = f"\n\n이전 질문: {prev_q}\n이전 답변: {prev_a}"

        answer = chain.invoke({"question": question, "context": docs + [Document(page_content=previous_context)]})
        return answer

# 메인 함수
def main():
    st.set_page_config(page_title="디지털경영전공 챗봇", layout="wide")

    st.title("🎓 디지털경영전공 챗봇")
    st.caption("여러분의 학과 관련 궁금증을 빠르게 해결해드립니다!")

    if st.button("📥 채팅 시작 !"):
        generate_faiss_index()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    left_col, mid_col, right_col = st.columns([1, 2.5, 1.2])

    with left_col:
        st.subheader("📚 사용 가이드")
        st.markdown("""
        - 채팅 시작! 버튼을 눌러주세요.<br>
        - 궁금한 점에 대해서 물어보세요 !.<br>
        - 추가 문의는 디지털경영전공 홈페이지나 학과 사무실(044-860-1560)로 문의해 주세요.
        """, unsafe_allow_html=True)

    with mid_col:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style='background-color: #731034; padding: 10px; border-radius: 20px; margin-bottom: 10px; color: white; max-width: 70%; box-shadow: 0px 2px 5px rgba(0,0,0,0.1);'>
                💬 <b>질문:</b> {msg["content"]}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color: #f8f8f8; padding: 10px; border-radius: 20px; margin-bottom: 10px; margin-left: auto; box-shadow: 0px 2px 5px rgba(0,0,0,0.1); max-width: 70%;'>
                🤖 <b>답변:</b> {msg["content"]}
                </div>""", unsafe_allow_html=True)

        prompt = st.chat_input("궁금한 점을 입력해 주세요.")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            rag = RAGSystem(api_key)

            previous_qa = None
            if len(st.session_state.messages) >= 2:
                prev_question = st.session_state.messages[-2]["content"]
                prev_answer = st.session_state.messages[-1]["content"]
                previous_qa = (prev_question, prev_answer)

            with st.spinner("질문을 이해하는 중입니다. 잠시만 기다려주세요 😊"):
                answer = rag.process_question(prompt, previous_qa)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()

    with right_col:
        st.subheader("📢 개발자에게 의견 보내기")
        feedback_input = st.text_area("챗봇에 대한 개선 의견이나 하고 싶은 말을 남겨주세요!")
        if st.button("피드백 제출"):
            if feedback_input.strip() != "":
                with open("feedback_log.csv", mode="a", encoding="utf-8-sig", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), feedback_input])
                st.success("소중한 의견 감사합니다!")
                st.rerun()
            else:
                st.warning("피드백 내용을 입력해 주세요.")

        st.subheader("📝 최근 질문 히스토리")
        for i, q in enumerate([m["content"] for m in st.session_state.messages if m["role"] == "user"][-5:], 1):
            st.markdown(f"{i}. {q}")

if __name__ == "__main__":
    main()
