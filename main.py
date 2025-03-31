import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitters import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.memory import ConversationSummaryMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from typing import List, Tuple
import os
import csv
import time

# API 키 로드
time.sleep(1)
api_key = st.secrets["openai"]["API_KEY"]

# PDF 처리 클래스 정의
class PDFProcessor:
    #PDF를 문서 list로 변환
    @staticmethod
    def pdf_to_documents(pdf_path: str) -> List[Document]:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        for d in documents:
            d.metadata['file_path'] = pdf_path
        return documents
    @staticmethod
    def chunk_documents(documents: List[Document]) -> List[Document]:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
        splitter = SemanticChunker(embeddings=embeddings, chunk_size=800)
        return splitter.split_documents(documents)

# PDF 인덱스 생성 함수
def generate_faiss_index():
    pdf_dir = "data/"
    all_documents = []
    
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        st.warning("data/ 폴더가 생성되었습니다. PDF 파일을 여기에 넣고 다시 실행해주세요.")
        return

    pdf_files = [file for file in os.listdir(pdf_dir) if file.endswith(".pdf")]
    if not pdf_files:
        st.error("data/ 폴더에 PDF 파일이 없습니다. PDF를 추가한 후 다시 실행해주세요.")
        return

    #PDF 파일 문서화
    for file_name in pdf_files:
        docs = PDFProcessor.pdf_to_documents(os.path.join(pdf_dir, file_name))
        all_documents.extend(docs)

    #문서 chunking, vector embedding 생성, 인덱싱
    chunks = PDFProcessor.chunk_documents(all_documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index_internal")
    st.success(f"{len(pdf_files)}개의 PDF 파일로 인덱스 생성이 완료되었습니다.")

#RAG
class RAGSystem:
    def __init__(self, api_key: str):
        self.api_key = api_key

        # LLM 초기화
        self.llm = ChatOpenAI(
            model="gpt-4o", 
            openai_api_key=self.api_key, 
            temperature=0
        )
        
        # 대화 요약 메모리 추가
        self.memory = ConversationSummaryMemory(
            llm=self.llm, 
            return_messages=True,
            max_token_limit=300,
            memory_key="history",
            input_key="input",
            output_key="output"
        )

        # RAG chain 구성
        self.rag_chain = self.get_rag_chain()

    # vector DB 불러오기
    @st.cache_resource
    def get_vector_db(_self):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_self.api_key)
        return FAISS.load_local("faiss_index_internal", embeddings, allow_dangerous_deserialization=True)

    # prompt template 구성 + RAG chain 구성
    def get_rag_chain(self) -> Runnable:
        template = """
        아래 컨텍스트와 대화 기록을 바탕으로 질문에 답변해 주세요:

        1. 답변은 최대 4문장 이내로 간결하고 명확하게 작성합니다.
        2. 중요한 내용은 핵심만 요약해서 전달합니다.
        3. 답변이 어려우면 "잘 모르겠습니다."라고 정중히 답변합니다.
        4. 질문에 '디지털경영전공' 단어가 없어도 관련 정보를 PDF에서 찾아서 답변합니다.
        5. 이해하기 쉬운 짧은 문장과 불릿 포인트로 정리합니다.
        6. 마지막에 "추가로 궁금하신 점이 있다면 언제든지 말씀해주세요."라고 안내합니다.
        7. 한국어 외 언어로 질문 시 해당 언어로 번역하여 답변합니다.
        8. 관련된 참고 사항이 있다면 간단히 덧붙입니다.
        9. 챗봇 어투는 항상 친절하고 단정하게 유지합니다.
        10. 핵심 내용은 **굵게** 표시해 강조합니다.
        11. 복잡한 정보는 **불릿 포인트**로 요약 정리합니다.
        12. 전공 과목 안내 시에는 전체 리스트를 구체적으로 나열합니다.
        13. 추가 안내는 "추가로 궁금한 점이 있다면 언제든지 말씀해주세요."로 마무리합니다.
        14. 같은 말을 반복하지 마세요

        이전 대화 요약: {history}
        
        컨텍스트: {context}
        질문: {question}

        답변:
        """
        prompt = PromptTemplate.from_template(template)
        return prompt | self.llm | StrOutputParser()

    # 사용자의 질문을 처리하고 답변 반환
    def process_question(self, question: str) -> str:

        # 관련 문서 검색
        vector_db = self.get_vector_db()
        retriever = vector_db.as_retriever(search_kwargs={"k": 10})
        docs = retriever.invoke(question)

        # 대화 기록 요약 불러오기
        history_summary = self.memory.load_memory_variables({})['history']

        # LLM 호출
        answer = self.rag_chain.invoke({
            "question": question,
            "context": docs,
            "history": history_summary,
        })

        # 대화 내용을 memory에 저장
        self.memory.save_context({"input": question}, {"output": answer})

        return answer

def main():
    st.set_page_config(page_title="디지털경영전공 챗봇", layout="wide")
    st.title("🎓 디지털경영전공 챗봇")
    st.caption("학과에 대한 다양한 질문에 친절하게 답변해드립니다.")

    if "step" not in st.session_state:
        st.session_state.step = "init"
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 🎯 사이드바: 버튼 이동 및 설정
    with st.sidebar:
        st.header("📂 설정")
        if st.button("📥 채팅 시작 !"):
            generate_faiss_index()
            st.toast("PDF 인덱스 생성 완료!", icon="✅")
            st.session_state.step = "chat"
            st.rerun()

        st.divider()
        st.markdown("🧾 [디지털경영전공 홈페이지](https://example.com)")
        st.markdown("📞 학과 사무실: 044-860-1560")

    # 단계 분기
    if st.session_state.step == "init":
        st.info("📥 채팅 시작 버튼을 눌러 챗봇을 활성화하세요!")

    elif st.session_state.step == "chat":
        # 채팅 출력
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 사용자 입력 처리
        user_input = st.chat_input("궁금한 점을 입력해 주세요.")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            rag = RAGSystem(st.secrets["openai"]["API_KEY"])

            with st.spinner("질문을 이해하는 중입니다. 잠시만 기다려주세요."):
                answer = rag.process_question(user_input)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()

#streamlit 앱 실행 시작
if __name__ == "__main__":
    main()
