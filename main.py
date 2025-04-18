import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableMap, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.memory import ConversationSummaryMemory
from typing import List
import os
import csv
import time

# LangSmith 관련 패키지
from langsmith import Client

# API 키 로드
time.sleep(1)
api_key = st.secrets["openai"]["API_KEY"]

# LangSmith API 키 및 프로젝트 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["langsmith"]["API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = "디지털경영전공_챗봇"

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

# PDF 인덱스 생성
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

    for file_name in pdf_files:
        docs = PDFProcessor.pdf_to_documents(os.path.join(pdf_dir, file_name))
        all_documents.extend(docs)

    chunks = PDFProcessor.chunk_documents(all_documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index_internal")
    st.success(f"{len(pdf_files)}개의 PDF 파일로 인덱스 생성이 완료되었습니다.")

# RAG 시스템
class RAGSystem:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.langsmith_client = Client()

        self.llm = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=self.api_key,
            temperature=0,
            tags=["디지털경영전공_챗봇", "llm_call"]
        )

        self.memory = ConversationSummaryMemory(
            llm=self.llm,
            return_messages=True,
            max_token_limit=300,
            memory_key="history",
            input_key="input",
            output_key="output"
        )

        self.rag_chain = self.get_rag_chain()

    @st.cache_resource
    def get_vector_db(_self):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_self.api_key)
        return FAISS.load_local("faiss_index_internal", embeddings, allow_dangerous_deserialization=True)

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

        def retrieve_context(inputs: dict):
            vector_db = self.get_vector_db()
            retriever = vector_db.as_retriever(search_kwargs={"k": 7})
            return retriever.invoke(inputs["question"])

        return RunnableMap({
            "question": lambda x: x["question"],
            "context": RunnableLambda(retrieve_context).with_config(tags=["retriever"]),
            "history": lambda x: x["history"]
        }) | prompt.with_config(tags=["prompt"]) \
          | self.llm.with_config(tags=["llm"]) \
          | StrOutputParser().with_config(tags=["output_parser"]) \
          .with_config(tags=["디지털경영전공_챗봇", "rag_chain"], run_name="디지털경영전공_RAG_전체체인")

    def process_question(self, question: str) -> str:
        try:
            vector_db = self.get_vector_db()
            retriever = vector_db.as_retriever(search_kwargs={"k": 7})
            docs = retriever.invoke(question)
            history_summary = self.memory.load_memory_variables({})['history']

            answer = self.rag_chain.invoke(
                {
                    "question": question,
                    "context": docs,
                    "history": history_summary,
                },
                config={
                    "run_name": "디지털경영전공_챗봇_질의응답",
                    "metadata": {"question_text": question}
                }
            )

            self.memory.save_context({"input": question}, {"output": answer})

            try:
                self.evaluate_response(question, answer)
            except:
                pass

            return answer

        except Exception as e:
            st.error(f"질문 처리 중 오류가 발생했습니다: {str(e)}")
            return "죄송합니다. 질문을 처리하는 도중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."

    def evaluate_response(self, question: str, answer: str):
        try:
            eval_chain = ChatOpenAI(
                model="gpt-4o",
                openai_api_key=self.api_key,
                temperature=0
            ) | StrOutputParser()

            eval_prompt = f"""
            다음 질문과 답변의 품질을 1-10점 사이로 평가해주세요:

            질문: {question}
            답변: {answer}

            점수만 숫자로 반환해주세요.
            """
            score = eval_chain.invoke(eval_prompt, config={"tags": ["평가", "quality_check"]})
            score_value = float(score.strip()) if score.strip().replace('.', '', 1).isdigit() else 5.0
            print(f"응답 품질 점수: {score_value}/10")

        except Exception as e:
            print(f"평가 중 오류 발생: {e}")
            pass

# Streamlit 메인 앱
def main():
    st.set_page_config(page_title="디지털경영전공 챗봇", layout="wide")
    st.title("🎓 디지털경영전공 챗봇")
    st.caption("학과에 대한 다양한 질문에 친절하게 답변해드립니다.")

    if st.sidebar.checkbox("개발 모드 보기"):
        st.sidebar.markdown("### LangSmith 대시보드")
        st.sidebar.markdown("[대시보드 열기](https://smith.langchain.com)")
        st.sidebar.info("LangSmith 대시보드에서 챗봇의 동작과 성능을 모니터링할 수 있습니다.")

    if st.button("📥 채팅 시작 !"):
        generate_faiss_index()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    left_col, mid_col, right_col = st.columns([1, 2.5, 1.2])

    with left_col:
        st.subheader("📚 사용 가이드")
        st.markdown("""
        - '📥 채팅 시작 !' 버튼을 눌러주세요.<br>
        - 궁금한 내용을 입력하시면 관련 정보를 PDF 기반으로 안내해드립니다.<br>
        - 추가 문의는 디지털경영전공 홈페이지 또는 학과 사무실(044-860-1560)로 연락 바랍니다.
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
            try:
                rag = RAGSystem(st.secrets["openai"]["API_KEY"])
                with st.spinner("질문을 이해하는 중입니다. 잠시만 기다려주세요."):
                    answer = rag.process_question(prompt)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = "죄송합니다. 시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.error(f"오류 발생: {str(e)}")
            st.rerun()

    with right_col:
        st.subheader("📢 개발자에게 의견 보내기")
        feedback_input = st.text_area("챗봇에 대한 개선 의견이나 하고 싶은 말을 남겨주세요.")
        if st.button("피드백 제출"):
            if feedback_input.strip() != "":
                with open("feedback_log.csv", mode="a", encoding="utf-8-sig", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), feedback_input])
                st.success("소중한 의견 감사합니다.")
                st.rerun()
            else:
                st.warning("피드백 내용을 입력해 주세요.")

        st.subheader("📝 최근 질문 히스토리")
        for i, q in enumerate([m["content"] for m in st.session_state.messages if m["role"] == "user"][-5:], 1):
            st.markdown(f"{i}. {q}")

# 앱 실행
if __name__ == "__main__":
    main()
