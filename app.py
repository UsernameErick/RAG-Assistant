import os 
from dotenv import load_dotenv
from google import genai
import chromadb
from chromadb.utils import embedding_functions
import streamlit as st

# файл streamlit. красивая веб-обертка для нашего чата

# загрузка конфигурации и ключей
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY не найден в файле .env!")
    st.stop()

client = genai.Client(api_key=api_key)

# инициализация базы данных и модели (кэш streamlit чтобы не загружать модель каждый клик)
@st.cache_resource
def init_chroma():
    local_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="github_readmes", embedding_function=local_embedder)
    return collection

collection = init_chroma()

# внешний вид страницы
st.set_page_config(page_title="Github READMEs RAG assistant", page_icon="📚", layout="centered")
st.title("GitHub README RAG Assistant")
st.write("Чат локальной базы знаний по проектам и Gemini")

# инициализация истории сессии для интерфейса и бэкенда
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# отрисовка предыдущий сообщений чата при перезагрузке страницы
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Источники и метрики сходства"):
                for src, dist in message["sources"]:
                    st.text(f"* {src} (дистанция: {dist:.4f})")

def contextualize_query(history, current_question):
    """Превращает уточняющий вопрос в самостоятельный запрос с учетом истории."""
    if not history:
        return current_question

    history_text = "\n".join([f"{msg["role"]}: {msg["content"]}" for msg in history[-4:]])
    prompt = f"""Учитывая историю диалога ниже, перепиши последний вопрос пользователя в самостоятельный, 
            полный поисковый запрос, понятный без контекста. Не отвечай на вопрос, только перепиши его.

            История диалога:
            {history_text}

            Последний вопрос пользователя: {current_question}

            Самостоятельный поисковый запрос:"""

    response = client.models.generate_content(model="gemini-3.5-flash", contents=prompt)
    return response.text.strip()

# обработка нового ввода от пользователя
if user_query := st.chat_input("Задай вопрос по твоим проектам..."):
    # выводим сообщение пользователя в интерфейс
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # генерация ответа ассистента
    with st.chat_message("assistant"):
        with st.spinner("Поиск информации в базе"):
            # контекстуализация запроса
            search_query = contextualize_query(st.session_state.chat_history, user_query)

            # поиск в chroma_db
            results = collection.query(query_texts = [search_query], n_results= 3)

            retrieved_docs = results["documents"][0] if results["documents"] else []
            retrieved_meta = results["metadatas"][0] if results["metadatas"] else []
            retrieved_distances = results["distances"][0] if results["distances"] else []

            if not retrieved_docs:
                answer = "В базе не найдено подходящей информации по запросу."
                sources_info = []
            else:
                # сборка контекста
                context_text = "\n\n---\n\n".join([f"[Источник: {meta['source']}]\n{doc}" for doc, meta in zip(retrieved_docs, retrieved_meta)])
                # сборка истории для промпта (скользящее окно)
                history_prompt_part = ""
                recent_history = st.session_state.chat_history[-6:]
                for msg in recent_history:
                    role_label = "Пользователь" if msg["role"] == "user" else "Ассистент"
                    history_prompt_part += f"{role_label}: {msg['content']}\n"

                prompt = f"""Ты — умный ассистент по проектам пользователя. 
                        Используй историю диалога и предоставленный ниже контекст из README-файлов, чтобы ответить на последний вопрос.
                        Если в контексте нет ответа, так и скажи.

                        ИСТОРИЯ ДИАЛОГА:
                        {history_prompt_part}

                        НОВЫЙ КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
                        {context_text}

                        ПОСЛЕДНИЙ ВОПРОС ПОЛЬЗОВАТЕЛЯ:
                        {user_query}"""


                # запрос к Gemini
                response_gemini = client.models.generate_content(model="gemini-3.5-flash", contents = prompt)
                answer = response_gemini.text
                sources_info = list(zip([m["source"] for m in retrieved_meta], retrieved_distances))

            # вывод ответа в чат
            st.markdown(answer)

            if sources_info:
                with st.expander("Источники и метрики сходства"):
                    for src, dist in sources_info:
                        st.text(f"* {src} (дистанция: {dist:.4f})")

    # сохраняем в историю сеанса
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources_info})
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

