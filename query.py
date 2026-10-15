import os
from dotenv import load_dotenv # функция считывания .env
from google import genai # официальный SDK от гугл
import chromadb
from chromadb.utils import embedding_functions

# ввод запроса на человеческом языке, chroma_db переводит в вектор и находит среди всех README 3 самых близких по смыслу чанка. Gemini API генерирует ответ
# история/контекст = если пишется вопрос связанный с предыдущими, он переписывается Gemini в полноценный независимый вопрос, потом как обычно ChromaDB ищет ответ, и Gemini делает удобоваримую формулировку

# загрузка переменных окружения и API-ключ Genai
load_dotenv() # считывает .env в текущей папке
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY не найден в файле .env!")

# инициализируем клиент Google GenAI (просто подключение к сервисам)
client = genai.Client(api_key=api_key)

# подключение локальной модели эмбеддингов
local_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# подключение к локальной chromadb
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="github_readmes", embedding_function=local_embedder) # подключение к уже существующей таблице github_readmes из ingest.py

# реализация истории сообщений. список сообщений для Gemini для сохранения контекста
chat_history = []

def contextualize_query(history: list, current_question: str) -> str:
    """
    Если пользователь задал уточняющий вопрос (например, "А какие там технологии?"),
    то эта функция превращает его в поисковый запрос с учетом истории.
    """
    if not history:
        return current_question

    # собираем последнюю пару сообщений для контекста
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-4:]]) # брать только 4 последних элемента списка

    prompt = f"""Учитывая историю диалога ниже, перепиши последний вопрос пользователя в
    самостоятельный, полный поисковый запрос, понятный без контекста. Не отвечай на вопрос, только перепиши его.
    История диалога:
    {history_text}
    Последний вопрос пользователя: {current_question}
    Самостоятельный поисковый запрос:"""
    
    response = client.models.generate_content(model = "gemini-3.5-flash", contents=prompt)
    return response.text.strip()

def ask_rag(user_query: str):
    print(f"\nВопрос: {user_query}")

    # с учетом истории превращаем вопрос в поисковый запрос для базы
    search_query = contextualize_query(chat_history, user_query)
    if search_query != user_query:
        print(f"Сформирован поисковый запрос для базы: {search_query}")

    print("Поиск релевантных чанков в ChromaDB...")

    # поиск 3 самых подходящих чанков текста
    results = collection.query(query_texts=[user_query], n_results=3) # переводит user_query в вектор из 384 чисел, сравнивает угол вектора вопроса и всех чанков в базе, дает 3 ближайших

    retrieved_docs = results["documents"][0]
    retrieved_meta = results["metadatas"][0]
    retrieved_distances = results['distances'][0] # добавка дистанций (число будет показывать насколько близок ответ к запросу)

    if not retrieved_docs:
        print("В базе не найдено подходящей информации")
        return

    # собираем найденный текст в одну строку. сборка промпта для отправки в Gemini
    context_text = "\n\n---\n\n".join([f"[Источник: {meta['source']}]\n{doc}" for doc, meta in zip(retrieved_docs, retrieved_meta)]) # простой генератор списков "Источник - информация"

    # формируем историю для модели генерации ответов
    # передаем всю историю диалога чтобы она помнила контекст
    history_prompt_part = ""

    # берем только последине 6 сообщений (3 вопроса + 3 ответа). и проходимся по recent вместо chat_history
    recent_history = chat_history[-6:]
    for msg in recent_history:
        role_label = "Пользователь" if msg["role"] == "user" else "Ассистент" # роли нужны чтобы модель понимала где вопрос а где ответ и что на свои ответы можно опираться
        history_prompt_part += f"{role_label}: {msg['content']}\n"

    # формируем промпт для Gemini
    prompt = f"""Ты — умный ассистент по проектам пользователя. 
        Используй историю диалога и предоставленный ниже контекст из README-файлов, чтобы ответить на последний вопрос.
        Если в контексте нет ответа, так и скажи.

        ИСТОРИЯ ДИАЛОГА:
        {history_prompt_part}

        НОВЫЙ КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
        {context_text}

        ПОСЛЕДНИЙ ВОПРОС ПОЛЬЗОВАТЕЛЯ:
        {user_query}
    """

    print("Отправка запроса в Gemini API...")

    # Запрашиваем ответ у модели Gemini 1.5 Flash
    response = client.models.generate_content(model = "gemini-3.5-flash", contents = prompt)
    answer = response.text

    # ответ модели, под ним список файлов, из которых ChromaDB взяла данные
    print("\n" + "=" * 40) # 40 раз =
    print("ОТВЕТ GEMINI:")
    print("="*40)
    print(answer)
    print("="*40)

    print("\nНайденные источники")
    for meta, dist in zip(retrieved_meta, retrieved_distances):
        print(f" - {meta['source']} (чанк {meta['chunk_index']}) | Коэффициент расстояния: {dist:.4f}") # заодно и выведем distance 

    # Сохранение истории!
    chat_history.append({"role": "user", "content": user_query})
    chat_history.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    # интерактивный цикл вопросов
    while True:
        user_query = input("\nЗадай вопрос по твоим README (или напиши 'exit')")
        if user_query.lower() in ["exit", "quit", "выход"]:
            break
        if user_query.strip(): # если строка не пустая, то запускаем
            ask_rag(user_query)
            