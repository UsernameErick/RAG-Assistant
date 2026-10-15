import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" # Отключаем предупреждение о симлинках HuggingFace
from pathlib import Path
import chromadb # векторная библиотека, хранит тексты и их числовые отпечатки
from chromadb.utils import embedding_functions # функции автоматического перевода текста в векторы

# берутся readme, нарезаются на удобные чанки, переводим в векторы, складываем в специальную базу данных - коллекцию, создается chroma_db

# настройка модели эмбеддингов. модель весит ~90мб
local_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2") # модель берет любую фразу и превращает ее в массив из 384 чисел

# инициализируем ChromaDB с сохранением на диск
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# передаем функцию эмбеддингов прямо в коллекцию
collection = chroma_client.get_or_create_collection(name="github_readmes", embedding_function=local_embedder) # коллекция - аналог бд. сразу с embedder для автоперевода в числа

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]: # нарезка на чанки по 1000 символов, overlap - захватывает последние 200 симв. предыдущего, чтобы не терять контекст
    """Нарезка текста на куски с перекрытием"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def process_readmes(): # поиск ридми-файлов в папке, нарезка чанков
    data_dir = Path("./data")
    if not data_dir.exists():
        print("Папка ./data не найдена. Создай ее и положи туда .md-файлы.")
        return

    md_files = list(data_dir.glob("*.md")) # находим все .md-файлы в папке /data
    if not md_files:
        print("В папке ./data нет файлов .md!")
        return

    documents = [] # тексты чанков
    metadatas = [] # имя файла-источника, индекс чанка {"source": file_path.name, "chunk_index": i}
    ids = [] # уникальный текстовый id каждого чанка
    doc_id = 0 # численный простой id

    print("Чтение и нарезка файлов...")
    for file_path in md_files: # все .md-файлы в data
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read() # получение текста ридми

        chunks = chunk_text(content) # нарезка .md-файла

        for i, chunk in enumerate(chunks):
            documents.append(chunk) # закидываем чанки в общую кучу
            metadatas.append({"source": file_path.name, "chunk_index": i})
            ids.append(f"doc_{doc_id}")
            doc_id += 1

    print(f"Создано {len(documents)} чанков из {len(md_files)} файлов.")
    print("Генерация локальных эмбеддингов через PyTorch и сохранение в ChromaDB...")

    # ChromaDB сама рассчитывает векторы через local_embedder и сохранит их
    collection.add(documents = documents, metadatas = metadatas, ids = ids) # прогон каждого чанка через pytorch превращая их в векторы
    print("Успешно! База данных заполнена локально.")

if __name__ == "__main__":
    process_readmes()
    