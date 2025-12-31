# chat_manager.py
# Этот модуль абстрагирует взаимодействие с базой данных чатов.

from cross_gpt import sql_exec, let_log, use_rag
# Импортируем RAG-функции, чтобы не было циклического импорта
from rag_constructor import initialize_rag_database, rag_constructor, get_history
from cross_gpt import get_embs, coll_exec
import uuid # Нужен для vector_id TODO:

# --- Версии функций для СТАНДАРТНОГО режима (use_rag = False) ---

def _standard_initialize_schema():
    """Создает таблицу 'chats' для стандартного режима."""
    let_log("Initializing standard chat schema...")
    sql_exec('''CREATE TABLE IF NOT EXISTS chats (
        chat_id INT PRIMARY KEY,
        prompt TEXT,
        history TEXT
    )''')

def _standard_create_chat(chat_id: int, prompt: str):
    """Сохраняет новую сессию чата в таблицу 'chats'."""
    let_log(f"Creating standard chat session: {chat_id}")
    from cross_gpt import last_messages_marker
    sql_exec(
        'INSERT OR REPLACE INTO chats (chat_id, prompt, history) VALUES (?, ?, ?)',
        (chat_id, prompt, last_messages_marker)
    )

def _standard_get_chat_context(chat_id: int) -> tuple[str, str]:
    """Получает системный промпт и всю историю из таблицы 'chats'."""
    let_log(f"Getting standard context for chat: {chat_id}")
    result = sql_exec('SELECT prompt, history FROM chats WHERE chat_id=?', (chat_id,), fetchone=True)
    return result if result else (None, None)

def _standard_update_history(chat_id: int, message_text: str, role: str):
    """
    Принимает новое сообщение и роль, добавляет их к существующей
    истории и перезаписывает всё поле 'history' в таблице 'chats'.
    """
    let_log(f"Updating standard history for chat: {chat_id}")
    rr = _standard_get_chat_context(chat_id)
    let_log(rr)
    _, old_history = rr
    if old_history is None:
        old_history = ''
        
    # Формируем новый фрагмент текста (роль уже содержит нужные отступы)
    new_fragment = f"{role}{message_text}"
    updated_history = old_history + new_fragment
    
    sql_exec("UPDATE chats SET history = ? WHERE chat_id = ?", (updated_history, chat_id))

def _standard_delete_chat(chat_id: int):
    """Удаляет чат из таблицы 'chats'."""
    let_log(f"Deleting standard chat: {chat_id}")
    sql_exec("DELETE FROM chats WHERE chat_id = ?", (chat_id,))


# --- Версии функций для RAG-режима (use_rag = True) ---

def _rag_initialize_schema():
    """Инициализирует все таблицы, необходимые для RAG."""
    let_log("Initializing RAG chat schema...")
    initialize_rag_database()
    # Также создаем старую таблицу 'chats' для хранения системного промпта.
    # Это компромисс для совместимости.
    _standard_initialize_schema()

def _rag_create_chat(chat_id: int, prompt: str):
    """В RAG режиме мы сохраняем системный промпт в таблицу 'chats' для будущего использования."""
    let_log(f"Starting RAG chat session: {chat_id}. System prompt is saved.")
    # Используем стандартную функцию, чтобы не дублировать код
    _standard_create_chat(chat_id, prompt)

def _rag_get_chat_context(chat_id: int, user_message=False) -> tuple[str, str]:
    """
    Собирает контекст.
    - Если user_message есть: вызывает rag_constructor для RAG-сборки.
    - Если user_message нет: собирает ПОЛНУЮ историю чата в одну строку для просмотра.
    """
    let_log(f"Getting RAG context for chat: {chat_id}, user_message provided: {bool(user_message)}")
    
    # 1. Системный промпт нужен в любом случае
    system_prompt, _ = _standard_get_chat_context(chat_id)
    
    if not system_prompt:
        let_log(f"КРИТИЧЕСКАЯ ОШИБКА: системный промпт для RAG-чата {chat_id} не найден.")
        return ("Ошибка: системный промпт не найден.", None) 

    # 2. Главная логика
    if user_message:
        # --- Сценарий 1: Идем в RAG ---
        final_prompt = rag_constructor(
            chat_id=str(chat_id),
            system_prompt=system_prompt,
            current_message=user_message
            )
        # Возвращаем (final_prompt, None)
        return (final_prompt, None)
    
    else:
        # --- Сценарий 2: Собираем историю в строку (по запросу) ---
        let_log(f"Assembling full history string for chat: {chat_id}, including roles.")
        
        # Получаем историю из rag_messages. 
        # chat_id в БД RAG хранится как TEXT.
        messages_list = get_history(str(chat_id)) 
        
        # Собираем в строку, парся роли (КАК ВЫ ПРОСИЛИ)
        history_lines = [
            f"{msg.get('role', 'UNKNOWN')}{msg.get('full_text', '')}" 
            for msg in messages_list
        ]
        
        # Собираем финальную историю
        full_history_string = "".join(history_lines)
        
        # Возвращаем (system_prompt, full_history_string)
        # system_prompt возвращается как первый элемент, история — как второй.
        return (system_prompt, full_history_string)

def _rag_update_history(chat_id: int, message_text: str, role: str):
    """
    Добавляет новое сообщение в 'rag_messages' И СРАЗУ векторизует его,
    добавляя в ChromaDB (согласно TODO).
    """
    let_log(f"Updating RAG history for chat: {chat_id} with role: {role}")
    
    str_chat_id = str(chat_id)
    vector_id = str(uuid.uuid4())
    
    # 1. Добавляем в SQL, но c is_vectorized = True и vector_id
    # (Мы считаем, что векторизация будет успешной)
    sql_exec(
        "INSERT INTO rag_messages (chat_id, role, full_text, is_vectorized, vector_id, relevance_score) VALUES (?, ?, ?, ?, ?, ?)",
        (str_chat_id, role, message_text, True, vector_id, 0) # relevance_score по умолчанию 0
    )
    
    # 2. Векторизуем и добавляем в ChromaDB
    try:
        embedding = get_embs(message_text)
        coll_exec(
            action="add", coll_name="rag_collection",
            ids=[vector_id],
            documents=[message_text],
            metadatas=[{'chat_id': str_chat_id, 'role': role, 'relevance_score': 0}],
            embeddings=[embedding]
        )
        let_log(f"Сообщение {vector_id} сразу векторизовано и добавлено в RAG.")
    except Exception as e:
        let_log(f"##### ОШИБКА: Не удалось векторизовать сообщение {vector_id} при сохранении: {e} #####")
        # Откатываем флаг в SQL, если векторизация не удалась
        sql_exec(
            "UPDATE rag_messages SET is_vectorized = FALSE WHERE vector_id = ?",
            (vector_id,)
        )

def _rag_delete_chat(chat_id: int):
    """Удаляет все данные, связанные с чатом, из всех таблиц RAG."""
    let_log(f"Deleting RAG chat and all related data: {chat_id}")
    str_chat_id = str(chat_id)
    sql_exec("DELETE FROM rag_messages WHERE chat_id = ?", (str_chat_id,))
    sql_exec("DELETE FROM summaries WHERE chat_id = ?", (str_chat_id,))
    # Удаляем и запись с системным промптом
    _standard_delete_chat(chat_id)
    # TODO: Добавьте сюда логику очистки ChromaDB, если это необходимо
    # coll_exec(action="delete", coll_name="rag_collection", filters={'chat_id': str_chat_id})


# --- Финальный выбор функций на основе флага use_rag ---

if use_rag:
    initialize_schema = _rag_initialize_schema
    create_chat = _rag_create_chat
    get_chat_context = _rag_get_chat_context
    update_history = _rag_update_history
    delete_chat = _rag_delete_chat
else:
    initialize_schema = _standard_initialize_schema
    create_chat = _standard_create_chat
    get_chat_context = _standard_get_chat_context
    update_history = _standard_update_history
    delete_chat = _standard_delete_chat