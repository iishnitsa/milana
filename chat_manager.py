from cross_gpt import sql_exec, let_log, use_rag, get_embs, coll_exec, set_common_save_id, get_common_save_id
from rag_constructor import rag_constructor, get_history

def create_chat(chat_id: int, prompt: str):
    """Сохраняет системный промпт для чата."""
    let_log(f"Creating chat session: {chat_id}")
    sql_exec('INSERT OR REPLACE INTO system_prompts (chat_id, system_prompt) VALUES (?, ?)', (chat_id, prompt))

def get_chat_context(chat_id: int, user_message=None):
    """
    Получает контекст для чата.
    - Если передан user_message: возвращает финальный промпт, собранный через RAG-конструктор.
    - Если user_message не передан: возвращает (system_prompt, полная_история_в_виде_строки).
    """
    let_log(f"Getting chat context for chat: {chat_id}")
    # Получаем системный промпт
    row = sql_exec('SELECT system_prompt FROM system_prompts WHERE chat_id=?', (chat_id,), fetchone=True)
    system_prompt = row[0] if row else None
    if not system_prompt:
        let_log(f"WARNING: system prompt not found for chat {chat_id}")
        system_prompt = ""

    if user_message is not None:
        # Собираем промпт через RAG конструктор (он использует use_rag внутри)
        final_prompt = rag_constructor(str(chat_id), system_prompt, user_message)
        return final_prompt, None
    else:
        # Собираем полную историю в строку для просмотра
        history_list = get_history(str(chat_id))
        history_lines = [f"{msg.get('role', '')}{msg.get('full_text', '')}" for msg in history_list]
        full_history = "".join(history_lines)
        return system_prompt, full_history

def update_history(chat_id: int, message_text: str, role: str, vector_id='', local_message=True):
    """
    Добавляет новое сообщение в историю (таблица rag_messages).
    Если use_rag == True и vector_id не передан, генерирует вектор и сохраняет его в ChromaDB.
    Параметр local_message управляет логикой привязки к другому чату (оператор/исполнитель).
    """
    let_log(f"Updating history for chat: {chat_id} with role: {role}")
    str_chat_id = str(chat_id)

    # Генерация вектора, если включён RAG и вектор не передан
    if use_rag and not vector_id:
        set_common_save_id()
        vector_id = str(get_common_save_id())
        embedding = get_embs(message_text)
        # Формируем метаданные в зависимости от local_message
        if not local_message:
            # Для парных чатов (оператор/исполнитель)
            if chat_id % 2 == 0:
                oper_id = chat_id
                exec_id = chat_id - 1
            else:
                oper_id = chat_id + 1
                exec_id = chat_id
            metadatas = [{'chat_id_1': str(oper_id), 'chat_id_2': str(exec_id), 'role': role, 'relevance_score': 0}]
        else:
            metadatas = [{'chat_id_1': str_chat_id, 'role': role, 'relevance_score': 0}]
        # Добавляем в ChromaDB
        coll_exec(action="add", coll_name="rag_collection", ids=[vector_id], metadatas=metadatas, embeddings=[embedding])
        let_log(f"Message {vector_id} vectorized and added to RAG")
    else:
        # Если use_rag выключен, вектор не генерируем, vector_id остаётся пустым
        pass

    # Вставляем запись в rag_messages (всегда)
    if not local_message:
        # Для парных чатов: одна запись с полным текстом, другая — только связь через vector_id
        if chat_id % 2 == 0:
            oper_id = chat_id
            exec_id = chat_id - 1
        else:
            oper_id = chat_id + 1
            exec_id = chat_id
        sql_exec("INSERT INTO rag_messages (chat_id, role, full_text, is_vectorized, vector_id, relevance_score) VALUES (?, ?, ?, ?, ?, ?)",
                 (str(oper_id), role, message_text, bool(use_rag), vector_id, 0))
        sql_exec("INSERT INTO rag_messages (chat_id, vector_id) VALUES (?, ?)", (str(exec_id), vector_id))
        return vector_id
    else:
        sql_exec("INSERT INTO rag_messages (chat_id, role, full_text, is_vectorized, vector_id, relevance_score) VALUES (?, ?, ?, ?, ?, ?)",
                 (str_chat_id, role, message_text, bool(use_rag), vector_id, 0))
        return vector_id

def delete_chat(chat_id: int):
    """Удаляет все данные, связанные с чатом: сообщения, системный промпт и векторы (если включён RAG)."""
    let_log(f"Deleting chat and all related data: {chat_id}")
    str_chat_id = str(chat_id)

    # Удаляем из rag_messages
    sql_exec("DELETE FROM rag_messages WHERE chat_id = ?", (str_chat_id,))
    # Удаляем системный промпт
    sql_exec("DELETE FROM system_prompts WHERE chat_id = ?", (chat_id,))

    # Если use_rag включён, удаляем векторы из ChromaDB
    if use_rag:
        if chat_id % 2 == 0:  # оператор – удаляем всё, где он фигурирует
            coll_exec(action="delete", coll_name="rag_collection", filters={'$or': [{'chat_id_1': str_chat_id}, {'chat_id_2': str_chat_id}]})
        else:  # исполнитель – удаляем только локальные (где chat_id_1 = исполнитель и нет chat_id_2)
            coll_exec(action="delete", coll_name="rag_collection", filters={'$and': [{'chat_id_1': str_chat_id}, {'chat_id_2': {'$exists': False}}]})