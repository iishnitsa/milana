from cross_gpt import sql_exec, let_log, use_rag, get_embs, coll_exec, set_common_save_id, get_common_save_id, global_state
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
    # sql_exec(fetchone) unwraps single-column rows to a scalar string
    row = sql_exec('SELECT system_prompt FROM system_prompts WHERE chat_id=?', (chat_id,), fetchone=True)
    if row is None:
        system_prompt = None
    elif isinstance(row, (tuple, list)):
        system_prompt = row[0]
    else:
        system_prompt = row
    if not system_prompt:
        let_log(f"WARNING: system prompt not found for chat {chat_id}")
        system_prompt = ""

    if user_message is not None:
        # Собирает промпт через RAG конструктор (он использует use_rag внутри)
        final_prompt = rag_constructor(str(chat_id), system_prompt, user_message)
        return final_prompt, None
    else:
        # Собираем полную историю в строку для просмотра
        history_list = get_history(str(chat_id))
        history_lines = [f"{msg.get('role', '')}{msg.get('full_text', '')}" for msg in history_list]
        full_history = "".join(history_lines)
        return system_prompt, full_history

def _pair_ids(chat_id: int):
    if chat_id % 2 == 0:
        return chat_id - 1, chat_id
    return chat_id, chat_id + 1

def vectorize_message(vector_id: str, message_text: str, role: str, local_message: bool, chat_id: int) -> bool:
    """
    Пишет embedding в Chroma для уже существующей записи rag_messages.
    Возвращает True если удалось.
    """
    if not use_rag or not vector_id or not message_text:
        return False
    embedding = get_embs(message_text)
    if not embedding:
        let_log(f"[vectorize] skip empty embedding for vector_id={vector_id}")
        return False
    str_chat_id = str(chat_id)
    if not local_message:
        oper_id, exec_id = _pair_ids(chat_id)
        metadatas = [{'chat_id_1': str(oper_id), 'chat_id_2': str(exec_id), 'role': role, 'relevance_score': 0}]
    else:
        metadatas = [{'chat_id_1': str_chat_id, 'role': role, 'relevance_score': 0}]
    try:
        coll_exec(action="add", coll_name="rag_collection", ids=[vector_id], metadatas=metadatas, embeddings=[embedding])
        sql_exec("UPDATE rag_messages SET is_vectorized = 1 WHERE vector_id = ?", (vector_id,))
        let_log(f"Message {vector_id} vectorized and added to RAG (overflow/shared path)")
        return True
    except Exception as e:
        let_log(f"[vectorize] coll_exec failed for {vector_id}: {e}")
        return False

def vectorize_dropped_messages(chat_id, messages: list, overflow_happened: bool):
    """
    При переполнении векторизует сообщения, выпавшие из окна контекста.
    Shared (нужны обоим чатам / local_message=False): всегда при overflow.
    Internal (local): только если overflow_happened для этого чата.
    """
    if not use_rag or not overflow_happened or not messages:
        return
    for msg in messages:
        if msg.get('is_vectorized'):
            continue
        vector_id = msg.get('vector_id')
        text = msg.get('full_text') or ''
        role = msg.get('role') or ''
        if not vector_id or not text:
            continue
        # shared, если vector_id встречается у двух chat_id
        row = sql_exec(
            "SELECT COUNT(DISTINCT chat_id) FROM rag_messages WHERE vector_id = ?",
            (vector_id,), fetchone=True)
        is_shared = bool(row and row[0] and int(row[0]) > 1)
        # внутренние без переполнения — не трогаем (здесь overflow_happened уже True)
        # shared — векторизуем; local — тоже при overflow этого чата
        try:
            cid = int(chat_id) if str(chat_id).isdigit() else chat_id
        except Exception:
            cid = chat_id
        vectorize_message(vector_id, text, role, local_message=not is_shared, chat_id=cid)

def update_history(chat_id: int, message_text: str, role: str, vector_id='', local_message=True, force_vectorize=False):
    """
    Добавляет новое сообщение в историю (таблица rag_messages).

    При use_rag=True по умолчанию НЕ векторизуем сразу (только SQL),
    чтобы не раздувать Chroma. Векторизация — при переполнении
    (vectorize_dropped_messages) или force_vectorize=True.

    Shared (local_message=False): при force_vectorize/overflow векторизуется.
    Internal без overflow — embedding не пишется.

    If show_message_datetime: prefix full_text with [YYYY-MM-DD HH:MM:SS] so agents see time.
    """
    let_log(f"Updating history for chat: {chat_id} with role: {role}")
    str_chat_id = str(chat_id)

    # Agents see datetime on messages when option is on (not only human UI)
    try:
        if getattr(global_state, 'show_message_datetime', False) and message_text is not None:
            import time as _time
            ts = _time.strftime("%Y-%m-%d %H:%M:%S")
            # avoid double-prefix
            if not str(message_text).lstrip().startswith("["):
                message_text = f"[{ts}] {message_text}"
    except Exception:
        pass

    did_vectorize = False
    if use_rag:
        if not vector_id:
            set_common_save_id()
            vector_id = str(get_common_save_id())
        # Сразу в Chroma — только по явному force (обратная совместимость / критичные пути)
        if force_vectorize:
            did_vectorize = vectorize_message(vector_id, message_text, role, local_message, chat_id)
    elif not local_message:
        # RAG-off shared peer: still need vector_id so the stub row can resolve
        # full_text/role (otherwise history becomes "NoneNone" → FATAL only-system).
        # Does not touch Chroma / RAG-on path.
        if not vector_id:
            set_common_save_id()
            vector_id = str(get_common_save_id())

    is_vec_flag = bool(did_vectorize)

    # Вставляем запись в rag_messages (всегда)
    if not local_message:
        # Для парных чатов: одна запись с полным текстом, другая — только связь через vector_id
        oper_id, exec_id = _pair_ids(chat_id)
        sql_exec(
            "INSERT INTO rag_messages (chat_id, role, full_text, is_vectorized, vector_id, relevance_score) VALUES (?, ?, ?, ?, ?, ?)",
            (str(oper_id), role, message_text, is_vec_flag, vector_id, 0))
        sql_exec("INSERT INTO rag_messages (chat_id, vector_id) VALUES (?, ?)", (str(exec_id), vector_id))
        return vector_id
    else:
        sql_exec(
            "INSERT INTO rag_messages (chat_id, role, full_text, is_vectorized, vector_id, relevance_score) VALUES (?, ?, ?, ?, ?, ?)",
            (str_chat_id, role, message_text, is_vec_flag, vector_id, 0))
        return vector_id

def delete_history_messages_by_vector_ids(vector_ids: list):
    """Удаляет сообщения из rag_messages и Chroma по vector_id (синхрон с удалением из чата)."""
    if not vector_ids:
        return
    ids = [str(v) for v in vector_ids if v]
    if not ids:
        return
    let_log(f"Deleting history messages by vector_ids: {ids}")
    sql_exec(
        "DELETE FROM rag_messages WHERE vector_id IN ({})".format(','.join('?' * len(ids))),
        tuple(ids))
    if use_rag:
        try:
            coll_exec(action="delete", coll_name="rag_collection", ids=ids)
        except Exception as e:
            let_log(f"Chroma delete failed: {e}")

def delete_chat(chat_id: int):
    """Удаляет все данные, связанные с чатом: сообщения, системный промпт и векторы (если включён RAG)."""
    let_log(f"Deleting chat and all related data: {chat_id}")
    str_chat_id = str(chat_id)
    try:
        from cross_gpt import psm_pop
        psm_pop(chat_id)
    except Exception:
        global_state.psm_operator_person.pop(chat_id, None)
    # Удаляем из rag_messages
    sql_exec("DELETE FROM rag_messages WHERE chat_id = ?", (str_chat_id,))
    # Удаляем системный промпт
    sql_exec("DELETE FROM system_prompts WHERE chat_id = ?", (chat_id,))

    # Если use_rag включён, удаляем векторы из ChromaDB
    if use_rag:
        if chat_id % 2 != 0:  # оператор – удаляем всё, где он фигурирует # TODO: проверь
            coll_exec(action="delete", coll_name="rag_collection", filters={'$or': [{'chat_id_1': str_chat_id}, {'chat_id_2': str_chat_id}]})
        else:  # исполнитель – удаляем только локальные (где chat_id_1 = исполнитель и нет chat_id_2)
            coll_exec(action="delete", coll_name="rag_collection", filters={'$and': [{'chat_id_1': str_chat_id}, {'chat_id_2': {'$exists': False}}]})
