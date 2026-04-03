from cross_gpt import sql_exec, coll_exec, ask_model, get_embs, get_token_limit, get_text_tokens_coefficient, let_log, text_cutter, global_state
import re

ENABLE_ON_THE_FLY_COMPRESSION = False

global_summary_tokens_percent = 4 # 400%
recent_summary_tokens_percent = 2 # 200%

# TODO: После активации RAG, предусмотреть механизм,
# позволяющий принудительно использовать полные/неотфильтрованные
# версии сообщений, если окажется, что они были сокращены/отмечены
# как мусор по ошибке. Это может быть полезно для исправления
# контекста "на лету".
# Создать обновляемую (корректируемую) систему утверждений, для каждого
# диалога отдельная база знаний (наверное), типа Москва - столица России.

def initialize_rag_database():
    """Создает все необходимые таблицы в SQLite, если они не существуют."""
    let_log("##### Инициализация таблиц базы данных RAG... #####")
    sql_exec('''
        CREATE TABLE IF NOT EXISTS rag_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            full_text TEXT NOT NULL,
            is_vectorized BOOLEAN DEFAULT FALSE,
            vector_id TEXT UNIQUE,
            relevance_score INTEGER DEFAULT 0,  -- Новое поле для системы баллов
            is_compressed BOOLEAN DEFAULT FALSE, -- LET-BL-VER-1.0: Флаг для сжатых сообщений (TODO)
            global_summary TEXT DEFAULT NULL,   -- Глобальная сводка диалога (хранится в сообщении с максимальным ID, охваченным сводкой)
            recent_summary TEXT DEFAULT NULL    -- Сводка последней темы (хранится в сообщении с максимальным ID, охваченным сводкой)
        );
    ''')
    let_log("##### Таблицы базы данных RAG готовы. #####\n")

def get_history(chat_id: str) -> list[dict]:
    """
    Получает историю всех сообщений для данного чата.
    LET-BL-VER-1.0: Добавлены vector_id, is_compressed, chat_id.
    """
    query = "SELECT id, role, full_text, is_vectorized, relevance_score, vector_id, is_compressed, chat_id FROM rag_messages WHERE chat_id = ? ORDER BY id ASC"
    rows = sql_exec(query, (chat_id,), fetchall=True)
    if not rows: 
        let_log(f"История не найдена для чата {chat_id}")
        return []
    history = [{
        'id': r[0], 
        'role': r[1], 
        'full_text': r[2], 
        'is_vectorized': r[3], 
        'relevance_score': r[4], 
        'vector_id': r[5],
        'is_compressed': r[6],
        'chat_id': r[7]
    } for r in rows]
    let_log(f"Получена история для чата {chat_id}: {len(history)} сообщений")
    return history

def is_context_overflow(context_text: str) -> bool:
    estimated_tokens = len(context_text) * get_text_tokens_coefficient()
    token_limit = get_token_limit()
    overflow = estimated_tokens > (token_limit - 2000) # TODO:
    let_log(f"##### Проверка переполнения контекста #####")
    let_log(f"Примерное количество токенов: {estimated_tokens}")
    let_log(f"Лимит токенов: {token_limit}")
    let_log(f"Переполнение: {overflow}\n")
    return overflow

def get_summary(chat_id: str, summary_type: str) -> tuple[int, str | None]:
    """
    Возвращает (message_id, summary_text) для последней сводки указанного типа.
    Если сводки нет, возвращает (0, None).
    """
    column = 'global_summary' if summary_type == 'global' else 'recent_summary'
    row = sql_exec(f"SELECT id, {column} FROM rag_messages WHERE chat_id = ? AND {column} IS NOT NULL ORDER BY id DESC LIMIT 1", (chat_id,), fetchone=True)
    if row and row[1]: return row[0], row[1]  # id, текст сводки
    return 0, None

def set_summary(chat_id: str, summary_type: str, message_id: int, summary_text: str):
    """
    Сохраняет сводку в сообщении с указанным id. Одновременно очищает поле сводки этого типа у всех других сообщений чата,
    чтобы гарантировать, что хранится только одна сводка каждого типа.
    """
    column = 'global_summary' if summary_type == 'global' else 'recent_summary'
    sql_exec(f"UPDATE rag_messages SET {column} = NULL WHERE chat_id = ?", (chat_id,))
    sql_exec(f"UPDATE rag_messages SET {column} = ? WHERE id = ?", (summary_text, message_id))
    let_log(f"Сохранена {summary_type} сводка в сообщении ID {message_id} для чата {chat_id}")

def create_hierarchical_summary(chat_id: str, messages: list[dict], summary_type: str, token_threshold: int = 2000):
    """
    Создает иерархическую сводку, основываясь на объеме текста в чанках.
    token_threshold - примерный лимит токенов на один чанк для суммаризации.
    Сохраняет сводку в сообщении с максимальным id из списка messages.
    """
    from cross_gpt import prompt_chunk_summary, prompt_global_summary, prompt_recent_summary, no_markdown_instruction
    let_log(f"##### [{chat_id}] Создание иерархической '{summary_type}' сводки на основе {len(messages)} сообщений #####")
    let_log(f"Обработка {len(messages)} сообщений с порогом токенов: {token_threshold}")
    if not messages: 
        let_log("Нет сообщений для суммаризации")
        return
    chunks = []
    current_chunk = []
    current_chunk_text_len = 0
    for msg in messages:
        msg_len = len(msg['full_text'])
        # Проверяем, не превысит ли добавление нового сообщения порог
        if current_chunk and (current_chunk_text_len + msg_len) * get_text_tokens_coefficient() > token_threshold:
            chunks.append(current_chunk)
            current_chunk = [msg]
            current_chunk_text_len = msg_len
        else:
            current_chunk.append(msg)
            current_chunk_text_len += msg_len
    if current_chunk: chunks.append(current_chunk) # Не забываем добавить последний чанк, если он остался
    let_log(f"Создано {len(chunks)} чанков для суммаризации")
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        chunk_text = "\n".join([f"{msg['role']}{msg['full_text']}" for msg in chunk])
        let_log(f"##### Суммаризация чанка {i+1}/{len(chunks)} #####")
        let_log(f"Текст чанка: {chunk_text}...")
        chunk_summary = ask_model(chunk_text, system_prompt=prompt_chunk_summary + '\n' + no_markdown_instruction).strip()
        if chunk_summary:
            chunk_summaries.append(chunk_summary)
            let_log(f"Сводка чанка: {chunk_summary}")
        else: let_log(f"Не удалось создать сводку для чанка {i+1}")
    if not chunk_summaries:
        let_log(f"##### [{chat_id}] Не удалось создать сводки чанков. #####\n")
        return
    summaries_text = "\n".join(chunk_summaries)
    if summary_type == 'global': final_prompt = prompt_global_summary
    else: final_prompt = prompt_recent_summary
    let_log("##### Создание финальной сводки #####")
    let_log(f"Сводки чанков: {summaries_text}")
    let_log(f"Финальный промпт: {final_prompt}\n- {summaries_text}")
    final_summary = ask_model(f"\n- {summaries_text}", system_prompt=final_prompt + '\n' + no_markdown_instruction).strip()
    if final_summary:
        max_id = max(msg['id'] for msg in messages)
        set_summary(chat_id, summary_type, max_id, final_summary)
        let_log(f"##### [{chat_id}] Сохранена новая иерархическая '{summary_type}' сводка (до ID {max_id}): {final_summary} #####\n")
    else: let_log(f"##### [{chat_id}] Не удалось создать финальную сводку #####\n")

def compact_messages_llm(messages_to_compact: list[dict]) -> str:
    """
    Принимает группу сообщений (или одно сообщение) и просит LLM сжать их в единую суть.
    """
    let_log(f"##### Сжатие {len(messages_to_compact)} сообщений #####")
    let_log(f"ID сообщений: {[m['id'] for m in messages_to_compact]}")
    let_log(f"##### ПРИШЕДШИЙ СПИСОК #####")
    let_log(f"{messages_to_compact}")
    text_to_compact = "".join([f"{msg['role']}{msg['full_text']}" for msg in messages_to_compact])
    compacted_text = text_cutter(text_to_compact)
    let_log(f"Сжатый текст: {compacted_text}")
    let_log("##### Сжатие завершено #####\n")
    return compacted_text

def _compress_message_in_db(message: dict) -> dict | None:
    """
    Сжимает ОДНО сообщение, обновляет его в БД и RAG, и возвращает обновленное сообщение.
    Вызывается из prompt_assembler при переполнении.
    """
    if not message or message.get('is_compressed'): return message
    msg_id = message['id']
    vector_id = message.get('vector_id')
    chat_id = message.get('chat_id')
    let_log(f"##### Сжатие 'на лету' сообщения ID: {msg_id} #####")
    compacted_text = text_cutter(message['full_text'], cut_message=True)
    if not compacted_text or compacted_text == message['full_text']:
        let_log(f"Сжатие не удалось или текст не изменился. Помечаем как 'is_compressed'.")
        sql_exec("UPDATE rag_messages SET is_compressed = TRUE WHERE id = ?", (msg_id,))
        message['is_compressed'] = True
        return message
    sql_exec("UPDATE rag_messages SET full_text = ?, is_compressed = TRUE WHERE id = ?", (compacted_text, msg_id))
    message['full_text'] = compacted_text
    message['is_compressed'] = True
    return message

def _calculate_tokens(text: str) -> int:
    """Приблизительное количество токенов в тексте."""
    return int(len(text) * get_text_tokens_coefficient())

def _get_sum_tokens_since(chat_id: str, since_id: int) -> int:
    """Сумма токенов сообщений с id > since_id."""
    rows = sql_exec("SELECT full_text FROM rag_messages WHERE chat_id = ? AND id > ?", (chat_id, since_id), fetchall=True)
    total = 0
    for (text,) in rows: total += _calculate_tokens(text)
    return total

def _get_last_messages_by_tokens(chat_id: str, token_threshold: int, after_id: int = 0) -> list[dict]:
    """
    Возвращает список сообщений (словарей), начиная с последнего, сумма токенов которых >= token_threshold.
    Ограничивается сообщениями с id > after_id (чтобы не включать уже охваченные сводкой).
    """
    rows = sql_exec("SELECT id, role, full_text FROM rag_messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,), fetchall=True)
    messages = []
    for r in rows:
        if r[0] > after_id: messages.append({'id': r[0], 'role': r[1], 'full_text': r[2]})
    result = []
    total_tokens = 0
    for msg in reversed(messages):
        tokens = _calculate_tokens(msg['full_text'])
        result.append(msg)
        total_tokens += tokens
        if total_tokens >= token_threshold: break
    result.reverse()
    return result

def _fit_documents_to_token_limit(documents: list[str], token_limit: int) -> str:
    if not documents: return ""
    token_coeff = get_text_tokens_coefficient()
    result_parts = []
    used_tokens = 0
    for doc in documents:
        doc_tokens = len(doc) * token_coeff
        if used_tokens + doc_tokens <= token_limit:
            result_parts.append(doc)
            used_tokens += doc_tokens
        else:
            # Нужно вписать часть текущего документа
            remaining_tokens = token_limit - used_tokens
            if remaining_tokens <= 0: break
            # Бинарный поиск по символам для приближения к лимиту токенов
            lo, hi = 0, len(doc)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if mid * token_coeff <= remaining_tokens: lo = mid
                else: hi = mid - 1
            truncated = doc[:lo]
            result_parts.append(truncated)
            break
    return "".join(result_parts)

def rerank_rag_results(chat_id: str, initial_results: dict) -> list[str]:
    # Заглушка
    if not initial_results: return []
    documents = initial_results.get('documents', [])
    if isinstance(documents, list) and documents and isinstance(documents[0], list): documents = documents[0]
    return documents

def prompt_assembler(chat_id: str, system_prompt: str, current_message: str, history: list[dict]) -> str:
    """
    Собирает финальный промпт. НЕ включает current_message в промпт и расчет токенов.
    Реализует RAG (при усечении), сжатие "на лету" и сохранение нечетного количества сообщений.
    """
    let_log(f"##### [{chat_id}] Сборка промпта (v7 - No doc duplication) #####")
    let_log(f"Текущее сообщение (не включается): {current_message}")
    let_log(f"Длина истории: {len(history)}")
    from cross_gpt import last_messages_marker, rag_context_marker, global_summary_marker, recent_summary_marker
    SAFETY_MARGIN = 2000
    RAG_RESERVE_PERCENTAGE = 0.15

    # Получаем лимит токенов
    token_limit = get_token_limit()

    # Получаем текущие сводки
    last_global_id, global_summary = get_summary(chat_id, 'global')
    last_recent_id, recent_summary = get_summary(chat_id, 'recent')

    # Базовые части промпта (со сводками)
    base_prompt_parts = [system_prompt]
    if global_summary:
        base_prompt_parts.append(f"{global_summary_marker}{global_summary}")
    if recent_summary:
        base_prompt_parts.append(f"{recent_summary_marker}{recent_summary}")
    base_prompt_str_no_rag = "".join(base_prompt_parts)
    base_tokens_no_rag = _calculate_tokens(base_prompt_str_no_rag)

    # Оцениваем, будет ли усечение истории
    available_for_history = token_limit - base_tokens_no_rag - SAFETY_MARGIN - int(token_limit * RAG_RESERVE_PERCENTAGE)
    total_history_tokens = _get_sum_tokens_since(chat_id, 0)
    history_was_truncated = total_history_tokens > available_for_history

    # Проверяем и создаём глобальную сводку
    new_global_tokens = _get_sum_tokens_since(chat_id, last_global_id)
    if (history_was_truncated or last_global_id > 0) and new_global_tokens >= global_summary_tokens_percent * token_limit:
        let_log(f"##### [{chat_id}] Условие для глобальной сводки выполнено. Создаём... #####")
        # Собираем сообщения для суммаризации: старая сводка + новые сообщения
        messages_for_global = []
        if last_global_id > 0:
            # Добавляем старую сводку как псевдо-сообщение
            messages_for_global.append({'id': last_global_id, 'role': 'global_summary', 'full_text': global_summary})
        # Добавляем новые сообщения (id > last_global_id)
        rows = sql_exec("SELECT id, role, full_text FROM rag_messages WHERE chat_id = ? AND id > ? ORDER BY id ASC", (chat_id, last_global_id), fetchall=True)
        for r in rows:
            messages_for_global.append({'id': r[0], 'role': r[1], 'full_text': r[2]})
        create_hierarchical_summary(chat_id, messages_for_global, 'global')
        # Обновляем сводки после создания
        last_global_id, global_summary = get_summary(chat_id, 'global')
        last_recent_id, recent_summary = get_summary(chat_id, 'recent')
        # Пересобираем базовые части с новыми сводками
        base_prompt_parts = [system_prompt]
        if global_summary:
            base_prompt_parts.append(f"{global_summary_marker}{global_summary}")
        if recent_summary:
            base_prompt_parts.append(f"{recent_summary_marker}{recent_summary}")
        base_prompt_str_no_rag = "".join(base_prompt_parts)
        base_tokens_no_rag = _calculate_tokens(base_prompt_str_no_rag)

    # Проверяем и создаём недавнюю сводку
    new_recent_tokens = _get_sum_tokens_since(chat_id, last_recent_id)
    if (history_was_truncated or last_recent_id > 0) and new_recent_tokens >= recent_summary_tokens_percent * token_limit:
        let_log(f"##### [{chat_id}] Условие для недавней сводки выполнено. Создаём... #####")
        # Собираем сообщения для суммаризации: только новые сообщения после последней недавней сводки
        messages_for_recent = []
        rows = sql_exec("SELECT id, role, full_text FROM rag_messages WHERE chat_id = ? AND id > ? ORDER BY id ASC", (chat_id, last_recent_id), fetchall=True)
        for r in rows:
            messages_for_recent.append({'id': r[0], 'role': r[1], 'full_text': r[2]})
        create_hierarchical_summary(chat_id, messages_for_recent, 'recent')
        # Обновляем сводки после создания
        last_global_id, global_summary = get_summary(chat_id, 'global')
        last_recent_id, recent_summary = get_summary(chat_id, 'recent')
        # Пересобираем базовые части с новыми сводками
        base_prompt_parts = [system_prompt]
        if global_summary:
            base_prompt_parts.append(f"{global_summary_marker}{global_summary}")
        if recent_summary:
            base_prompt_parts.append(f"{recent_summary_marker}{recent_summary}")
        base_prompt_str_no_rag = "".join(base_prompt_parts)
        base_tokens_no_rag = _calculate_tokens(base_prompt_str_no_rag)

    # Теперь продолжаем сборку промпта как раньше, используя актуальные сводки
    RAG_RESERVE_TOKENS = int(token_limit * RAG_RESERVE_PERCENTAGE)
    available_tokens_for_history = token_limit - base_tokens_no_rag - SAFETY_MARGIN - RAG_RESERVE_TOKENS
    let_log(f"##### [{chat_id}] Расчет лимитов для истории (резерв RAG: {RAG_RESERVE_TOKENS:.0f}) #####")
    let_log(f"Лимит токенов (общий): {token_limit}")
    let_log(f"Токены (База): {base_tokens_no_rag:.0f}")
    let_log(f"Токены (Safety Margin): {SAFETY_MARGIN}")
    let_log(f"Доступно для истории: {available_tokens_for_history:.0f}\n")

    history_strings_list = []
    history_token_count = 0
    history_included_vector_ids = []
    mutable_history = list(history)
    original_history_length = len(history)

    if available_tokens_for_history <= 0:
        let_log(f"##### [{chat_id}] ВНИМАНИЕ: Нет места для истории. Пропускаем. #####")
    elif mutable_history:
        let_log(f"######################## ИСТОРИЯ (v7, {len(mutable_history)} сообщ.) ###################")
        for msg in reversed(mutable_history):
            msg_role = msg.get('role')
            msg_content = msg.get('full_text', '')
            msg_str = f"{msg_role}{msg_content}"
            msg_tokens = _calculate_tokens(msg_str)
            if (history_token_count + msg_tokens) <= available_tokens_for_history:
                history_strings_list.append(msg_str)
                history_token_count += msg_tokens
                if msg.get('vector_id'): history_included_vector_ids.append(msg['vector_id'])
            else:
                let_log(f"Сообщение ID {msg['id']} (токены: {msg_tokens:.0f}) не помещается.")
                if ENABLE_ON_THE_FLY_COMPRESSION and not msg.get('is_compressed'):
                    let_log(f"Запуск сжатия 'на лету' для ID {msg['id']}...")
                    compressed_msg = _compress_message_in_db(msg)
                    msg_content_compressed = compressed_msg.get('full_text', '')
                    msg_str_compressed = f"{msg_role}{msg_content_compressed}"
                    msg_tokens_compressed = _calculate_tokens(msg_str_compressed)
                    if (history_token_count + msg_tokens_compressed) <= available_tokens_for_history:
                        let_log(f"Сжатое сообщение ID {msg['id']} (токены: {msg_tokens_compressed:.0f}) теперь помещается.")
                        history_strings_list.append(msg_str_compressed)
                        history_token_count += msg_tokens_compressed
                        if msg.get('vector_id'): history_included_vector_ids.append(msg['vector_id'])
                    else:
                        let_log(f"Даже сжатое сообщение ID {msg['id']} (токены: {msg_tokens_compressed:.0f}) не помещается. Усечение.")
                        break
                else:
                    let_log("Достигнут лимит токенов. Усечение.")
                    break
        history_strings_list.reverse()

    rag_prompt_part = ""
    history_was_truncated = len(history_strings_list) < original_history_length
    available_tokens_for_rag_actual = token_limit - base_tokens_no_rag - history_token_count - SAFETY_MARGIN

    if history_was_truncated and available_tokens_for_rag_actual > 0:
        let_log(f"##### [{chat_id}] RAG АКТИВИРОВАН. Доступно токенов: {available_tokens_for_rag_actual:.0f} #####")
        rag_token_limit_final = available_tokens_for_rag_actual
        recent_context = ""
        if len(history) >= 2:
            recent_context = "\n".join([f"{msg.get('full_text')}" for msg in history[-2:]])
        expanded_query = f"{recent_context}\n{current_message}"
        query_embedding = get_embs(expanded_query)
        rag_filters = {'chat_id': chat_id, '$nin': {'vector_id': history_included_vector_ids}}
        try:
            initial_results = coll_exec(
                action="query",
                coll_name="rag_collection",
                query_embeddings=[query_embedding],
                n_results=10,
                filters=rag_filters,
                fetch=["ids"]
            )
        except Exception as e:
            let_log(f"Ошибка coll_exec RAG: {e}")
            initial_results = None
        # Если есть ids, получаем тексты из SQLite
        retrieved_texts = []
        if isinstance(initial_results, dict) and initial_results.get('ids') and initial_results['ids']:
            # ids могут быть списком списков (из-за query_embeddings=[...])
            ids_list = initial_results['ids'][0] if isinstance(initial_results['ids'][0], list) else initial_results['ids']
            if ids_list:
                # Построим запрос IN
                placeholders = ','.join('?' * len(ids_list))
                query = f"SELECT vector_id, full_text FROM rag_messages WHERE chat_id = ? AND vector_id IN ({placeholders})"
                rows = sql_exec(query, (chat_id,) + tuple(ids_list), fetchall=True)
                if rows:
                    texts_map = {row[0]: row[1] for row in rows}
                    # Сохраняем порядок ids_list
                    retrieved_texts = [texts_map.get(vid, "") for vid in ids_list]
                    let_log(f"Получено {len(retrieved_texts)} текстов из SQLite")
                else: let_log("Не найдены тексты для полученных vector_id")
        else: let_log("RAG поиск не дал результатов или вернул некорректный формат")
        # Если есть тексты, передаём их в rerank_rag_results в ожидаемом формате
        if retrieved_texts:
            # Создаём структуру, похожую на старую, с ключом documents
            results_with_texts = {'documents': [retrieved_texts]}  # как было раньше
            top_docs = rerank_rag_results(chat_id, results_with_texts)
            if top_docs:
                retrieved_context = _fit_documents_to_token_limit(top_docs, rag_token_limit_final)
                rag_prompt_part = rag_context_marker + f"{retrieved_context}\n"
                let_log(f"Добавлен RAG контекст: {len(retrieved_context)} символов (токенов ~{rag_token_limit_final})")
            else: let_log("Не удалось получить переранжированные документы")
        else: let_log("RAG поиск не дал результатов или тексты не найдены")
    else:
        let_log(f"##### [{chat_id}] RAG НЕ АКТИВИРОВАН. Условие: (Усечение: {history_was_truncated}, Доступно места: {available_tokens_for_rag_actual > 0}) #####")

    # Обеспечиваем нечётное количество сообщений в истории
    while len(history_strings_list) > 1 and len(history_strings_list) % 2 == 0:
        removed_msg = history_strings_list.pop(0)
        let_log(f"Удалено самое старое сообщение для сохранения нечетности: {removed_msg[:100]}...")

    history_final_str = "".join(history_strings_list)
    final_prompt_parts = [
        base_prompt_str_no_rag,
        rag_prompt_part,
        last_messages_marker,
        history_final_str]
    final_prompt = "".join(final_prompt_parts)

    let_log(f"##### Финальный промпт собран #####")
    let_log(f"Общая длина промпта: {len(final_prompt)} символов")
    return final_prompt

def rag_constructor(chat_id: str, system_prompt: str, current_message: str) -> str:
    """
    Главная функция-оркестратор RAG.
    Получает историю сообщений из БД и собирает финальный промпт.
    """
    let_log("##### RAG CONSTRUCTOR ЗАПУЩЕН #####")
    let_log(f"Chat ID: {chat_id}")
    let_log(f"Длина системного промпта: {len(system_prompt)}")
    let_log(f"Текущее сообщение: {current_message}")

    # Получаем историю сообщений
    history = get_history(chat_id)

    # Собираем промпт
    final_prompt = prompt_assembler(
        chat_id=chat_id,
        system_prompt=system_prompt,
        current_message=current_message,
        history=history
    )

    let_log("##### RAG CONSTRUCTOR ЗАВЕРШЕН #####\n")
    return final_prompt