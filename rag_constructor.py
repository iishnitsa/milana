from cross_gpt import sql_exec, coll_exec, ask_model, get_embs, get_token_limit, get_text_tokens_coefficient, let_log, text_cutter, global_state, use_rag
import re

ENABLE_ON_THE_FLY_COMPRESSION = False

global_summary_tokens_percent = 4 # 400%
recent_summary_tokens_percent = 2 # 200%

def get_history(chat_id: str) -> list[dict]:
    query = "SELECT id, role, full_text, is_vectorized, relevance_score, vector_id, is_compressed, chat_id FROM rag_messages WHERE chat_id = ? ORDER BY id ASC"
    rows = sql_exec(query, (chat_id,), fetchall=True)
    if not rows:
        let_log(f"История не найдена для чата {chat_id}")
        return []
    history = []
    missing_link_ids = []
    for r in rows:
        msg = {
            'id': r[0],
            'role': r[1],
            'full_text': r[2],
            'is_vectorized': r[3],
            'relevance_score': r[4],
            'vector_id': r[5],
            'is_compressed': r[6],
            'chat_id': r[7]
        }
        if not msg['full_text'] and msg['vector_id']:
            missing_link_ids.append(str(msg['vector_id']))
        history.append(msg)
    # Батч-резолв full_text по vector_id (меньше sql_exec / открытий)
    if missing_link_ids:
        by_vid = _batch_resolve_texts_by_vector_ids(missing_link_ids)
        for msg in history:
            if not msg['full_text'] and msg['vector_id']:
                row_text = by_vid.get(str(msg['vector_id']))
                if row_text:
                    msg['full_text'] = row_text[0]
                    if row_text[1]:
                        msg['role'] = row_text[1]
                    if row_text[2] is not None:
                        msg['is_compressed'] = row_text[2]
                    if row_text[3] is not None:
                        msg['relevance_score'] = row_text[3]
    let_log(f"Получена история для чата {chat_id}: {len(history)} сообщений")
    return history


def _batch_resolve_texts_by_vector_ids(vector_ids: list) -> dict:
    """vector_id → (full_text, role, is_compressed, relevance_score). Один/несколько IN-запросов."""
    ids = list(dict.fromkeys(str(v) for v in vector_ids if v))
    if not ids:
        return {}
    result = {}
    chunk = 400
    for i in range(0, len(ids), chunk):
        part = ids[i:i + chunk]
        placeholders = ','.join('?' * len(part))
        # Берём строки, где full_text уже заполнен (link-сообщения смотрят на «полную» копию)
        rows = sql_exec(
            f"SELECT vector_id, full_text, role, is_compressed, relevance_score FROM rag_messages "
            f"WHERE vector_id IN ({placeholders}) AND full_text IS NOT NULL AND full_text != ''",
            tuple(part),
            fetchall=True,
        ) or []
        for row in rows:
            vid = str(row[0])
            if vid not in result:
                result[vid] = (row[1], row[2], row[3], row[4])
    return result

def is_context_overflow(context_text: str) -> bool:
    estimated_tokens = len(context_text) * get_text_tokens_coefficient()
    token_limit = get_token_limit()
    overflow = estimated_tokens > (token_limit - 2000)
    let_log(f"##### Проверка переполнения контекста #####")
    let_log(f"Примерное количество токенов: {estimated_tokens}")
    let_log(f"Лимит токенов: {token_limit}")
    let_log(f"Переполнение: {overflow}\n")
    return overflow

def get_summary(chat_id: str, summary_type: str) -> tuple[int, str | None]:
    column = 'global_summary' if summary_type == 'global' else 'recent_summary'
    row = sql_exec(f"SELECT id, {column} FROM rag_messages WHERE chat_id = ? AND {column} IS NOT NULL ORDER BY id DESC LIMIT 1", (chat_id,), fetchone=True)
    if row and row[1]:
        return row[0], row[1]
    return 0, None

def set_summary(chat_id: str, summary_type: str, message_id: int, summary_text: str):
    column = 'global_summary' if summary_type == 'global' else 'recent_summary'
    sql_exec(f"UPDATE rag_messages SET {column} = NULL WHERE chat_id = ?", (chat_id,))
    sql_exec(f"UPDATE rag_messages SET {column} = ? WHERE id = ?", (summary_text, message_id))
    let_log(f"Сохранена {summary_type} сводка в сообщении ID {message_id} для чата {chat_id}")

def create_hierarchical_summary(chat_id: str, messages: list[dict], summary_type: str, token_threshold: int = 2000):
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
        if current_chunk and (current_chunk_text_len + msg_len) * get_text_tokens_coefficient() > token_threshold:
            chunks.append(current_chunk)
            current_chunk = [msg]
            current_chunk_text_len = msg_len
        else:
            current_chunk.append(msg)
            current_chunk_text_len += msg_len
    if current_chunk:
        chunks.append(current_chunk)
    let_log(f"Создано {len(chunks)} чанков для суммаризации")
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        chunk_text = ""
        for msg in chunk:
            text = msg['full_text']
            if not text and msg.get('vector_id'):
                text = _get_text_by_vector_id(msg['vector_id'])
            chunk_text += f"{msg['role']}{text}\n"
        let_log(f"##### Суммаризация чанка {i+1}/{len(chunks)} #####")
        let_log(f"Текст чанка: {chunk_text}...")
        chunk_summary = ask_model(chunk_text, system_prompt=prompt_chunk_summary + '\n' + no_markdown_instruction, purpose='summary').strip()
        if chunk_summary:
            chunk_summaries.append(chunk_summary)
            let_log(f"Сводка чанка: {chunk_summary}")
        else:
            let_log(f"Не удалось создать сводку для чанка {i+1}")
    if not chunk_summaries:
        let_log(f"##### [{chat_id}] Не удалось создать сводки чанков. #####\n")
        return
    summaries_text = "\n".join(chunk_summaries)
    if summary_type == 'global':
        final_prompt = prompt_global_summary
    else:
        final_prompt = prompt_recent_summary
    let_log("##### Создание финальной сводки #####")
    let_log(f"Сводки чанков: {summaries_text}")
    let_log(f"Финальный промпт: {final_prompt}\n- {summaries_text}")
    final_summary = ask_model(f"\n- {summaries_text}", system_prompt=final_prompt + '\n' + no_markdown_instruction, purpose='summary').strip()
    if final_summary:
        max_id = max(msg['id'] for msg in messages)
        set_summary(chat_id, summary_type, max_id, final_summary)
        let_log(f"##### [{chat_id}] Сохранена новая иерархическая '{summary_type}' сводка (до ID {max_id}): {final_summary} #####\n")
    else:
        let_log(f"##### [{chat_id}] Не удалось создать финальную сводку #####\n")

def compact_messages_llm(messages_to_compact: list[dict]) -> str:
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
    if not message or message.get('is_compressed'):
        return message
    vector_id = message.get('vector_id')
    if not vector_id:
        return message
    if not message.get('full_text'):
        row = sql_exec("SELECT id, full_text FROM rag_messages WHERE vector_id = ? AND full_text IS NOT NULL AND full_text != '' LIMIT 1", (vector_id,), fetchone=True)
        if not row:
            let_log(f"Не найдена запись с текстом для vector_id {vector_id}")
            return message
        text_id, full_text = row
    else:
        text_id = message['id']
        full_text = message['full_text']
    compacted_text = text_cutter(full_text, cut_message=True)
    if not compacted_text or compacted_text == full_text:
        sql_exec("UPDATE rag_messages SET is_compressed = TRUE WHERE vector_id = ?", (vector_id,))
        message['is_compressed'] = True
        return message
    sql_exec("UPDATE rag_messages SET full_text = ?, is_compressed = TRUE WHERE id = ?", (compacted_text, text_id))
    sql_exec("UPDATE rag_messages SET is_compressed = TRUE WHERE vector_id = ?", (vector_id,))
    message['full_text'] = compacted_text
    message['is_compressed'] = True
    return message

def _calculate_tokens(text): return int(len(text) * get_text_tokens_coefficient())

def _get_text_by_vector_id(vector_id: str) -> str:
    if not vector_id:
        return ""
    row = sql_exec("SELECT full_text FROM rag_messages WHERE vector_id = ? AND full_text IS NOT NULL AND full_text != '' LIMIT 1", (vector_id,), fetchone=True)
    return row[0] if row else ""

def _get_sum_tokens_since(chat_id, since_id):
    rows = sql_exec(
        "SELECT full_text, vector_id FROM rag_messages WHERE chat_id = ? AND id > ?",
        (chat_id, since_id), fetchall=True)
    total = 0
    for full_text, vector_id in rows:
        if full_text:
            total += _calculate_tokens(full_text)
        elif vector_id:
            text = _get_text_by_vector_id(vector_id)
            total += _calculate_tokens(text)
    return total

def _get_last_messages_by_tokens(chat_id: str, token_threshold: int, after_id: int = 0) -> list[dict]:
    rows = sql_exec("SELECT id, role, full_text FROM rag_messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,), fetchall=True)
    messages = []
    for r in rows:
        if r[0] > after_id:
            messages.append({'id': r[0], 'role': r[1], 'full_text': r[2]})
    result = []
    total_tokens = 0
    for msg in reversed(messages):
        tokens = _calculate_tokens(msg['full_text'])
        result.append(msg)
        total_tokens += tokens
        if total_tokens >= token_threshold:
            break
    result.reverse()
    return result

def _fit_documents_to_token_limit(documents: list[str], token_limit: int) -> str:
    if not documents:
        return ""
    token_coeff = get_text_tokens_coefficient()
    result_parts = []
    used_tokens = 0
    for doc in documents:
        doc_tokens = len(doc) * token_coeff
        if used_tokens + doc_tokens <= token_limit:
            result_parts.append(doc)
            used_tokens += doc_tokens
        else:
            remaining_tokens = token_limit - used_tokens
            if remaining_tokens <= 0:
                break
            lo, hi = 0, len(doc)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if mid * token_coeff <= remaining_tokens:
                    lo = mid
                else:
                    hi = mid - 1
            truncated = doc[:lo]
            result_parts.append(truncated)
            break
    return "".join(result_parts)

def rerank_rag_results(chat_id: str, initial_results: dict) -> list[str]:
    if not initial_results:
        return []
    documents = initial_results.get('documents', [])
    if isinstance(documents, list) and documents and isinstance(documents[0], list):
        documents = documents[0]
    return documents

def prompt_assembler(chat_id: str, system_prompt: str, current_message: str, history: list[dict], role: str = None) -> str:
    """
    Собирает финальный промпт. Включает current_message в историю для расчётов (как временное сообщение).
    Если use_rag == False, резерв для RAG равен 0, поиск не выполняется.
    Создаёт глобальную сводку при переполнении, после чего старые сообщения исключаются.
    """
    let_log(f"##### [{chat_id}] Сборка промпта (v8 - Unified) #####")
    let_log(f"Текущее сообщение (будет добавлено в историю): {current_message}")
    let_log(f"Роль текущего сообщения: {role}")
    let_log(f"Длина истории: {len(history)}")

    from cross_gpt import last_messages_marker, rag_context_marker, global_summary_marker, recent_summary_marker

    SAFETY_MARGIN = 2000
    # Резерв для RAG: если use_rag выключен, он равен 0
    RAG_RESERVE_PERCENTAGE = 0.15 if use_rag else 0.0

    token_limit = get_token_limit()
    last_global_id, global_summary = get_summary(chat_id, 'global')
    last_recent_id, recent_summary = get_summary(chat_id, 'recent')

    # Базовая часть промпта (системный промпт + сводки)
    base_prompt_parts = [system_prompt]
    if global_summary:
        base_prompt_parts.append(f"{global_summary_marker}{global_summary}")
    if recent_summary:
        base_prompt_parts.append(f"{recent_summary_marker}{recent_summary}")
    base_prompt_str_no_rag = "".join(base_prompt_parts)
    base_tokens_no_rag = _calculate_tokens(base_prompt_str_no_rag)

    # Доступно для истории (с учётом резерва RAG)
    available_for_history = token_limit - base_tokens_no_rag - SAFETY_MARGIN - int(token_limit * RAG_RESERVE_PERCENTAGE)

    # Создаём копию истории и добавляем временное сообщение (текущий запрос)
    mutable_history = list(history)
    # NEXT RELEASE: id=0 + filter `m['id'] > last_global_id` drops this temp when
    # last_global_id==0; get_chat_context often omits role. Do not "fix" lightly (RAG-on).
    temp_msg = {
        'id': 0,  # специальный ID, который не сохраняется
        'role': role or '',
        'full_text': current_message,
        'is_compressed': False,
        'vector_id': None
    }
    mutable_history.append(temp_msg)

    # Общее количество токенов всей истории (включая временное)
    total_history_tokens = _calculate_tokens(
        "".join([f"{(m.get('role') or '')}{(m.get('full_text') or '')}" for m in mutable_history])
    )

    history_was_truncated = total_history_tokens > available_for_history

    # Проверяем и создаём глобальную сводку (если переполнение или уже есть сводка)
    new_global_tokens = _get_sum_tokens_since(chat_id, last_global_id)
    if (history_was_truncated or last_global_id > 0) and new_global_tokens >= global_summary_tokens_percent * token_limit:
        let_log(f"##### [{chat_id}] Условие для глобальной сводки выполнено. Создаём... #####")
        messages_for_global = []
        if last_global_id > 0:
            messages_for_global.append({'id': last_global_id, 'role': 'global_summary', 'full_text': global_summary})
        rows = sql_exec("SELECT id, role, full_text FROM rag_messages WHERE chat_id = ? AND id > ? ORDER BY id ASC", (chat_id, last_global_id), fetchall=True)
        for r in rows:
            messages_for_global.append({'id': r[0], 'role': r[1], 'full_text': r[2]})
        # Также включаем временное сообщение, если оно не пустое
        if current_message.strip():
            messages_for_global.append(temp_msg)
        create_hierarchical_summary(chat_id, messages_for_global, 'global')
        last_global_id, global_summary = get_summary(chat_id, 'global')
        last_recent_id, recent_summary = get_summary(chat_id, 'recent')
        # Пересобираем базовую часть с новыми сводками
        base_prompt_parts = [system_prompt]
        if global_summary:
            base_prompt_parts.append(f"{global_summary_marker}{global_summary}")
        if recent_summary:
            base_prompt_parts.append(f"{recent_summary_marker}{recent_summary}")
        base_prompt_str_no_rag = "".join(base_prompt_parts)
        base_tokens_no_rag = _calculate_tokens(base_prompt_str_no_rag)
        available_for_history = token_limit - base_tokens_no_rag - SAFETY_MARGIN - int(token_limit * RAG_RESERVE_PERCENTAGE)

    # Проверяем и создаём недавнюю сводку (только если use_rag включён, так как в стандартном режиме она не нужна)
    if use_rag:
        new_recent_tokens = _get_sum_tokens_since(chat_id, last_recent_id)
        if (history_was_truncated or last_recent_id > 0) and new_recent_tokens >= recent_summary_tokens_percent * token_limit:
            let_log(f"##### [{chat_id}] Условие для недавней сводки выполнено. Создаём... #####")
            messages_for_recent = []
            rows = sql_exec("SELECT id, role, full_text FROM rag_messages WHERE chat_id = ? AND id > ? ORDER BY id ASC", (chat_id, last_recent_id), fetchall=True)
            for r in rows:
                messages_for_recent.append({'id': r[0], 'role': r[1], 'full_text': r[2]})
            if current_message.strip():
                messages_for_recent.append(temp_msg)
            create_hierarchical_summary(chat_id, messages_for_recent, 'recent')
            last_global_id, global_summary = get_summary(chat_id, 'global')
            last_recent_id, recent_summary = get_summary(chat_id, 'recent')
            base_prompt_parts = [system_prompt]
            if global_summary:
                base_prompt_parts.append(f"{global_summary_marker}{global_summary}")
            if recent_summary:
                base_prompt_parts.append(f"{recent_summary_marker}{recent_summary}")
            base_prompt_str_no_rag = "".join(base_prompt_parts)
            base_tokens_no_rag = _calculate_tokens(base_prompt_str_no_rag)
            available_for_history = token_limit - base_tokens_no_rag - SAFETY_MARGIN - int(token_limit * RAG_RESERVE_PERCENTAGE)

    # Теперь собираем историю, исключая сообщения до last_global_id (если есть глобальная сводка)
    # Также исключаем сообщения, которые уже включены в сводку (id <= last_global_id)
    history_to_use = [m for m in mutable_history if m['id'] > last_global_id]

    # Усекаем историю по токенам
    history_strings_list = []
    history_token_count = 0
    history_included_vector_ids = []

    if available_for_history <= 0:
        let_log(f"##### [{chat_id}] ВНИМАНИЕ: Нет места для истории. Пропускаем. #####")
    else:
        for msg in reversed(history_to_use):
            # Skip unresolved stubs (role/full_text None) — avoids "NoneNone" in prompt
            role_s = msg.get('role') or ''
            text_s = msg.get('full_text') or ''
            if not role_s and not text_s:
                let_log(f"Пропуск пустого/нерезолвленного сообщения id={msg.get('id')} vector_id={msg.get('vector_id')!r}")
                continue
            msg_str = f"{role_s}{text_s}"
            msg_tokens = _calculate_tokens(msg_str)
            if (history_token_count + msg_tokens) <= available_for_history:
                history_strings_list.append(msg_str)
                history_token_count += msg_tokens
                if msg.get('vector_id'):
                    history_included_vector_ids.append(msg['vector_id'])
            else:
                let_log(f"Сообщение ID {msg['id']} (токены: {msg_tokens:.0f}) не помещается.")
                if ENABLE_ON_THE_FLY_COMPRESSION and not msg.get('is_compressed'):
                    let_log(f"Запуск сжатия 'на лету' для ID {msg['id']}...")
                    compressed_msg = _compress_message_in_db(msg)
                    msg_str_compressed = f"{compressed_msg.get('role') or ''}{compressed_msg.get('full_text') or ''}"
                    msg_tokens_compressed = _calculate_tokens(msg_str_compressed)
                    if (history_token_count + msg_tokens_compressed) <= available_for_history:
                        let_log(f"Сжатое сообщение ID {msg['id']} (токены: {msg_tokens_compressed:.0f}) теперь помещается.")
                        history_strings_list.append(msg_str_compressed)
                        history_token_count += msg_tokens_compressed
                        if msg.get('vector_id'):
                            history_included_vector_ids.append(msg['vector_id'])
                    else:
                        let_log(f"Даже сжатое сообщение ID {msg['id']} не помещается. Усечение.")
                        break
                else:
                    let_log("Достигнут лимит токенов. Усечение.")
                    break
        history_strings_list.reverse()

    # Обеспечиваем нечётное количество сообщений (удаляем самое старое, если чётное)
    while len(history_strings_list) > 1 and len(history_strings_list) % 2 == 0:
        removed = history_strings_list.pop(0)
        let_log(f"Удалено самое старое сообщение для сохранения нечетности: {removed[:100]}...")

    history_final_str = "".join(history_strings_list)
    global_state.current_agent_history_for_filesystem = history_final_str
    # При переполнении — векторизуем выпавшие из окна сообщения (shared / internal при overflow)
    if use_rag and history_was_truncated:
        try:
            from chat_manager import vectorize_dropped_messages
            included_ids = set(history_included_vector_ids)
            dropped = [
                m for m in history_to_use
                if m.get('vector_id') and m.get('vector_id') not in included_ids and m.get('id', 0) != 0
            ]
            vectorize_dropped_messages(chat_id, dropped, overflow_happened=True)
        except Exception as e:
            let_log(f"[RAG] vectorize_dropped_messages: {e}")
    # RAG-часть (только если use_rag включён и история была усечена)
    rag_prompt_part = ""
    if use_rag and history_was_truncated:
        available_tokens_for_rag_actual = token_limit - base_tokens_no_rag - history_token_count - SAFETY_MARGIN
        if available_tokens_for_rag_actual > 0:
            let_log(f"##### [{chat_id}] RAG АКТИВИРОВАН. Доступно токенов: {available_tokens_for_rag_actual:.0f} #####")
            rag_token_limit_final = available_tokens_for_rag_actual
            # Query for RAG: newest-first (current, then history[-1], [-2], …).
            # get_embs keeps a prefix / halves on overflow → prefers recent text.
            # TODO: optional — emb each message separately and intersect nearest hits.
            _parts = []
            _cur = (current_message or "").strip()
            if _cur:
                _parts.append(_cur)
            for _m in reversed(history_to_use or []):
                _t = str((_m.get("full_text") if isinstance(_m, dict) else "") or "").strip()
                if _t:
                    _parts.append(_t)
            expanded_query = "\n".join(_parts)
            try:
                query_embedding = get_embs(expanded_query) if expanded_query else []
            except Exception as e:
                let_log(f"RAG: get_embs failed ({e}) — поиск пропущен, диалог продолжается")
                query_embedding = []
            if not query_embedding:
                let_log("RAG: пустой query embedding — поиск пропущен")
                initial_results = None
            else:
                rag_filters = {
                    '$or': [
                        {'chat_id_1': chat_id},
                        {'chat_id_2': chat_id}
                    ],
                    '$nin': {'vector_id': history_included_vector_ids}
                }
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

            retrieved_texts = []
            if isinstance(initial_results, dict) and initial_results.get('ids') and initial_results['ids']:
                ids_list = initial_results['ids'][0] if isinstance(initial_results['ids'][0], list) else initial_results['ids']
                if ids_list:
                    placeholders = ','.join('?' * len(ids_list))
                    query = f"SELECT vector_id, full_text FROM rag_messages WHERE chat_id = ? AND vector_id IN ({placeholders})"
                    rows = sql_exec(query, (chat_id,) + tuple(ids_list), fetchall=True)
                    if rows:
                        texts_map = {row[0]: row[1] for row in rows}
                        retrieved_texts = [texts_map.get(vid, "") for vid in ids_list]
                        let_log(f"Получено {len(retrieved_texts)} текстов из SQLite")
                    else:
                        let_log("Не найдены тексты для полученных vector_id")
            else:
                let_log("RAG поиск не дал результатов или вернул некорректный формат")

            if retrieved_texts:
                results_with_texts = {'documents': [retrieved_texts]}
                top_docs = rerank_rag_results(chat_id, results_with_texts)
                if top_docs:
                    retrieved_context = _fit_documents_to_token_limit(top_docs, rag_token_limit_final)
                    rag_prompt_part = rag_context_marker + f"{retrieved_context}\n"
                    let_log(f"Добавлен RAG контекст: {len(retrieved_context)} символов (токенов ~{rag_token_limit_final})")
                else:
                    let_log("Не удалось получить переранжированные документы")
            else:
                let_log("RAG поиск не дал результатов или тексты не найдены")
    else:
        if use_rag:
            let_log(f"##### [{chat_id}] RAG НЕ АКТИВИРОВАН. Условие: (Усечение: {history_was_truncated}) #####")
        else:
            let_log(f"##### [{chat_id}] RAG отключён (use_rag=False) #####")

    # Финальный промпт
    final_prompt_parts = [base_prompt_str_no_rag, rag_prompt_part, last_messages_marker, history_final_str]
    final_prompt = "".join(final_prompt_parts)

    let_log(f"##### Финальный промпт собран #####")
    let_log(f"Общая длина промпта: {len(final_prompt)} символов")
    return final_prompt

def rag_constructor(chat_id: str, system_prompt: str, current_message: str, role: str = None) -> str:
    """
    Главная функция-оркестратор RAG.
    Получает историю сообщений из БД и собирает финальный промпт.
    """
    let_log("##### RAG CONSTRUCTOR ЗАПУЩЕН #####")
    let_log(f"Chat ID: {chat_id}")
    let_log(f"Длина системного промпта: {len(system_prompt)}")
    let_log(f"Текущее сообщение: {current_message}")
    let_log(f"Роль текущего сообщения: {role}")
    history = get_history(chat_id)
    final_prompt = prompt_assembler(chat_id=chat_id, system_prompt=system_prompt, current_message=current_message, history=history, role=role)
    let_log("##### RAG CONSTRUCTOR ЗАВЕРШЕН #####\n")
    return final_prompt