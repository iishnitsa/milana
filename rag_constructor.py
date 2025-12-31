# rag_constructor.py

# Импорт необходимых функций из вашего основного модуля и стандартных библиотек
from cross_gpt import sql_exec, coll_exec, ask_model, get_embs, get_token_limit, get_text_tokens_coefficient, let_log, text_cutter
import uuid
import re

# =LET-BL-VER-1.0: Глобальный переключатель для опционального сжатия
# True - сжимать сообщения "на лету" при переполнении (согласно TODO)
# False - не сжимать, просто отбрасывать старые
ENABLE_ON_THE_FLY_COMPRESSION = False

# ==============================================================================
# TODO: Задачи из ТЗ, оставленные как комментарии для будущего развития
# ==============================================================================
# TODO: После активации RAG, предусмотреть механизм,
# позволяющий принудительно использовать полные/неотфильтрованные
# версии сообщений, если окажется, что они были сокращены/отмечены
# как мусор по ошибке. Это может быть полезно для исправления
# контекста "на лету".
#
# TODO: Создать обновляемую (корректируемую) систему утверждений, для каждого
# диалога отдельная база знаний (наверное), типа Москва - столица России.
# ==============================================================================

# --- 0. Инициализация БД ---
def initialize_rag_database():
    """Создает все необходимые таблицы в SQLite, если они не существуют."""
    let_log("##### Инициализация таблиц базы данных RAG... #####")
    sql_exec('''
        CREATE TABLE IF NOT EXISTS rag_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            role TEXT NOT NULL,
            full_text TEXT NOT NULL,
            is_vectorized BOOLEAN DEFAULT FALSE,
            vector_id TEXT UNIQUE,
            relevance_score INTEGER DEFAULT 0,  -- Новое поле для системы баллов
            is_compressed BOOLEAN DEFAULT FALSE -- LET-BL-VER-1.0: Флаг для сжатых сообщений (TODO)
        );
    ''')
    # LET-BL-VER-1.0: Таблица chat_states удалена, т.к. состояние чата больше не используется
    sql_exec('''
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            summary_type TEXT NOT NULL,
            summary_text TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    let_log("##### Таблицы базы данных RAG готовы. #####\n")

# --- 1. Функции для работы с БД ---
# LET-BL-VER-1.0: Функции get_chat_status и set_chat_status удалены.

def get_latest_summary(chat_id: str, summary_type: str) -> str | None:
    summary = sql_exec("SELECT summary_text FROM summaries WHERE chat_id = ? AND summary_type = ? ORDER BY timestamp DESC LIMIT 1", (chat_id, summary_type), fetchone=True)
    if summary:
        let_log(f"Найдена {summary_type} сводка для чата {chat_id}: {summary}")
    else:
        let_log(f"Сводка {summary_type} не найдена для чата {chat_id}")
    return summary

def get_history(chat_id: str, only_unvectorized: bool = False) -> list[dict]:
    """
    Получает историю. Флаг 'only_unvectorized' теперь в основном бесполезен,
    но оставлен для обратной совместимости, если где-то используется.
    LET-BL-VER-1.0: Добавлены vector_id, is_compressed, chat_id.
    """
    query = "SELECT id, role, full_text, is_vectorized, relevance_score, vector_id, is_compressed, chat_id FROM rag_messages WHERE chat_id = ?"
    if only_unvectorized:
        query += " AND is_vectorized = FALSE"
    query += " ORDER BY timestamp ASC"
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

# --- 2. Обновленный Иерархический Суммаризатор ---
def create_hierarchical_summary(chat_id: str, messages: list[dict], summary_type: str, token_threshold: int = 2000):
    """
    Создает иерархическую сводку, основываясь на объеме текста в чанках.
    token_threshold - примерный лимит токенов на один чанк для суммаризации.
    """
    let_log(f"##### [{chat_id}] Создание иерархической '{summary_type}' сводки на основе токенов... #####")
    let_log(f"Обработка {len(messages)} сообщений с порогом токенов: {token_threshold}")
    
    if not messages: 
        let_log("Нет сообщений для суммаризации")
        return

    # 1. Динамическое разделение на чанки по объему текста
    chunks = []
    current_chunk = []
    current_chunk_text_len = 0

    for msg in messages:
        msg_len = len(msg['full_text'])
        # Проверяем, не превысит ли добавление нового сообщения порог
        if current_chunk and (current_chunk_text_len + msg_len) * get_text_tokens_coefficient() > token_threshold:
            # Если превысит, сохраняем текущий чанк и начинаем новый
            chunks.append(current_chunk)
            current_chunk = [msg]
            current_chunk_text_len = msg_len
        else:
            # Иначе, добавляем в текущий чанк
            current_chunk.append(msg)
            current_chunk_text_len += msg_len
    
    # Не забываем добавить последний чанк, если он остался
    if current_chunk:
        chunks.append(current_chunk)

    let_log(f"Создано {len(chunks)} чанков для суммаризации")
    
    chunk_summaries = []
    # 2. Суммаризация каждого чанка
    for i, chunk in enumerate(chunks):
        chunk_text = "\n".join([f"{msg['role']}: {msg['full_text']}" for msg in chunk])
        prompt = f"Напиши очень краткую (1-2 предложения) суть фрагмента диалога, присланного пользователем"
        
        let_log(f"##### Суммаризация чанка {i+1}/{len(chunks)} #####")
        let_log(f"Текст чанка: {chunk_text}...")  # Логируем первые 200 символов
        
        chunk_summary = ask_model(chunk_text, system_prompt=prompt).strip()
        if chunk_summary:
            chunk_summaries.append(chunk_summary)
            let_log(f"Сводка чанка: {chunk_summary}")
        else:
            let_log(f"Не удалось создать сводку для чанка {i+1}")

    if not chunk_summaries:
        let_log(f"##### [{chat_id}] Не удалось создать сводки чанков. #####\n")
        return

    # 3. Финальная сводка из сводок чанков
    summaries_text = "\n- ".join(chunk_summaries)
    # Сводка
    final_prompt = f"На основе этих ключевых моментов, напиши общую {summary_type} сводку (2-4 предложения)"
    
    let_log("##### Создание финальной сводки из чанков #####")
    let_log(f"Сводки чанков: {summaries_text}")
    let_log(f"Финальный промпт: {final_prompt}\n- {summaries_text}")
    
    final_summary = ask_model(f"\n- {summaries_text}", system_prompt=final_prompt).strip()

    if final_summary:
        sql_exec("INSERT INTO summaries (chat_id, summary_type, summary_text) VALUES (?, ?, ?)", (chat_id, summary_type, final_summary))
        let_log(f"##### [{chat_id}] Сохранена новая иерархическая '{summary_type}' сводка: {final_summary} #####\n")
    else:
        let_log(f"##### [{chat_id}] Не удалось создать финальную сводку #####\n")


# --- 3. LET-BL-VER-1.0: Модуль сжатия (оставлен, т.к. нужен для новой логики) ---
def compact_messages_llm(messages_to_compact: list[dict]) -> str:
    """
    Принимает группу сообщений (или одно сообщение) и просит LLM сжать их в единую суть.
    """
    let_log(f"##### Сжатие {len(messages_to_compact)} сообщений #####")
    let_log(f"ID сообщений: {[m['id'] for m in messages_to_compact]}")
    let_log(f"##### ПРИШЕДШИЙ СПИСОК #####")
    let_log(f"{messages_to_compact}")
    
    text_to_compact = "\n".join([f"{msg['role']}: {msg['full_text']}" for msg in messages_to_compact])
    compacted_text = text_cutter(text_to_compact)
    '''
    prompt = (
        "Проанализируй следующий фрагмент диалога. Твоя задача - создать из него один сжатый абзац.\n"
        "Требования:\n"
        "1. Убери всю 'воду', незначительные реплики и очевидные подтверждения (вроде 'ок', 'понял').\n"
        "2. Объедини связанные мысли, вопросы и ответы в единое целое.\n"
        "3. Сохрани всю критически важную информацию, технические детали и принятые решения.\n"
        "Результат должен быть квинтэссенцией этого фрагмента."
    )
    
    let_log(f"Промпт для сжатия: {prompt}\nФрагмент для сжатия:\n{text_to_compact}...")
    
    try: compacted_text = ask_model(f"Фрагмент для сжатия:{text_to_compact}", system_prompt=prompt).strip()
    except: pass # TODO:
    '''
    let_log(f"Сжатый текст: {compacted_text}")
    let_log("##### Сжатие завершено #####\n")
    
    return compacted_text

# LET-BL-VER-1.0: Функция shorten_and_vectorize_buffer удалена.
# Она создавала 'summary_chunk', что противоречит вашим TODO.

# LET-BL-VER-1.0: Новая вспомогательная функция для сжатия "на лету"
def _compress_message_in_db(message: dict) -> dict | None:
    """
    Сжимает ОДНО сообщение, обновляет его в БД и RAG, и возвращает обновленное сообщение.
    Вызывается из prompt_assembler при переполнении.
    """
    if not message or message.get('is_compressed'):
        return message # Уже сжато или нет сообщения
    
    msg_id = message['id']
    vector_id = message.get('vector_id')
    chat_id = message.get('chat_id') # Получаем из get_history
    
    let_log(f"##### Сжатие 'на лету' сообщения ID: {msg_id} #####")
    
    # Используем compact_messages_llm для сжатия одного сообщения
    compacted_text = compact_messages_llm([message]) # Оборачиваем в список
    
    if not compacted_text or compacted_text == message['full_text']:
        let_log(f"Сжатие не удалось или текст не изменился. Помечаем как 'is_compressed'.")
        # Помечаем, чтобы не пытаться сжать снова
        sql_exec("UPDATE rag_messages SET is_compressed = TRUE WHERE id = ?", (msg_id,))
        message['is_compressed'] = True
        return message

    # 1. Обновляем в SQL
    sql_exec(
        "UPDATE rag_messages SET full_text = ?, is_compressed = TRUE WHERE id = ?",
        (compacted_text, msg_id)
    )
    
    # 2. Обновляем в RAG (ChromaDB), если есть vector_id
    if vector_id:
        try:
            new_embedding = get_embs(compacted_text)
            coll_exec(
                action="update", coll_name="rag_collection",
                ids=[vector_id],
                documents=[compacted_text],
                embeddings=[new_embedding],
                metadatas=[{'chat_id': chat_id, 'role': message.get('role'), 'relevance_score': message.get('relevance_score')}]
            )
            let_log(f"Обновлен RAG-документ {vector_id} сжатым текстом.")
        except Exception as e:
            let_log(f"##### ОШИБКА: Не удалось обновить RAG для сжатого сообщения {vector_id}: {e} #####")

    # Возвращаем обновленный объект сообщения
    message['full_text'] = compacted_text
    message['is_compressed'] = True
    return message


# --- 4. Новая система оценки релевантности ---
def update_message_scores(chat_id: str):
    """
    Ретроспективно обновляет баллы релевантности сообщений на основе фидбэка пользователя.
    """
    let_log(f"##### [{chat_id}] Обновление баллов сообщений #####")
    
    # Получаем 3-4 последних сообщения для анализа последовательности
    last_messages = sql_exec(
        "SELECT id, role, full_text FROM rag_messages WHERE chat_id = ? ORDER BY timestamp DESC LIMIT 4", 
        (chat_id,), 
        fetchall=True
    )

    if len(last_messages) < 2:
        let_log("Недостаточно контекста для анализа")
        return  # Недостаточно контекста для анализа

    latest_message = last_messages[0]  # Только что пришло от пользователя
    previous_message = last_messages[1]  # Скорее всего, ответ ассистента

    let_log(f"Последнее сообщение: {latest_message[1]}: {latest_message[2]}")
    let_log(f"Предыдущее сообщение: {previous_message[1]}: {previous_message[2]}")

    # --- Правило №1: Пользователь подтверждает, что решение сработало ---
    positive_keywords = ['спасибо', 'сработало', 'помогло', 'отлично', 'то что нужно', 'работает', 'решил'] # TODO:
    # Обращаемся к элементам кортежа по индексам:
    # [0] - id, [1] - role, [2] - full_text
    if latest_message[1] == 'user' and any(kw in latest_message[2].lower() for kw in positive_keywords):
        # Если предыдущее сообщение от ассистента, повышаем его ценность
        if previous_message[1] == 'assistant':
            let_log(f"##### [{chat_id}] Обнаружена положительная обратная связь. Повышение рейтинга сообщения ID {previous_message[0]} #####")
            sql_exec("UPDATE rag_messages SET relevance_score = relevance_score + 5 WHERE id = ?", (previous_message[0],))

    # --- Правило №2: Пользователь сообщает об ошибке ---
    negative_keywords = ['ошибка', 'не работает', 'сломалось', 'не помогло', 'error', 'не получается', 'неправильно'] # TODO:
    if latest_message[1] == 'user' and any(kw in latest_message[2].lower() for kw in negative_keywords):
        # Если ассистент перед этим дал код или инструкцию, понижаем ценность этого сообщения
        if previous_message[1] == 'assistant':
            let_log(f"##### [{chat_id}] Обнаружена отрицательная обратная связь. Понижение рейтинга сообщения ID {previous_message[0]} #####")
            sql_exec("UPDATE rag_messages SET relevance_score = relevance_score - 5 WHERE id = ?", (previous_message[0],))
    
    let_log("##### Обновление баллов завершено #####\n")

def rerank_rag_results(chat_id: str, initial_results: dict) -> list[str]:
    """
    ЗАГЛУШКА: Простая версия, возвращает первые 3 документа
    """
    if not initial_results:
        return []
    
    # Пытаемся извлечь документы
    documents = initial_results.get('documents', [])
    
    # Если документы вложены в список (стандартный формат ChromaDB)
    if isinstance(documents, list) and documents and isinstance(documents[0], list):
        documents = documents[0]
    
    # Возвращаем первые 3 или все, если меньше
    return documents[:3]

def prompt_assembler(chat_id: str, system_prompt: str, current_message: str, history: list[dict]) -> str:
    """
    Собирает финальный промпт. НЕ включает current_message в промпт и расчет токенов.
    Реализует RAG (при усечении), сжатие "на лету" и сохранение нечетного количества сообщений.
    """
    
    let_log(f"##### [{chat_id}] Сборка промпта (v6 - Fix coll_exec) #####")
    let_log(f"Текущее сообщение (не включается): {current_message}")
    let_log(f"Длина истории: {len(history)}")
    from cross_gpt import last_messages_marker, rag_context_marker
    # --- КОНСТАНТЫ И РЕЗЕРВЫ ---
    SAFETY_MARGIN = 2000
    RAG_RESERVE_PERCENTAGE = 0.15
    
    # --- 1. Сборка "Базового" промпта (без RAG и истории) ---
    base_prompt_parts = [system_prompt]
    
    global_summary = get_latest_summary(chat_id, 'global')
    if global_summary: 
        base_prompt_parts.append(f"\nГлобальная сводка диалога:\n{global_summary}")
    
    recent_summary = get_latest_summary(chat_id, 'recent_topic')
    if recent_summary:
        base_prompt_parts.append(f"\nСводка последней темы:\n{recent_summary}")
    
    # --- 2. Расчет лимитов токенов (С ДИНАМИЧЕСКИМ РЕЗЕРВОМ RAG) ---
    base_prompt_str_no_rag = "".join(base_prompt_parts)
    base_tokens_no_rag = len(base_prompt_str_no_rag) * get_text_tokens_coefficient()
    token_limit = get_token_limit()
    
    # Динамический резерв RAG: 15% от общего лимита
    RAG_RESERVE_TOKENS = int(token_limit * RAG_RESERVE_PERCENTAGE)
    
    # Доступно для истории: Общий лимит - База - Safety Margin - Резерв RAG
    available_tokens_for_history = token_limit - base_tokens_no_rag - SAFETY_MARGIN - RAG_RESERVE_TOKENS

    let_log(f"##### [{chat_id}] Расчет лимитов для истории (резерв RAG: {RAG_RESERVE_TOKENS:.0f}) #####")
    let_log(f"Лимит токенов (общий): {token_limit}")
    let_log(f"Токены (База): {base_tokens_no_rag:.0f}")
    let_log(f"Токены (Safety Margin): {SAFETY_MARGIN}")
    let_log(f"Доступно для истории: {available_tokens_for_history:.0f}\n")

    # --- 3. Сборка истории (С ВОССТАНОВЛЕННОЙ ПРОВЕРКОЙ ПЕРЕПОЛНЕНИЯ) ---
    history_strings_list = []
    history_token_count = 0
    history_included_vector_ids = []
    
    mutable_history = list(history) 
    original_history_length = len(history) 
    
    if available_tokens_for_history <= 0:
        let_log("##### [{chat_id}] ВНИМАНИЕ: Нет места для истории. Пропускаем. #####")
    
    elif mutable_history:
        let_log(f"######################## ИСТОРИЯ (v6, {len(mutable_history)} сообщ.) ###################")

        for msg in reversed(mutable_history):
            msg_role = msg.get('role', 'user')
            msg_content = msg.get('full_text', '')
            
            # КОРРЕКЦИЯ ФОРМАТИРОВАНИЯ: Только роль и контент
            msg_str = f"{msg_role}{msg_content}" 
            msg_tokens = len(msg_str) * get_text_tokens_coefficient()

            # ПРОВЕРКА ПЕРЕПОЛНЕНИЯ: 
            if (history_token_count + msg_tokens) <= available_tokens_for_history:
                history_strings_list.append(msg_str)
                history_token_count += msg_tokens
                
                if msg.get('vector_id'):
                    history_included_vector_ids.append(msg['vector_id'])
            else:
                let_log(f"Сообщение ID {msg['id']} (токены: {msg_tokens:.0f}) не помещается.")
                
                if ENABLE_ON_THE_FLY_COMPRESSION and not msg.get('is_compressed'):
                    let_log(f"Запуск сжатия 'на лету' для ID {msg['id']}...")
                    compressed_msg = _compress_message_in_db(msg) 
                    
                    msg_content_compressed = compressed_msg.get('full_text', '')
                    msg_str_compressed = f"{msg_role}{msg_content_compressed}" 
                    msg_tokens_compressed = len(msg_str_compressed) * get_text_tokens_coefficient()
                    
                    if (history_token_count + msg_tokens_compressed) <= available_tokens_for_history:
                        let_log(f"Сжатое сообщение ID {msg['id']} (токены: {msg_tokens_compressed:.0f}) теперь помещается.")
                        history_strings_list.append(msg_str_compressed)
                        history_token_count += msg_tokens_compressed
                        if msg.get('vector_id'):
                            history_included_vector_ids.append(msg['vector_id'])
                    else:
                        let_log(f"Даже сжатое сообщение ID {msg['id']} (токены: {msg_tokens_compressed:.0f}) не помещается. Усечение.")
                        break 
                else:
                    let_log("Достигнут лимит токенов. Усечение.")
                    break 
        
        history_strings_list.reverse()

    # --- 3.5. RAG (долгосрочная память) - УСЛОВНАЯ АКТИВАЦИЯ И ИСПРАВЛЕНИЕ COLL_EXEC ---
    
    rag_prompt_part = ""
    history_was_truncated = len(history_strings_list) < original_history_length 
    available_tokens_for_rag_actual = token_limit - base_tokens_no_rag - history_token_count - SAFETY_MARGIN
    
    if history_was_truncated and available_tokens_for_rag_actual > 0:
        
        let_log(f"##### [{chat_id}] RAG АКТИВИРОВАН. Доступно токенов: {available_tokens_for_rag_actual:.0f} #####")

        rag_token_limit_final = available_tokens_for_rag_actual 
        
        recent_context = ""
        if len(history) >= 2:
            recent_context = "".join([f"{msg.get('role')}{msg.get('full_text')}" for msg in history[-2:]])
        expanded_query = f"{recent_context}\nuser: {current_message}"
        
        query_embedding = get_embs(expanded_query)
        
        rag_filters = {
            'chat_id': chat_id,
            '$nin': {'vector_id': history_included_vector_ids} 
        }
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Блок вызова coll_exec из предыдущей стабильной версии
        try:
            initial_results = coll_exec(
                action="query",
                coll_name="rag_collection",
                query_embeddings=[query_embedding],
                n_results=10,
                filters=rag_filters,
                fetch=["documents", "metadatas", "ids", "distances"]
            )
        except Exception as e:
            let_log(f"Ошибка coll_exec RAG: {e}")
            initial_results = None
        
        # Проверка результатов: сначала проверка типа, затем проверка наличия данных
        if isinstance(initial_results, dict) and (
            initial_results.get('documents') and 
            initial_results['documents'] and 
            initial_results['documents'][0]
        ):
            top_docs = rerank_rag_results(chat_id, initial_results)
            
            if top_docs:
                retrieved_context_full = "".join(top_docs)
                
                rag_token_limit_chars = (rag_token_limit_final / get_text_tokens_coefficient()) 
                max_chars_rag = int(rag_token_limit_chars * 3.5) 
                retrieved_context = retrieved_context_full[:max_chars_rag]
                
                rag_prompt_part = rag_context_marker + f"{retrieved_context}\n"
                
                let_log(f"Добавлен RAG контекст: {len(retrieved_context)} символов")
            else:
                let_log("Не удалось получить переранжированные документы")
        else:
            let_log("RAG поиск не дал результатов или вернул некорректный формат (возможно, ошибку API)")
    
    else:
        let_log(f"##### [{chat_id}] RAG НЕ АКТИВИРОВАН. Условие: (Усечение: {history_was_truncated}, Доступно места: {available_tokens_for_rag_actual > 0}) #####")

    # --- 4. Финальная проверка на нечетность ---
    while len(history_strings_list) > 1 and len(history_strings_list) % 2 == 0:
        removed_msg = history_strings_list.pop(0) # Удаляем самое старое
        let_log(f"Удалено самое старое сообщение для сохранения нечетности: {removed_msg[:100]}...")

    # --- 5. Финальная сборка промпта ---
    
    history_final_str = "".join(history_strings_list)
    
    final_prompt_parts = [
        base_prompt_str_no_rag,  # system + summaries
        rag_prompt_part,         # RAG контекст
        last_messages_marker,
        history_final_str        # динамически собранная история
    ]
    
    final_prompt = "".join(final_prompt_parts)
    
    let_log(f"##### Финальный промпт собран #####")
    let_log(f"Общая длина промпта: {len(final_prompt)} символов")
    
    return final_prompt

# --- 6. Вспомогательные функции ---
# (пусто)

# --- 7. Главная функция-оркестратор (`rag_constructor`) ---

# *** ИЗМЕНЕНО: rag_constructor УПРОЩЕН (LET-BL-VER-1.0) ***
def rag_constructor(chat_id: str, system_prompt: str, current_message: str) -> str:
    """
    Главная функция-оркестратор. RAG всегда включен.
    """
    let_log("##### RAG CONSTRUCTOR ЗАПУЩЕН (v2) #####")
    let_log(f"Chat ID: {chat_id}")
    let_log(f"Длина системного промпта: {len(system_prompt)}")
    let_log(f"Текущее сообщение: {current_message}")

    # (chat_manager.py уже сохранил сообщение и векторизовал его)
    
    # Обновляем баллы релевантности (эта функция не зависит от chat_status)
    update_message_scores(chat_id)
    
    # --- 🟦 БЛОК ТРИГГЕРОВ СУММАРИЗАЦИИ 🟦 ---
    # (Логика без изменений, т.к. она использует 'summaries' и не создает 'summary_chunk')
        
    message_count_row = sql_exec(
        "SELECT COUNT(id) FROM rag_messages WHERE chat_id = ?",
        (chat_id,),
        fetchone=True
    )
    
    total_messages = 0
    if isinstance(message_count_row, int):
        total_messages = message_count_row
    elif isinstance(message_count_row, (tuple, list)) and len(message_count_row) > 0:
        total_messages = int(message_count_row[0])

    let_log(f"##### [{chat_id}] Проверка триггеров суммаризации. Всего сообщений: {total_messages} #####")

    GLOBAL_SUMMARY_TRIGGER = 100 # хз как но сообщений нечётное число каждый раз после запроса оказывается
    RECENT_SUMMARY_TRIGGER = 50

    if total_messages > 0 and total_messages % GLOBAL_SUMMARY_TRIGGER == 0:
        let_log(f"##### [{chat_id}] Сработал триггер ГЛОБАЛЬНОЙ сводки ({total_messages} сообщений) #####")
        all_history = get_history(chat_id) 
        create_hierarchical_summary(chat_id, all_history, 'global')

    elif total_messages > 0 and total_messages % RECENT_SUMMARY_TRIGGER == 0:
        let_log(f"##### [{chat_id}] Сработал триггер СВОДКИ ПОСЛЕДНЕЙ ТЕМЫ ({total_messages} сообщений) #####")
        
        recent_rows = sql_exec(
            "SELECT id, role, full_text, is_vectorized, relevance_score, vector_id, is_compressed, chat_id FROM rag_messages WHERE chat_id = ? ORDER BY timestamp DESC LIMIT ?",
            (chat_id, RECENT_SUMMARY_TRIGGER),
            fetchall=True
        )
        
        if recent_rows:
            recent_rows.reverse()
            recent_history = [
                {'id': r[0], 'role': r[1], 'full_text': r[2], 'is_vectorized': r[3], 'relevance_score': r[4], 'vector_id': r[5], 'is_compressed': r[6], 'chat_id': r[7]} 
                for r in recent_rows
            ]
            create_hierarchical_summary(chat_id, recent_history, 'recent_topic')
        else:
             let_log(f"##### [{chat_id}] Не удалось получить историю для 'recent_topic' сводки. #####")

    # --- 🟦 КОНЕЦ БЛОКА СУММАРИЗАЦИИ 🟦 ---
    
    # --- Предварительное сжатие ОЧЕНЬ длинных сообщений ---
    # (Эта логика оставлена, т.к. она полезна и сжимает *текущее* сообщение до того,
    # как оно попадет в prompt_assembler)
    last_msg_data = sql_exec(
        "SELECT id, full_text FROM rag_messages WHERE chat_id = ? ORDER BY timestamp DESC LIMIT 1", 
        (chat_id,), 
        fetchone=True
    )
    
    last_msg_id = None
    if last_msg_data:
        last_msg_id = last_msg_data[0]
    else:
        let_log(f"##### [{chat_id}] КРИТИЧЕСКАЯ ОШИБКА: не найдено последнее сообщение для обработки. #####")
    
    # LET-BL-VER-1.0: Ваши TODO (730-738) реализованы в prompt_assembler и chat_manager
    
    if last_msg_id and len(current_message) * get_text_tokens_coefficient() > (get_token_limit() * 0.25):
        # TODO: (Оставлено) вот тут надо не гет токен лимит а сравнение с остатками...
        let_log(f"##### [{chat_id}] Текущее сообщение слишком длинное, предварительное сжатие... #####")
        
        # Используем compact_messages_llm, который мы оставили/вернули
        single_message_buffer = [{'id': last_msg_id, 'role': 'user', 'full_text': current_message}]
        compacted_text = compact_messages_llm(single_message_buffer)
        
        if compacted_text:
            current_message = compacted_text # 1. Обновляем переменную (для RAG-запроса)
            let_log(f"##### [{chat_id}] Сообщение сжато: {current_message} #####")
            
            # 2. Обновляем сообщение в БД (оно уже было сохранено в chat_manager)
            sql_exec(
                "UPDATE rag_messages SET full_text = ?, is_compressed = TRUE WHERE id = ?", # Помечаем как сжатое
                (current_message, last_msg_id)
            )
            # 3. TODO: Здесь также нужно обновить RAG (ChromaDB), если он уже был векторизован
            let_log(f"##### [{chat_id}] Запись ID {last_msg_id} в БД обновлена сжатым текстом. #####")
    
    # LET-BL-VER-1.0: Удален блок "УПРЕЖДАЮЩЕЕ СЖАТИЕ STM" (line 794)
    # LET-BL-VER-1.0: Удален блок "ПРОВЕРКИ ПЕРЕПОЛНЕНИЯ" (line 815)
    
    # --- Финальная сборка ---
    # Мы просто получаем ПОЛНУЮ историю и вызываем сборщик 1 раз.
    # Сборщик сам разберется с RAG, лимитами и сжатием "на лету".
    history = get_history(chat_id)
    
    final_prompt = prompt_assembler(
        chat_id=chat_id,
        system_prompt=system_prompt,
        current_message=current_message,
        history=history
    )
    
    let_log("##### RAG CONSTRUCTOR ЗАВЕРШЕН #####\n")
    
    return final_prompt