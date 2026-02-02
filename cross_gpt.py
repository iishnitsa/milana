import os
from sqlite3 import connect, OperationalError
import chromadb
from chromadb.config import Settings
import importlib.util
import importlib
import sys
import re
import traceback
import time
import difflib
import pickle
import gzip
import json
import numpy as np
from sklearn.decomposition import PCA
import io
import base64
import inspect
from multiprocessing import queues
Empty = queues.Empty

class GlobalState:
    def __init__(self):
        self.stop_agent = False
        self.dialog_state = True
        self.dialog_result = ''
        self.conversations = 0
        self.tools_commands_dict = {}
        self.last_task_for_executor = {}
        self.now_try = '/:0'
        self.common_save_id = 0
        self.main_now_task = ''
        self.another_tools = []
        self.tools_str = ''
        self.milana_module_tools = []
        self.ivan_module_tools = []
        self.module_tools_keys = []
        self.max_critic_reactions = 1#3
        self.critic_reactions = {}
        self.critic_wants_retry = False
        self.critic_comment = ''
        self.librarian_max_attempts = 2
        self.system_tools_keys = []
        self.summ_attach = ''
        self.now_agent_id = 1
        self.gigo_web_search_allowed = True
        self.hierarchy_limit = 0
        self.write_results = 0
        self.need_owerwrite_operator = False
        self.need_owerwrite_executor = False
        self.task_delegated = False
        self.dialog_ended = False
global_state = GlobalState()

chat_path = ''
do_chat_construct = False
native_func_call = False
use_rag = None
ui_conn = None
cache_path = ''
cache_can_write = False
agent_func = None
start_dialog_command_name = ''
default_handlers_names = { # это из настроек должно выгружаться
    'doc': 'process_docx',
    'docx': 'process_docx',
    'txt': 'process_text',
    'pdf': 'process_pdf',
    'png': 'process_image',
    'jpg': 'process_image',
    'jpeg': 'process_image',
    'zip': 'process_zip',
    'xlsx': 'process_excel',
    'xls': 'process_excel',
}
# Список важных функций и модулей
important_functions = [
    'cacher',
    'read_cache',
    'write_cache',
    'rollback_cache',
    'get_embs',
    'ask_model',
    'sql_exec',
    'coll_exec',
    'tools_selector',
    'text_cutter',
    'get_input_message',
    'send_output_message',
    'librarian',
    'send_log_to_ui',
    'gigo'
]
# Список модулей, импорт которых считается важным
important_modules = ['chat_manager']
actual_handlers_names = {}
another_tools_files_addresses = []
input_info_loaders = {}
info_loaders = None
unified_tags = {
    "bos": "",
    "eos": "",
    "sys_start": "",
    "sys_end": "",
    "user_start": "",
    "user_end": "",
    "assist_start": "",
    "assist_end": "",
    "tool_def_start": "",
    "tool_def_end": "",
    "tool_call_start": "",
    "tool_call_end": "",
    "tool_result_start": "",
    "tool_result_end": "",
}
use_user = False
chunk_size = 1000 # TODO:
get_provider_embs = None
ask_provider_model = None
ask_provider_model_chat = None
memory_sql = None
client = None
milana_collection = None
user_collection = None
rag_collection = None
cache_counter = 1 # всегда начинается с 1
language = ''
is_print_log = True
is_save_log = True

def cacher(func):
    """Декоратор для функций с кэшированием (ask_model, get_embs, coll_exec, sql_exec)"""
    def wrapper(*args, **kwargs):
        cached = read_cache()
        if cached != [False]:
            if isinstance(cached[1], dict) and '__exception__' in cached[1]:
                # Восстанавливаем исключение из кэша
                exc_data = cached[1]['__exception__']
                # Создаем новое исключение с сообщением
                exc = RuntimeError(exc_data['message'])
                # Traceback не восстанавливаем - он не сериализуем
                raise exc
            let_log(f"[Используется кэшированный результат для {func.__name__}]")
            return cached[1]
        try:
            result = func(*args, **kwargs)
            write_cache(result)
            return result
        except Exception as e: # Сохраняем исключение в кэше без traceback
            exc_data = {
                '__exception__': {
                    'type': type(e).__name__,
                    'message': str(e),
                    'traceback_str': traceback.format_exc() # Traceback сохраняем как строку, а не как объект
                }
            }
            write_cache(exc_data)
            raise
    return wrapper

def ask_with_fallback(prompt, **kwargs):
    """Вызывает ask_model с автоматическим fallback через text_cutter при ошибке переполнения"""
    try: return ask_model(prompt, **kwargs)
    except RuntimeError as e:
        if 'ContextOverflowError' in str(e):
            cut_prompt = text_cutter(prompt)
            if 'attachments' in kwargs: kwargs['attachments'] = text_cutter(kwargs['attachments'])
            return ask_model(cut_prompt, **kwargs)
        else: raise

def load_locale(module_file, current_lang='en'):
    """Загружает локализацию для модуля из соответствующего файла"""
    locale_data = {}
    lang_file = module_file.replace('.py', '_lang.py')
    if os.path.isfile(lang_file):
        try:
            spec = importlib.util.spec_from_file_location('lang_module', lang_file)
            lang_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(lang_module)
            if hasattr(lang_module, 'locales'): locale_data = lang_module.locales.get(current_lang, {})
            if not locale_data and current_lang != 'en': let_log(f"⚠ Нет локализации для языка '{current_lang}' в {lang_file}")
        except Exception as e:
            if current_lang != 'en': let_log(f"⚠ Ошибка загрузки локализации {lang_file}: {e}")
    elif current_lang != 'en': let_log(f"Файл локализации {lang_file} не найден")
    return locale_data

def update_task(original_task, dialog_result, user_feedback, critic_feedback=None, is_critic=False):
    """Обновляет задачу с учетом фидбэка пользователя или критика"""
    review_texts = [user_review_text1, user_review_text2, user_review_text3, user_review_text4]
    has_review = any(text in original_task for text in review_texts)
    if is_critic and critic_feedback:
        if has_review: return original_task + user_review_text2 + dialog_result + user_review_text4 + critic_feedback
        else: return user_review_text1 + original_task + user_review_text2 + dialog_result + user_review_text4 + critic_feedback
    else:
        if has_review: return original_task + user_review_text2 + dialog_result + user_review_text3 + user_feedback
        else: return user_review_text1 + original_task + user_review_text2 + dialog_result + user_review_text3 + user_feedback

def load_special_mod(file_path, mod_type):
    """Загружает специальный модуль (web_search, ask_user)"""
    try:
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, 'main'):
            # Загружаем локализацию для специального модуля
            current_lang = globals().get('language', 'en')
            locale_data = load_locale(file_path, current_lang)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                match = re.match(r'^\s*[\'"]{3}\s*\n\s*([^\n]+)\n\s*([^\n]+)', content)
                if match:
                    cmd_name = match.group(1).strip()
                    desc = match.group(2).strip()
                    
                    # Применяем локализацию, если она есть
                    if locale_data and 'module_doc' in locale_data:
                        if len(locale_data['module_doc']) >= 2:
                            cmd_name = locale_data['module_doc'][0] or cmd_name
                            desc = locale_data['module_doc'][1] or desc
                    
                    if mod_type == 'web_search': globals()['web_search'] = module.main
                    elif mod_type == 'ask_user': globals()['ask_user'] = module.main
                    return (cmd_name, desc, module.main)
                else: let_log(f"⚠ Не удалось извлечь command_name из {file_path}"); return None
        else: let_log(f"⚠ Файл {mod_type} не содержит функцию main"); return None
    except Exception as e: let_log(f"⚠ Ошибка загрузки {mod_type}: {e}"); return None

def extract_command_markers(text, commands_dict):
    """Извлекает все маркеры команд из текста с использованием find_and_match_command"""
    if not text or not commands_dict: return []
    markers = []
    # Ищем маркеры по всему тексту
    pos = 0
    while pos < len(text):
        # Ищем следующий потенциальный маркер
        next_marker = text.find("!!!", pos)
        if next_marker == -1: break
        # Проверяем окрестность маркера (первые 5 символов от позиции маркера)
        check_text = text[max(0, next_marker-5):next_marker+10]
        match = find_and_match_command(check_text, commands_dict)
        if match:
            found_key, content = match
            # Находим конец маркера в основном тексте
            marker_end = text.find("!!!", next_marker + 3)
            if marker_end != -1:
                markers.append({
                    'start': next_marker,
                    'end': marker_end + 3,
                    'key': found_key,
                    'content': content,
                    'full_match': text[next_marker:marker_end+3]
                })
                pos = marker_end + 3
            else: pos = next_marker + 3
        else: pos = next_marker + 3
    return markers

def find_work_folder(file_name):
    real_path = os.path.realpath(file_name)
    if real_path.find('\\') != -1: slash = '\\'
    else: slash = '/'
    return file_name[:file_name.rfind(slash)], slash

folder_path, slash = find_work_folder(__file__)
sys.path.append(os.path.join(folder_path, 'system_tools'))
sys.path.append(os.path.join(folder_path, 'system_tools', 'milana'))
sys.path.append(os.path.join(folder_path, 'system_tools', 'ivan'))

def let_log(t):
    t = str(t)
    if is_print_log: print(t)
    if is_save_log:
        conn = connect(cache_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                id INTEGER PRIMARY KEY,
                value TEXT
            )
        ''')
        conn.commit()
        cursor.execute('SELECT MAX(id) FROM cache')
        row = cursor.fetchone()
        max_id = row[0] if row and row[0] is not None else -1
        conn.close()
        if max_id <= cache_counter:
            log_file = os.path.join(chat_path, 'log.txt')
            with open(log_file, 'a', encoding='utf-8') as f: f.write(f'{t}\n')

def traceprint(*args, **kwargs):
    stack = traceback.extract_stack()
    caller = stack[-2]
    line_number = caller.lineno
    filename = caller.filename.split("/")[-1]
    if not args: let_log(f"[{filename}:{line_number}]")
    else: let_log(f"[{filename}:{line_number}]:", *args, **kwargs)

def read_cache():
    global cache_counter
    global cache_can_write
    cache_conn = None
    try:
        cache_conn = connect(cache_path)
        cache_cursor = cache_conn.cursor()
        let_log(f"Попытка чтения кэша с id: {cache_counter}")
        cache_cursor.execute('SELECT value FROM cache WHERE id = ?', (cache_counter,))
        result = cache_cursor.fetchone()
        cache_conn.close()
        cache_conn = None
        if result is None:
            if cache_can_write: raise RuntimeError('Read/write sequence violation in the save system! Write command was expected.')
            cache_can_write = True
            let_log(f"Запись кэша с id {cache_counter} не найдена.")
            return [False]
        stored_data = result[0]
        marker = stored_data[0:1]
        data_part = stored_data[1:]
        if marker == b'\x00': decompressed_bytes = data_part
        elif marker == b'\x01': decompressed_bytes = gzip.decompress(data_part)
        try: deserialized_value = pickle.loads(decompressed_bytes)
        except Exception as pickle_error: raise
        let_log(f"[CACHE READ] id={cache_counter}")
        cache_counter += 1
        return [True, deserialized_value]
    except OperationalError as e:
        error_msg = str(e).lower()
        if "no such table" in error_msg or "cache" in error_msg:
            let_log(f"Таблица кэша не найдена, выполняется инициализация: {e}")
            if cache_conn:
                cache_cursor.execute("PRAGMA max_page_count = 2147483647;")
                cache_cursor.execute('''
                    CREATE TABLE IF NOT EXISTS cache (
                        id INTEGER PRIMARY KEY,
                        value BLOB
                    )
                ''')
                cache_conn.commit()
                cache_conn.close()
                cache_conn = None
                let_log("Таблица кэша только что создана, записей нет.")
                if cache_can_write: raise RuntimeError('Read/write sequence violation in the save system! Write command was expected.')
                cache_can_write = True
                return [False]
            else: raise
        else: raise
    except Exception as e:
        if cache_conn: cache_conn.close()
        error_msg = f'{e}'
        send_ui_no_cache(error_msg)
        raise SystemExit(error_msg)

def write_cache(content):
    global cache_counter
    global cache_can_write
    cache_conn = None
    try:
        if not cache_can_write: raise RuntimeError('Read/write sequence violation in the save system! Write command was expected.')
        cache_conn = connect(cache_path)
        cache_cursor = cache_conn.cursor()
        pickled_bytes = pickle.dumps(content, protocol=pickle.HIGHEST_PROTOCOL)
        raw_size = len(pickled_bytes)
        compressed = gzip.compress(pickled_bytes, compresslevel=9)
        size_uncompressed = 1 + raw_size
        size_compressed = 1 + len(compressed)
        if size_compressed < size_uncompressed: final_data = b'\x01' + compressed
        else: final_data = b'\x00' + pickled_bytes
        cache_cursor.execute('INSERT INTO cache (id, value) VALUES (?, ?)', (cache_counter, final_data))
        cache_conn.commit()
        cache_conn.close()
        cache_conn = None
        cache_can_write = False
        cache_counter += 1
        return True
    except Exception as e:
        if cache_conn: cache_conn.close()
        e = f'{e}'
        send_ui_no_cache(e)
        raise SystemExit(e)

def rollback_cache(num_records):
    global cache_counter
    cache_conn = None
    if num_records >= cache_counter: num_records = cache_counter - 1
    if num_records == 0: return False
    try:
        cache_conn = connect(cache_path)
        cache_cursor = cache_conn.cursor()
        cache_cursor.execute('DELETE FROM cache WHERE id IN (SELECT id FROM cache ORDER BY id DESC LIMIT ?)', (num_records,))
        cache_conn.commit()
        cache_conn.close()
        cache_conn = None
        cache_counter -= num_records
        let_log(f"Выполнен откат последних {num_records} записей.")
        cache_can_write = True
        return True
    except Exception as e:
        if cache_conn: cache_conn.close()
        e = f'{e}'
        send_ui_no_cache(e)
        raise SystemExit(e)

def send_ui_no_cache(t, attach=None, comm=''):
    message_data = {
        'text': t,
        'attachments': attach,
        'command': comm
    }
    try: ui_conn[1].put(message_data)
    except: pass

def send_log_to_ui(message: str):
    try: ui_conn[2].put(message)
    except Exception as e: let_log(f"Failed to send log to UI: {e}")

@cacher
def get_input_message(command=None, timeout=None, wait=False):
    answer = None
    if command:
        # Получаем сообщение из очереди
        while True:
            try:
                msg = ui_conn[0].get(block=(timeout is not None), timeout=timeout)
                if msg.get('command') == command: answer = msg; break
            except Empty: pass
            except Exception as e: let_log(f"Ошибка при получении сообщения: {e}"); break
    elif not wait:
        try: answer = ui_conn[0].get(block=(timeout is not None), timeout=timeout)
        except Empty: pass
        except Exception as e: let_log(f"Ошибка при получении сообщения: {e}")
    else:
        while True:
            try: answer = ui_conn[0].get(block=(timeout is not None), timeout=timeout); break
            except Empty: pass
            except Exception as e: let_log(f"Ошибка при получении сообщения: {e}"); break
    return answer

@cacher
def send_output_message(text=None, attachments=None, command=None):
    message_data = {
        'text': text or '',
        'attachments': attachments or None,
        'command': command
    }
    try: ui_conn[1].put(message_data)
    except Exception as e: let_log(f"Ошибка при отправке сообщения: {e}"); return
    return True

@cacher
def sql_exec(query, params=(), fetchone=False, fetchall=False):
    let_log('ОЧЕРЕДЬ')
    let_log(query)
    """Выполняет SQL-запрос с поддержкой кэширования"""
    try:
        cursor = memory_sql.cursor()
        cursor.execute(query, params)
        memory_sql.commit()
        result = None
        if fetchone:
            result = cursor.fetchone()
            if result and len(result) == 1: result = result[0]
        elif fetchall: result = cursor.fetchall()
        let_log('РЕЗУЛЬТАТ')
        let_log(result)
        return result
    except Exception as e:
        let_log(f"Ошибка SQL-запроса: {query} с {params} — {e}")
        return None

@cacher
def coll_exec(action: str,
              coll_name: str,
              *,
              # семантический поиск
              query_embeddings=None,
              filters: dict | None = None,
              doc_contains: str | None = None,
              # CRUD-поля
              ids: list[str] | None = None,
              documents: list[str] | None = None,
              metadatas: list[dict] | None = None,
              embeddings: list[list[float]] | None = None,
              # что вернуть
              fetch: str | list[str] = "documents",
              # параметры запроса
              n_results: int = 10,
              limit: int | None = None,
              offset: int | None = None,
              # поведение плоского fetch
              first: bool = True,
              flatten: bool = False,
              # для modify
              new_name: str | None = None,
              new_meta: dict | None = None,
              # если коллекция не в globals()
              client=None,
              # фильтрация релевантности
              relevance_coeff: float = 0.9,
              **kwargs):
    """
    Универсальная обёртка с кэшированием и поддержкой add/update/delete/query/get/count/modify.
    Поддержка $in и $nin для vector_id через множественные запросы (встроена).
    ВАЖНО: в этой версии обёртка не вычисляет расстояния (нет эвклидов/косинусов).
    
    Сжатие происходит ВСЕГДА для документов в milana_collection/user_collection.
    Формат: 'z' + Base64(gzip(данные)) для сжатых, 'n' + текст для несжатых.
    """
    def _compress_doc_always(doc: str | bytes) -> str:
        """
        Всегда пытается сжать документ.
        Возвращает: 
          - 'z' + Base64(gzip(документ)) если итоговый размер МЕНЬШЕ
          - 'n' + оригинальная строка если сжатие не выгодно
        """
        if doc is None: return None
        if isinstance(doc, (bytes, bytearray)):
            raw_bytes = bytes(doc)
            original_str = doc.decode("utf-8") if hasattr(doc, 'decode') else str(doc)
        else:
            original_str = str(doc)
            raw_bytes = original_str.encode("utf-8")
        uncompressed_str = "n" + original_str
        uncompressed_size = len(uncompressed_str.encode("utf-8"))
        try:
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9) as gz:
                gz.write(raw_bytes)
            compressed_bytes = buf.getvalue()
            compressed_b64 = base64.b64encode(compressed_bytes).decode("ascii")
            compressed_str = "z" + compressed_b64
            compressed_size = len(compressed_str.encode("utf-8"))
            if compressed_size < uncompressed_size: return compressed_str
            else: return uncompressed_str
        except: return uncompressed_str
    def _decompress_doc_always(comp: str) -> str:
        """
        Распаковывает документ с защитой от ложного распознавания.
        Определяет сжатие по первому символу: 'z'=сжато, 'n'=несжато.
        """
        if comp is None: return None
        if not comp: return comp
        first_char = comp[0]
        content = comp[1:]
        if first_char == 'z':
            decoded_bytes = base64.b64decode(content)
            buf = io.BytesIO(decoded_bytes)
            with gzip.GzipFile(fileobj=buf, mode='rb') as gz: decompressed_bytes = gz.read()
            return decompressed_bytes.decode("utf-8")
        elif first_char == 'n': return content
    def _compress_documents_always(coll_name_local, documents_list):
        """Всегда сжимает список документов с проверкой выгоды"""
        if documents_list is None: return None
        if coll_name_local not in ("milana_collection", "user_collection"): return documents_list
        out = []
        for idx, d in enumerate(documents_list):
            if d is None:
                out.append(None)
                continue
            compressed = _compress_doc_always(d)
            out.append(compressed)
        return out
    def _decompress_documents_always(coll_name_local, documents_list):
        """Распаковывает список документов"""
        if documents_list is None: return None
        if coll_name_local not in ("milana_collection", "user_collection"): return documents_list
        out = []
        for d in documents_list:
            if d is None:
                out.append(None)
                continue
            decompressed = _decompress_doc_always(d)
            out.append(decompressed)
        return out
    coll = globals().get(coll_name) or (client and client.get_collection(coll_name))
    if coll is None and action != "delete_collection": raise NameError(f"Collection '{coll_name}' not found")
    def _make_where(d):
        if not d: return None
        clauses = []
        for k, v in d.items():
            if isinstance(v, list): clauses.append({k: {"$in": v}})
            elif isinstance(v, dict) and any(op in v for op in ["$gt", "$gte", "$lt", "$lte", "$ne", "$eq", "$in", "$nin"]):
                clauses.append({k: v})
            else: clauses.append({k: v})
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}
    def _extract(resp, include):
        if len(include) > 1:
            out = {}
            for key in include:
                data = resp.get(key, []) or []
                if key == "documents":
                    data = _decompress_documents_always(coll_name, data)
                if flatten and isinstance(data, list) and data and isinstance(data[0], list):
                    data = [i for sub in data for i in sub]
                out[key] = data
            return out
        key = include[0]
        data = resp.get(key, []) or []
        if key == "documents": data = _decompress_documents_always(coll_name, data)
        if first: return data[0] if data else None
        if isinstance(data, list) and data and isinstance(data[0], list): return [i for sub in data for i in sub]
        return data
    def _filter_relevance(resp, coeff: float):
        if "distances" not in resp or resp["distances"] is None or not resp["distances"]: return resp
        dists = resp["distances"][0]
        if not dists: return resp
        best = min(dists)
        threshold = best * (1.0 + (1.0 - coeff))
        keep_idx = [i for i, d in enumerate(dists) if d <= threshold]
        if not keep_idx: return {k: [] for k in resp}
        out = {}
        for k, v in resp.items():
            if isinstance(v, list) and v and isinstance(v[0], list):
                out[k] = [[row[i] for i in keep_idx] for row in v]
            elif isinstance(v, list): out[k] = [v[i] for i in keep_idx]
            else: out[k] = v
        return out
    def _process_in_nin_operators(coll, filters, coll_name_local, get_results=True):
        let_log(f"[{coll_name_local}] Запуск обхода (in/nin) для ID с фильтрами: {filters}")
        nin_ids = set(filters.get('$nin', {}).get('vector_id', []))
        in_ids = set(filters.get('$in', {}).get('vector_id', []))
        base_where_filter = {
            k: v for k, v in filters.items()
            if k not in ('$in', '$nin', 'vector_id')
        }
        all_ids = set()
        offset = 0
        batch_size = 1000
        where_for_get = _make_where(base_where_filter)
        while True:
            r = coll.get(
                where=where_for_get,
                limit=batch_size,
                offset=offset,
                include=[]
            )
            current_ids = r.get('ids', [])
            if not current_ids:
                break
            all_ids.update(current_ids)
            offset += batch_size
            if len(current_ids) < batch_size: break
        let_log(f"[{coll_name_local}] Найдено {len(all_ids)} ID до фильтрации $in/$nin.")
        final_ids = all_ids
        if nin_ids: final_ids = final_ids - nin_ids
        if in_ids: final_ids = final_ids.intersection(in_ids)
        final_ids_list = list(final_ids)
        let_log(f"[{coll_name_local}] Осталось {len(final_ids_list)} ID после фильтрации $in/$nin.")
        if not get_results: return final_ids_list
        if final_ids_list:
            return coll.get(
                ids=final_ids_list,
                include=['metadatas', 'documents', 'embeddings']
            )
        return {'ids': [], 'metadatas': [], 'documents': [], 'embeddings': []}
    try:
        id_filters_present = (
            filters and
            (
                (filters.get('$nin') and isinstance(filters.get('$nin'), dict) and 'vector_id' in filters['$nin']) or
                (filters.get('$in') and isinstance(filters.get('$in'), dict) and 'vector_id' in filters['$in'])
            )
        )
        if action in ("query", "get") and id_filters_present:
            processed = _process_in_nin_operators(
                coll, filters, coll_name, get_results=True
            )
            if isinstance(processed, dict) and 'ids' in processed:
                resp = processed
                include = fetch if isinstance(fetch, list) else [fetch]
                if include == ["all"]:
                    include = ["ids", "documents", "metadatas", "embeddings", "distances"]
                if action == "query":
                    resp = _filter_relevance(resp, relevance_coeff)
                out = _extract(resp, include)
                return out
        if action == "add":
            docs_to_send = _compress_documents_always(coll_name, documents)
            out = coll.add(ids=ids, documents=docs_to_send, metadatas=metadatas, embeddings=embeddings, **kwargs)
            return out
        if action == "update":
            docs_to_send = _compress_documents_always(coll_name, documents)
            out = coll.update(ids=ids, documents=docs_to_send, metadatas=metadatas, embeddings=embeddings, **kwargs)
            return out
        if action == "delete":
            out = coll.delete(ids=ids, where=_make_where(filters), **kwargs)
            return out
        if action == "count":
            out = coll.count()
            return out
        if action == "modify":
            out = coll.modify(name=new_name, metadata=new_meta)
            return out
        if action == "delete_collection":
            if client is None: raise ValueError("client required for delete_collection")
            out = client.delete_collection(coll_name)
            return out
        if action in ("query", "get"):
            include = fetch if isinstance(fetch, list) else [fetch]
            if include == ["all"]: include = ["ids", "documents", "metadatas", "embeddings", "distances"]
            params = {}
            if action == "query":
                params.update({
                    "query_embeddings": query_embeddings or [],
                    "where": _make_where(filters),
                    "n_results": n_results
                })
                if doc_contains: params["where_document"] = {"$contains": doc_contains}
            else:
                params.update({
                    "where": _make_where(filters),
                    "limit": limit,
                    "offset": offset
                })
                if doc_contains: params["where_document"] = {"$contains": doc_contains}
            params["include"] = include
            params.update(kwargs)
            resp = (coll.query if action == "query" else coll.get)(**params)
            if not resp.get("ids") or not any(resp["ids"]):
                out = _extract(resp, include)
                return out
            if action == "query": resp = _filter_relevance(resp, relevance_coeff)
            out = _extract(resp, include)
            return out
        raise ValueError(f"[coll_exec] Unsupported action: {action}")
    except Exception as e:
        let_log(f"[coll_exec] Ошибка ({action}): {e}")
        return None

def load_chat_settings(chat_id):
    """Загрузка настроек чата из SQLite БД"""
    settings = {}
    settings_rows = sql_exec("SELECT key, value FROM settings", fetchall=True)
    if settings_rows:
        let_log(settings_rows)
        let_log(type(settings_rows))
        settings.update({row[0]: row[1] for row in settings_rows})
    another_tools_files = []
    default_mods = sql_exec("SELECT adress FROM default_mods WHERE enabled=?", (1,), fetchall=True)
    if default_mods: another_tools_files.extend([row[0] for row in default_mods])
    custom_mods = sql_exec("SELECT adress FROM custom_mods", fetchall=True)
    if custom_mods: another_tools_files.extend([row[0] for row in custom_mods])
    settings["another_tools"] = another_tools_files
    return settings

def load_initial_data(chat_id):
    """Загрузка начальных данных: задачи и вложений"""
    # Загрузка задачи (первого сообщения)
    task = sql_exec("SELECT text FROM messages WHERE id=?", (1,), fetchone=True)
    task = task if task else ""
    attachments = sql_exec("SELECT attachments FROM messages WHERE id=?", (1,), fetchone=True)
    if attachments and attachments:
        try: attachments = eval(attachments)
        except: attachments = []
    else: attachments = []
    return task, attachments

def globalize_language_packet(language):
    global container
    try:
        # Динамическая загрузка языкового модуля
        lang_module = __import__(f'lang.{language}.system_texts', fromlist=['system_text_container'])
        container = lang_module.system_text_container()
    except ImportError as e:
        let_log(f"Ошибка загрузки языкового модуля '{language}': {str(e)}")
        # Запасной вариант: попробовать загрузить стандартный модуль
        try:
            from texts import system_text_container
            container = system_text_container()
            let_log(f"Используются тексты по умолчанию")
        except ImportError:
            let_log("Критическая ошибка: не найден модуль с текстами!")
            return
    for attr in dir(container):
        if attr.startswith('__'): continue
        value = getattr(container, attr)
        if isinstance(value, str):
            try: setattr(container, attr, value)
            except Exception as e:
                let_log(f"Ошибка '{attr}': {e}")
                pass
    # Экспортируем атрибуты контейнера в глобальную область видимости
    for attr in dir(container):
        if attr.startswith('__'): continue
        value = getattr(container, attr)
        globals()[attr] = value
    let_log(f"Языковой пакет '{language}' загружен, переменные экспортированы")

def _check_module_uses_cross_gpt(file_contents):
    """Проверяет, использует ли модуль функции из cross_gpt, которые требуют кэширования."""
    # Удаляем комментарии из содержимого файла
    lines = file_contents.split('\n')
    clean_lines = []
    for line in lines:
        # Удаляем комментарии (все, что после #)
        if '#' in line: line = line[:line.index('#')]
        clean_lines.append(line)
    clean_content = '\n'.join(clean_lines)
    # 1. Проверяем импорт всего модуля cross_gpt или chat_manager
    for module in important_modules:
        if f'import {module}' in clean_content: return True
        if f'from {module} import' in clean_content: return True
    # 2. Проверяем многострочные импорты из cross_gpt
    # Ищем паттерн: from cross_gpt import ( ... )
    import_pattern = r'from\s+cross_gpt\s+import\s*\(([^)]+)\)'
    matches = re.findall(import_pattern, clean_content, re.DOTALL | re.IGNORECASE)
    for match in matches:
        # Разбиваем импортируемые имена по запятым
        imports = [imp.strip().split()[0] for imp in match.split(',') if imp.strip()]
        # Проверяем, есть ли среди них важные функции
        for imp in imports:
            # Убираем возможные as-алиасы
            if ' as ' in imp: imp = imp.split(' as ')[0].strip()
            if imp in important_functions: return True
    # 3. Проверяем однострочные импорты из cross_gpt
    # Ищем паттерн: from cross_gpt import func1, func2, func3
    single_line_pattern = r'from\s+cross_gpt\s+import\s+([^\(\n]+)'
    matches = re.findall(single_line_pattern, clean_content, re.IGNORECASE)
    for match in matches:
        # Исключаем импорт с *
        if '*' in match: return True
        # Разбиваем импортируемые имена по запятым
        imports = [imp.strip().split()[0] for imp in match.split(',') if imp.strip()]
        # Проверяем, есть ли среди них важные функции
        for imp in imports:
            # Убираем возможные as-алиасы
            if ' as ' in imp: imp = imp.split(' as ')[0].strip()
            if imp in important_functions: return True
    # 4. Проверяем импорты внутри функций (могут быть многострочными)
    # Ищем все вхождения from cross_gpt import независимо от позиции
    all_imports = re.findall(r'from\s+cross_gpt\s+import\s+.*?(?=\n|$)', clean_content, re.DOTALL | re.IGNORECASE)
    for import_stmt in all_imports:
        # Извлекаем часть после import
        import_part = import_stmt.split('import', 1)[1].strip()
        # Проверяем многострочный ли это импорт
        if '(' in import_part and ')' in import_part:
            # Многострочный импорт в одной строке
            start = import_part.find('(') + 1
            end = import_part.rfind(')')
            import_list = import_part[start:end]
        else: import_list = import_part # Однострочный импорт
        # Разбиваем по запятым
        imports = [imp.strip().split()[0] for imp in import_list.split(',') if imp.strip()]
        # Проверяем, есть ли среди них важные функции
        for imp in imports:
            # Убираем возможные as-алиасы
            if ' as ' in imp: imp = imp.split(' as ')[0].strip()
            if imp == '*': return True
            if imp in important_functions: return True
    # 5. Проверяем использование cross_gpt.функция
    for func in important_functions:
        if f'cross_gpt.{func}' in clean_content: return True
    # 6. Проверяем импорт из chat_manager
    # Ищем from chat_manager import что-угодно
    if re.search(r'from\s+chat_manager\s+import', clean_content, re.IGNORECASE): return True
    # 7. Проверяем использование chat_manager.что-угодно
    if 'chat_manager.' in clean_content: return True
    return False

def mod_loader(adrs):
    loaded_modules = []
    for mod_file in adrs:
        try:
            let_log(f"Processing: {mod_file}")
            if not os.path.isfile(mod_file):
                let_log(f"Файл {mod_file} не найден")
                continue
            # Получаем текущий язык
            current_lang = globals().get('language', 'en')
            locale_data = load_locale(mod_file, current_lang)
            # Читаем содержимое файла для поиска первых строк
            with open(mod_file, encoding='utf-8') as f: file_contents = f.read()
            # Парсим первые строки из файла (многострочный комментарий в начале)
            command_name = None
            description = None
            # Новый улучшенный regex для обработки разных форматов многострочных комментариев
            doc_match = re.match(r'^\s*[\'"]{3}\s*\n\s*([^\n]+)\n\s*([^\n]+)', file_contents)
            if not doc_match: doc_match = re.match(r'^\s*[\'"]{3}\s*([^\n]+)\n\s*([^\n]+)', file_contents)
            if doc_match:
                command_name = doc_match.group(1).strip()
                description = doc_match.group(2).strip()
            # Если есть локализация, берем оттуда (теперь для любого языка)
            if locale_data:
                if 'module_doc' in locale_data and len(locale_data['module_doc']) >= 2:
                    command_name = locale_data['module_doc'][0] or command_name
                    description = locale_data['module_doc'][1] or description
            # Проверяем, что получили command_name и description
            if not command_name or not description:
                let_log(f"Модуль {mod_file} должен содержать command_name и description (первые 2 строки файла или локализацию)")
                continue
            # Загружаем основной модуль
            module_name = os.path.splitext(mod_file)[0]
            spec = importlib.util.spec_from_file_location(module_name, mod_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # Проверка функции main
            if not hasattr(module, 'main'):
                let_log(f"Модуль {mod_file} должен содержать функцию main.")
                continue
            main_func = module.main
            if not callable(main_func):
                let_log(f"main в модуле {mod_file} должна быть функцией.")
                continue
            if main_func.__code__.co_argcount != 1:
                let_log(f"Функция main в модуле {mod_file} должна принимать ровно 1 аргумент.")
                continue
            # Инициализация атрибутов
            should_initialize = re.search(r"if\s+not\s+hasattr\s*\(\s*main\s*,\s*['\"]attr_names['\"]\s*\)", file_contents)
            if should_initialize:
                try:
                    main_func(None)  # Инициализация атрибутов по умолчанию
                    attr_list = getattr(main_func, 'attr_names', [])
                    # Применяем локализацию для любого языка, если есть данные
                    if locale_data:
                        for attr in attr_list:
                            localized_key = f'main.{attr}'
                            if localized_key in locale_data:
                                setattr(main_func, attr, locale_data[localized_key])
                                # Только для не-английских языков пишем лог о применении локализации
                                if current_lang != 'en': let_log(f"✓ Локализация: {attr} в {mod_file}")
                except Exception as e:
                    let_log(f"⚠ Ошибка при инициализации {mod_file}: {e}")
                    continue
            else: let_log('НЕ ДОЛЖЕН')
            if _check_module_uses_cross_gpt(file_contents): global_state.system_tools_keys.append(command_name)
            # Добавляем модуль в список
            loaded_modules.append((command_name, description, main_func))
        except Exception as e:
            let_log(f"⚠ Ошибка при обработке {mod_file}: {e}")
            continue
    return loaded_modules

def system_tools_loader():
    def fpf(directory):
        return [
            os.path.join(directory, entry)
            for entry in os.listdir(directory)
            if (os.path.isfile(os.path.join(directory, entry)) and 
                entry.endswith('.py') and 
                not entry.endswith('_lang.py'))
        ]
    def globalize_by_filename(modules, files):
        target = sys.modules[__name__].__dict__
        for mod, path in zip(modules, files):
            filename = os.path.splitext(os.path.basename(path))[0]
            func = mod[2]
            target[filename] = func
            let_log(f"→ Глобализовано: {filename} → {func}")
    # Локальные переменные для поиска start_dialog
    found_start_dialog_index = -1
    found_start_dialog_command = ''
    # Пути
    common_path = os.path.join(folder_path, 'system_tools')
    milana_path = os.path.join(folder_path, f'system_tools{slash}milana')
    ivan_path = os.path.join(folder_path, f'system_tools{slash}ivan')
    # Собираем списки файлов
    common_files = fpf(common_path)
    milana_files = fpf(milana_path)
    ivan_files = fpf(ivan_path)
    # Загружаем модули в новом порядке: общие, милана, иван
    common_modules = mod_loader(common_files)
    globalize_by_filename(common_modules, common_files)
    milana_modules = mod_loader(milana_files)
    globalize_by_filename(milana_modules, milana_files)
    ivan_modules   = mod_loader(ivan_files)
    globalize_by_filename(ivan_modules, ivan_files)
    # Находим start_dialog в модулях Ивана
    for i, (cmd_t, desc_tokens, func) in enumerate(ivan_modules):
        filename = os.path.splitext(os.path.basename(ivan_files[i]))[0]
        if filename == 'start_dialog':
            found_start_dialog_index = i
            found_start_dialog_command = cmd_t
            break
    # Устанавливаем глобальную переменную start_dialog_command_name
    if found_start_dialog_index != -1:
        global start_dialog_command_name
        start_dialog_command_name = found_start_dialog_command
    # Формируем словари с ключом — кортеж токенов имени команды
    def to_dict(modules):
        d = {}
        for cmd_t, desc_tokens, func in modules: d[cmd_t] = (desc_tokens, func)
        return d
    common_dict = to_dict(common_modules)
    milana_dict = to_dict(milana_modules)
    ivan_dict = to_dict(ivan_modules)
    # Выводим отладочную информацию
    let_log("\nЗагруженные системные команды:")
    for cmd in global_state.system_tools_keys: let_log(cmd)
    return (
        {**common_dict, **ivan_dict},   # common + ivan
        {**common_dict, **milana_dict}  # common + milana
    )

@cacher
def get_embs(text: str): # TODO: протестировать
    # 1) кеширование
    send_log_to_ui('embeddings:\n' + text)
    start = time.time()
    # [ИСПРАВЛЕНИЕ 1 - БЕЗОПАСНОСТЬ] Добавляем ограничение на количество попыток
    # убери цикл вообще, оставь 2 попытки или одну, хз, а может просто так оставить
    max_attempts_emb = 4
    current_attempt = 0
    # 2) начинаем с одного куска — весь текст
    pieces = [text]
    while True:
        current_attempt += 1 # Увеличиваем счетчик
        if current_attempt > max_attempts_emb:
            let_log(f'Превышено максимальное количество попыток ({max_attempts_emb}) дробления текста для эмбеддингов. Возврат пустого списка.')
            return []
        try:
            # 3) ПРЕДВАРИТЕЛЬНАЯ ПРОВЕРКА ПЕРЕПОЛНЕНИЯ
            for i, p in enumerate(pieces):
                estimated_tokens = len(p) * text_tokens_coefficient
                if estimated_tokens > emb_token_limit: raise RuntimeError('ContextOverflowError')
            pieces = [piece for piece in pieces if piece.strip()]
            if not pieces:
                let_log("Текст состоит из пробелов, возврат пустого эмбеддинга.")
                all_embs = []
                break
            # 4) если проверка пройдена - получаем эмбеддинги
            all_embs_raw = []
            for p in pieces:
                try: answer = get_provider_embs(p)
                except Exception as e:
                    if 'ContextOverflowError' in str(e): raise
                    let_log(e)
                    send_ui_no_cache(f'{error_in_provider}\n{e}')
                    while True:
                        time.sleep(60)
                        try:
                            answer = get_provider_embs(p)
                            send_ui_no_cache(success_in_provider)
                            break
                        except Exception as e:
                            if 'ContextOverflowError' in str(e): raise
                all_embs_raw.append(answer)
            # [ИСПРАВЛЕНИЕ 2 - РАЗМЕРНОСТЬ] Разворачиваем (flatten) результат на один уровень.
            # Превращаем [[emb1], [emb2]] в [emb1, emb2]
            all_embs = [emb for sublist in all_embs_raw for emb in sublist]
            break # Успешный выход из цикла
        except RuntimeError as e:
            # [СТРОГАЯ ОБРАБОТКА ОШИБОК 1] Продолжаем цикл только при ContextOverflowError
            if 'ContextOverflowError' not in str(e): raise # Немедленно поднимаем любую другую RuntimeError, не связанную с переполнением
            let_log('интеллектуальное дробление для эмбеддингов')
            # 5) ИНТЕЛЛЕКТУАЛЬНОЕ ДЕЛЕНИЕ С ВЫЧИСЛЕНИЕМ КОЛИЧЕСТВА ЧАСТЕЙ
            new_pieces = []
            need_retry = False 
            for p in pieces:
                estimated_tokens = len(p) * text_tokens_coefficient
                if estimated_tokens <= emb_token_limit: new_pieces.append(p)
                else:
                    # Вычисляем на сколько частей нужно разделить (с запасом 10%)
                    # [ИСПРАВЛЕНИЕ - ДЕЛЕНИЕ] Делим не на 2, а на нужное количество
                    needed_parts = max(2, int(estimated_tokens / emb_token_limit * 1.10) + 1)
                    let_log(f'Разделение текста на {needed_parts} частей')
                    p_len = len(p)
                    part_len = p_len // needed_parts
                    for i in range(needed_parts):
                        start_idx = i * part_len
                        end_idx = (i + 1) * part_len if i < needed_parts - 1 else p_len
                        new_pieces.append(p[start_idx:end_idx])
                    need_retry = True
            # Защита от бесконечного цикла
            if not need_retry and current_attempt > 1:
                let_log("Логика деления не смогла уменьшить части. Выход.")
                return []
            pieces = new_pieces
            let_log(f'Разделено на {len(pieces)} частей')
        except Exception as e:
            # [СТРОГАЯ ОБРАБОТКА ОШИБОК 2] Любая другая критическая ошибка
            traceprint()
            let_log(f'Критическая ошибка (не ContextOverflowError) при получении эмбеддингов: {e}')
            raise # Немедленно поднимает
    elapsed = time.time() - start
    let_log(f'Получение эмбеддингов выполнено за {elapsed:.2f} секунд')
    # 6) УСРЕДНЯЕМ эмбеддинги
    if not all_embs: flat_embs = []
    elif len(all_embs) == 1: flat_embs = all_embs[0]
    else:
        let_log(f'Усреднение {len(all_embs)} эмбеддингов')
        # Работает корректно, так как all_embs теперь 2D: [emb1, emb2, ...]
        embs_array = np.array(all_embs)
        flat_embs = np.mean(embs_array, axis=0).tolist()
    # 7) сохраняем в кеш
    return flat_embs

def get_token_limit(): return token_limit

def get_text_tokens_coefficient(): return text_tokens_coefficient

def _execute_with_cache_and_error_handling(generation_func):
    """
    Вспомогательная функция для обработки кэширования и ошибок при генерации.
    """
    start = time.time()
    try: generated = generation_func()
    except Exception as e:
        if 'ContextOverflowError' in str(e): raise RuntimeError("ContextOverflowError")
        let_log(e)
        send_ui_no_cache(f'{error_in_provider}\n{e}')
        while True:
            time.sleep(60)
            try:
                start = time.time()
                generated = generation_func()
                send_ui_no_cache(success_in_provider)
                break
            except Exception as e:
                if 'ContextOverflowError' in str(e): raise RuntimeError("ContextOverflowError")
    elapsed = time.time() - start
    send_log_to_ui('model:\n' + generated)
    traceprint()
    let_log(generated)
    let_log(f'СГЕНЕРИРОВАНО {len(generated)} токенов за {elapsed:.2f}s')
    let_log(f'Генерация заняла {elapsed:.2f}s, вывод {len(generated)} токенов')
    return generated

@cacher
def ask_model(prompt_text, 
              system_prompt: str = None,
              all_user: bool = False,
              limit: int = None,
              temperature: float = 0.6,
              **extra_params) -> str:
    let_log(prompt_text)
    let_log(f'ВХОД {len(prompt_text)} токенов')
    # Проверка длины контекста
    if len(prompt_text) * text_tokens_coefficient > token_limit - 1000:
        raise RuntimeError("ContextOverflowError")
    # --- Обработка пользовательского ввода ---
    if use_user:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        input_text = simpledialog.askstring("Ввод текста", "Пожалуйста, введите текст:", parent=root)
        root.destroy()
        if input_text is not None:
            if input_text != "":
                return input_text
        let_log("[Пользователь нажал Cancel, используется генерация моделью]")
    # --- Обработка особых случаев (system_prompt и all_user) ---
    # Особый случай 1: system_prompt
    if system_prompt:
        let_log("Режим (Особый случай): system_prompt -> chat/completions")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text}
        ]
        generation_params = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": limit or token_limit
        }
        for name, val in extra_params.items(): generation_params[name] = val
        return _execute_with_cache_and_error_handling(lambda: _process_chat_response(ask_provider_model_chat(generation_params)))
    # Особый случай 2: all_user
    if all_user:
        let_log("Режим (Особый случай): all_user=True -> chat/completions")
        messages = [{"role": "user", "content": prompt_text}]
        generation_params = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": limit or token_limit
        }
        for name, val in extra_params.items(): generation_params[name] = val
        return _execute_with_cache_and_error_handling(lambda: _process_chat_response(ask_provider_model_chat(generation_params)))
    # --- Определение режима работы на основе do_chat_construct (1, 2, 3) ---
    # Режим 1: Подача строки (completions)
    if not do_chat_construct:
        let_log("Режим 1 (do_chat_construct=1): Подача строки -> completions")
        if not native_func_call: parsed_msgs = _parse_roles_to_messages_no_functions(prompt_text)
        else: parsed_msgs = _parse_roles_to_messages_functions(prompt_text, global_state.now_agent_id)
        generation_params = {
            "prompt": _serialize_messages_to_prompt(parsed_msgs),
            "temperature": temperature,
            "max_tokens": limit or token_limit,
            "echo": False
        }
        for name, val in extra_params.items(): generation_params[name] = val
        return _execute_with_cache_and_error_handling(lambda: ask_provider_model(generation_params))
    # Режим 2: Парсинг чата БЕЗ function call
    elif do_chat_construct and not native_func_call:
        let_log("Режим 2 (do_chat_construct=2): Парсинг (без функций) -> chat/completions")
        messages = _parse_roles_to_messages_no_functions(prompt_text) # Используем старый парсер
        generation_params = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": limit or token_limit
        }
        for name, val in extra_params.items(): generation_params[name] = val
        return _execute_with_cache_and_error_handling(lambda: _process_chat_response(ask_provider_model_chat(generation_params)))
    # Режим 3: Парсинг чата С function call
    elif do_chat_construct and native_func_call:
        let_log("Режим 3 (do_chat_construct=3): Парсинг (С функциями) -> chat/completions")
        let_log(global_state.now_agent_id)
        # 1. Парсим историю (как и раньше)
        messages = _parse_roles_to_messages_functions(prompt_text, global_state.now_agent_id)
        # 2. Получаем и форматируем доступные инструменты
        now_commands = global_state.tools_commands_dict.get(global_state.now_agent_id, {})
        let_log(now_commands)
        formatted_tools = _format_tools_for_api(now_commands)
        generation_params = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": limit or token_limit
        }
        # 3. Добавляем инструменты в запрос, если они есть
        let_log(formatted_tools)
        if formatted_tools:
            let_log('ДОБАВЛЯЕМ ИНСТРУМЕНТЫ')
            generation_params["tools"] = formatted_tools
            generation_params["tool_choice"] = "auto" # Позволяем модели решать, когда вызывать
        for name, val in extra_params.items(): generation_params[name] = val
        let_log(generation_params)
        # 4. Вызываем модель и получаем полный ответ
        try: openai_response = ask_provider_model_chat(generation_params)
        except Exception as e:
            if 'ContextOverflowError' in str(e): raise RuntimeError("ContextOverflowError")
            let_log(e)
            send_ui_no_cache(f'{error_in_provider}\n{e}')
            while True:
                time.sleep(60)
                try:
                    openai_response = ask_provider_model_chat(generation_params)
                    send_ui_no_cache(success_in_provider)
                    break
                except Exception as e:
                    if 'ContextOverflowError' in str(e): raise RuntimeError("ContextOverflowError")
        # 5. Обрабатываем ответ как словарь
        let_log(openai_response)
        # Извлекаем данные из ответа API
        if "choices" not in openai_response or not openai_response["choices"]:
            let_log("ask_model: Некорректный формат ответа - нет choices")
            raise RuntimeError("Некорректный формат ответа - нет choices")
        choice = openai_response["choices"][0]
        message = choice.get("message", {})
        response_content = message.get("content", "") or ""
        tool_calls = message.get("tool_calls")
        if tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call['function']['name']
                arguments_json_str = tool_call['function']['arguments']
                try:
                    args_dict = json.loads(arguments_json_str)
                    # Извлекаем значение по ключу 'arguments'
                    arguments_str = args_dict.get('arguments', '')
                except Exception: arguments_str = arguments_json_str
                # Собираем маркер, который ожидает tools_selector
                marker = f"\n!!!{function_name}!!!{arguments_str}"
                response_content = marker + response_content
            return response_content
        elif response_content: return response_content
        return response_content

def _process_chat_response(api_response):
    """
    Обрабатывает ответ от ask_provider_model_chat (словарь) и извлекает текстовый контент
    Теперь поддерживает как OpenAI-совместимый формат, так и формат Ollama
    """
    # Проверяем наличие поля choices (OpenAI-совместимый формат)
    if "choices" in api_response and api_response["choices"]:
        choice = api_response["choices"][0]
        if "message" in choice and "content" in choice["message"]:
            result = choice["message"]["content"].strip()
            let_log(f"_process_chat_response: Результат: '{result}'")
            return result
    # Если нет choices, проверяем прямой формат Ollama
    elif "message" in api_response:
        message = api_response["message"]
        if isinstance(message, dict) and "content" in message:
            result = message["content"].strip()
            let_log(f"_process_chat_response (Ollama формат): Результат: '{result}'")
            return result
        elif isinstance(message, str):
            result = message.strip()
            let_log(f"_process_chat_response (Ollama формат строка): Результат: '{result}'")
            return result
    # Проверяем поле response (для обратной совместимости)
    elif "response" in api_response:
        result = api_response["response"].strip()
        let_log(f"_process_chat_response (response поле): Результат: '{result}'")
        return result
    # Если ничего не найдено
    let_log(f"_process_chat_response: Некорректный формат ответа: {api_response}")
    raise RuntimeError("Некорректный формат ответа - невозможно извлечь содержимое")

def _parse_roles_to_messages_no_functions(prompt):
    """
    Полноценный парсинг текста с ролями в список сообщений для chat/completions.
    Теперь учитывает структуру промпта из RAG конструктора.
    """
    messages = []
    remaining_prompt = prompt
    # 1. Проверяем наличие метки "## Последние сообщения:"
    if last_messages_marker in remaining_prompt:
        # Разделяем на часть до метки (system) и после (диалог)
        system_part, dialog_part = remaining_prompt.split(last_messages_marker, 1)
        # Часть до метки - это системный промпт
        if system_part.strip(): messages.append({"role": "system", "content": system_part.strip()})
        remaining_prompt = dialog_part
    else:
        # Новая логика: ищем первую роль в промпте
        roles_to_find = [operator_role_text, worker_role_text, func_role_text]
        first_role_pos = -1
        first_role = None
        for role in roles_to_find:
            pos = remaining_prompt.find(role)
            if pos != -1 and (first_role_pos == -1 or pos < first_role_pos):
                first_role_pos = pos
                first_role = role
        # Если найдена роль, разделяем на system и остальное
        if first_role_pos != -1:
            system_content = remaining_prompt[:first_role_pos].strip()
            if system_content: messages.append({"role": "system", "content": system_content})
            remaining_prompt = remaining_prompt[first_role_pos:]
        else:
            # Если ролей нет, но текст начинается не с роли - считаем системным промптом
            if remaining_prompt.strip() and not any(remaining_prompt.strip().startswith(role) for role in roles_to_find):
                messages.append({"role": "system", "content": remaining_prompt.strip()})
                remaining_prompt = ""
    # 2. Определяем роли на основе фактического содержимого и чередования
    # Ищем все вхождения ролей в оставшемся промпте
    roles_to_find = [operator_role_text, worker_role_text, func_role_text]
    # Создаем список всех найденных маркеров ролей с их позициями
    found_roles = []
    for role in roles_to_find:
        start_idx = 0
        while True:
            pos = remaining_prompt.find(role, start_idx)
            if pos == -1: break
            found_roles.append({
                'pos': pos, 
                'role': role,
                'type': 'operator' if role == operator_role_text else 
                       'worker' if role == worker_role_text else 
                       'function'
            })
            start_idx = pos + len(role)
    # Сортируем по позиции
    found_roles.sort(key=lambda x: x['pos'])
    # Если не найдены роли, но текст есть - считаем все пользовательским сообщением
    if not found_roles and remaining_prompt.strip():
        # Удаляем только operator и worker маркеры, func_role_text оставляем как есть
        clean_content = remaining_prompt.strip()
        for role in [operator_role_text, worker_role_text]: clean_content = clean_content.replace(role, '').strip()
        messages.append({"role": "user", "content": clean_content})
        return messages
    # Обрабатываем найденные роли с учетом чередования
    # Теперь просто чередуем user/assistant после system
    current_role = "user"  # Начинаем с пользователя
    for i, role_info in enumerate(found_roles):
        role_text = role_info['role']
        role_type = role_info['type']
        # Начало контента для этой роли
        content_start = role_info['pos'] + len(role_text)
        # Конец контента - до следующей роли или до конца текста
        content_end = len(remaining_prompt)
        if i + 1 < len(found_roles): content_end = found_roles[i + 1]['pos']
        content = remaining_prompt[content_start:content_end].strip()
        # Очистка содержимого в зависимости от типа роли
        if role_type == 'function':
            # Для function роли: убираем только начальный \n если есть, но оставляем сам маркер
            clean_content = content
            # Убираем начальный перевод строки если он есть
            if clean_content.startswith('\n'): clean_content = clean_content[1:].strip()
            # Добавляем func_role_text в начало содержимого
            clean_content = func_role_text.replace('\n', '') + clean_content
        else:
            # Для operator и worker ролей: полностью удаляем маркеры
            clean_content = content
            for role in [operator_role_text, worker_role_text]: clean_content = clean_content.replace(role, '').strip()
        if not clean_content: continue
        # Определяем роль для API на основе простого чередования
        # Первое сообщение после system - user, следующее - assistant, и т.д.
        api_role = current_role
        # Переключаем роль для следующего сообщения
        current_role = "assistant" if current_role == "user" else "user"
        messages.append({
            "role": api_role, 
            "content": clean_content
        })
    # Если вообще нет сообщений, но промпт не пустой
    if not messages and prompt.strip():
        clean_content = prompt.strip()
        # Удаляем только operator и worker маркеры
        for role in [operator_role_text, worker_role_text]: clean_content = clean_content.replace(role, '').strip()
        messages.append({"role": "user", "content": clean_content})
    # Проверка на четность количества сообщений (включая системное)
    # Если нечетное - удаляем последнее сообщение
    if len(messages) % 2 != 0:
        removed_message = messages.pop()
        let_log(f"Удалено последнее сообщение (нечетное количество): {removed_message['role']} - {removed_message['content'][:100]}...")
    let_log(f"Спарсено сообщений (Режим 2): {len(messages)}")
    for i, msg in enumerate(messages): let_log(f"Сообщение {i}: {msg['role']} - {msg['content'][:100]}...")
    return messages

def _parse_roles_to_messages_functions(prompt_text, sid):
    """
    Парсер для режима с функциями, учитывающий структуру RAG промпта.
    Обрабатывает как вызовы функций (маркеры !!!), так и ответы функций (префикс func_text_for_parse).
    Включает проверку на одно сообщение и замену на ответ функции.
    """
    func_text_for_parse = func_role_text.replace('\n', '')
    # Используем базовый парсер, который теперь понимает RAG структуру
    base_messages = _parse_roles_to_messages_no_functions(prompt_text)
    if not base_messages: return base_messages
    # [НОВАЯ ЛОГИКА] Проверяем количество не-системных сообщений
    non_system_messages = [msg for msg in base_messages if msg.get("role") != "system"]
    if len(non_system_messages) == 1:
        # Если только одно не-системное сообщение
        single_message = non_system_messages[0]
        content = single_message.get("content", "")
        # Проверяем, является ли это сообщение ответом функции (содержит func_text_for_parse)
        if (start_dialog_command_name != '' and 
            global_state.now_agent_id % 2 != 0 and  # Проверяем чётность now_agent_id вместо conversations
            func_text_for_parse in content):
            let_log("Обнаружено одно сообщение с ответом функции - выполняем замену")
            # Вырезаем func_text_for_parse из начала контента (как при множественных сообщениях)
            cleaned_content = content
            if content.startswith(func_text_for_parse): cleaned_content = content[len(func_text_for_parse):].strip()
            elif func_text_for_parse in content: cleaned_content = content.replace(func_text_for_parse, '', 1).strip()
            # Создаем сообщение с ответом функции
            function_response_message = {
                "role": "function",
                "name": start_dialog_command_name,
                "content": cleaned_content
            }
            # Заменяем исходное сообщение на ответ функции в base_messages
            for i, msg in enumerate(base_messages):
                if msg.get("role") != "system" and msg.get("content") == content:
                    base_messages[i] = function_response_message
                    break
    # словарь команд для сессии (формат {'name':('desc', func)})
    try: now_commands = global_state.tools_commands_dict.get(sid, {})
    except Exception: now_commands = {}
    # для каждого сообщения заранее вычислим маркеры (если есть)
    markers_for_msg = []
    for m in base_messages:
        txt = m.get("content", "") if isinstance(m, dict) else ""
        matched = find_and_match_command(txt, now_commands)
        markers_for_msg.append(matched)  # либо (found_key, content) либо None
    result = []
    i = 0
    while i < len(base_messages):
        msg = dict(base_messages[i])  # shallow copy
        role = (msg.get("role") or "").lower()
        content = msg.get("content", "")
        # проверяем, похоже ли текущее сообщение на ответ функции
        content_l = (content.lstrip() or "").lower()
        is_function_like = (role == "function") or content_l.startswith(func_text_for_parse)
        if is_function_like:
            prev_index = i - 1
            if prev_index >= 0:
                # берем маркер, найденный в предыдущем сообщении
                prev_marker = markers_for_msg[prev_index]
                if prev_marker:
                    found_key, args_str = prev_marker
                    # проверяем, что такой ключ есть в now_commands
                    try:
                        if found_key in now_commands:
                            # удаляем маркер из текста предыдущего сообщения
                            prev_msg_target = result[-1] if result else base_messages[prev_index]
                            prev_txt = prev_msg_target.get("content", "")
                            # Используем extract_command_markers для поиска маркеров
                            markers = extract_command_markers(prev_txt, now_commands)
                            if markers:
                                # Находим маркер с нужным ключом
                                for marker in markers:
                                    if marker['key'] == found_key:
                                        # Удаляем этот маркер
                                        new_prev_txt = prev_txt[:marker['start']] + prev_txt[marker['end']:]
                                        new_prev_txt = new_prev_txt.strip()
                                        if result: result[-1]["content"] = new_prev_txt
                                        else: base_messages[prev_index]["content"] = new_prev_txt
                                        break
                            else:
                                # Fallback: если не нашли маркер через extract_command_markers
                                marker_token = "!!!" + found_key + "!!!"
                                if marker_token in prev_txt:
                                    new_prev_txt = prev_txt.replace(marker_token, "", 1).strip()
                                    if result: result[-1]["content"] = new_prev_txt
                                    else: base_messages[prev_index]["content"] = new_prev_txt
                            # создаем function-role сообщение с правильным именем и очищенным контентом
                            # вырезаем func_text_for_parse из начала контента
                            cleaned_content = content
                            if content_l.startswith(func_text_for_parse): cleaned_content = content[len(func_text_for_parse):].strip()
                            function_message = {
                                "role": "function",
                                "name": found_key,
                                "content": cleaned_content
                            }
                            result.append(function_message)
                            i += 1
                            continue
                        else: msg["role"] = "assistant"
                    except Exception: msg["role"] = "assistant"
                else: msg["role"] = "assistant"
            else: msg["role"] = "assistant"
        # проверяем, содержит ли текущее сообщение маркер вызова функции
        current_marker = markers_for_msg[i]
        if current_marker and role in ("assistant", ""):
            found_key, args_str = current_marker
            if found_key in now_commands:
                # создаем сообщение с tool_calls вместо обычного assistant
                tool_call_id = f"call_{i}_{len(result)}"
                tool_call = {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": found_key,
                        "arguments": args_str
                    }
                }
                # удаляем маркер из текста с помощью extract_command_markers
                cleaned_content = content
                markers = extract_command_markers(content, now_commands)
                if markers:
                    for marker in markers:
                        if marker['key'] == found_key:
                            cleaned_content = content[:marker['start']] + content[marker['end']:]
                            cleaned_content = cleaned_content.strip()
                            break
                else:
                    # Fallback
                    marker_token = "!!!" + found_key + "!!!"
                    if marker_token in content: cleaned_content = content.replace(marker_token, "", 1).strip()
                # создаем сообщение с tool_calls
                tool_message = {
                    "role": "assistant",
                    "content": cleaned_content,
                    "tool_calls": [tool_call]
                }
                result.append(tool_message)
                i += 1
                continue
        # добавляем текущее сообщение (если не заменили на function или tool_calls)
        result.append(msg)
        i += 1
    let_log(f"Спарсено сообщений (Режим 3): {len(result)}")
    let_log(f"Итоговые сообщения (Режим 3): {result}")
    return result

def _format_tools_for_api(commands_dict):
    """
    Преобразует словарь команд из global_state в формат, 
    ожидаемый API (например, OpenAI) для параметра 'tools'.
    Ожидает на входе: {'name': ('description', func), ...}
    Возвращает: 
    [
      {
        "type": "function",
        "function": {
          "name": "name",
          "description": "description",
          "parameters": { ... }
        }
      },
      ...
    ]
    """
    if not commands_dict: return []
    tools_list = []
    for name, details in commands_dict.items():
        if not isinstance(name, str) or not (isinstance(details, (tuple, list)) and len(details) >= 1): continue
        description = str(details[0])
        # Так как вся система ожидает ОДНУ строку аргументов 
        # (всё, что после !!!command!!!), мы определяем один
        # строковый параметр с именем "arguments".
        tool_definition = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arguments": {
                            "type": "string",
                            "description": "Аргументы для команды в виде единой строки (весь текст, который должен идти после !!!)."
                        }
                    },
                    "required": ["arguments"]
                }
            }
        }
        tools_list.append(tool_definition)
    return tools_list

def remove_role_markers_from_content(messages):
    """
    Удаляет маркеры ролей из содержимого сообщений.
    (Вызывается из _parse_roles_to_messages_no_functions)
    """
    cleaned_messages = []
    for msg in messages:
        content = msg["content"]
        if content.startswith(operator_role_text): content = content[len(operator_role_text):].strip()
        elif content.startswith(worker_role_text): content = content[len(worker_role_text):].strip()
        elif content.startswith(func_role_text):
            role_with_space = func_role_text.replace('\n', ' ')
            content = content.replace(func_role_text, role_with_space, 1).strip()
        content = content.replace(operator_role_text, '').replace(worker_role_text, '').strip()
        cleaned_messages.append({
            "role": msg["role"],
            "content": content
        })
    return cleaned_messages

def parse_prompt_response(prompt, default_value):
    try: response = ask_model(prompt, all_user=True)
    except RuntimeError as e:
        if 'ContextOverflowError' in str(e):
            try: response = ask_model(text_cutter(prompt), all_user=True)
            except Exception: return default_value
        else: raise
    first_chars = response[:5]
    for char in first_chars:
        if char.isdigit():
            try: return int(char)
            except ValueError: continue
    return default_value

def _serialize_messages_to_prompt(messages):
    """
    Сериализует список сообщений (role/content) в единую строку промпта,
    используя теги из словаря 'unified_tags'.
    Поддерживает роли: 'system', 'tools', 'user', 'assistant', 'tool_call', 'tool'/'function'.
    """
    serialized_prompt = []
    # 1. Извлечение служебных ролей (system, tools), которые стоят вне основного цикла
    system_prompt = next((m['content'] for m in messages if m['role'] == 'system'), None)
    tools_definition = next((m['content'] for m in messages if m['role'] == 'tools'), None)
    # Отфильтровываем служебные роли для итерации по основному диалогу
    dialog_messages = [m for m in messages if m['role'] not in ('system', 'tools')]
    if not dialog_messages: return ""
    # 2. Добавление описания инструментов (TOOLS) - в начале промпта
    if tools_definition and unified_tags.get('tool_def_start'):
        serialized_prompt.append(unified_tags['tool_def_start'])
        serialized_prompt.append(tools_definition) 
        serialized_prompt.append(unified_tags['tool_def_end'])
    # 3. Инициализация и обработка витков диалога
    # Начало последовательности
    serialized_prompt.append(unified_tags['bos'])
    first_user_processed = False
    for message in dialog_messages:
        role = message['role']
        content = message['content']
        # 3.1. Обработка первого USER-сообщения (включая SYSTEM)
        if role == 'user' and not first_user_processed:
            serialized_prompt.append(unified_tags['user_start'])
            # Вставка системного промпта
            if system_prompt and unified_tags.get('sys_start'):
                serialized_prompt.append(unified_tags['sys_start'])
                serialized_prompt.append(system_prompt)
                serialized_prompt.append(unified_tags['sys_end'])
            serialized_prompt.append(content)
            serialized_prompt.append(unified_tags['user_end'])
            first_user_processed = True
        # 3.2. Обработка остальных витков
        elif role == 'assistant' and first_user_processed:
            # Ответ ассистента
            serialized_prompt.append(unified_tags.get('assist_start', ''))
            serialized_prompt.append(content)
            serialized_prompt.append(unified_tags.get('assist_end', ''))
            serialized_prompt.append(unified_tags.get('eos', '')) # Закрывает виток
        elif role == 'user' and first_user_processed:
            # Новый виток пользователя
            serialized_prompt.append(unified_tags.get('bos', ''))
            serialized_prompt.append(unified_tags['user_start'])
            serialized_prompt.append(content)
            serialized_prompt.append(unified_tags['user_end'])
        elif role == 'tool_call' and unified_tags.get('tool_call_start'):
            # Вызов функции, сгенерированный моделью
            serialized_prompt.append(unified_tags['tool_call_start'])
            serialized_prompt.append(content)
            serialized_prompt.append(unified_tags['tool_call_end'])
            serialized_prompt.append(unified_tags.get('eos', ''))
        elif (role == 'tool' or role == 'function') and unified_tags.get('tool_result_start'):
            # Результат выполнения функции
            serialized_prompt.append(unified_tags['tool_result_start'])
            serialized_prompt.append(content)
            serialized_prompt.append(unified_tags['tool_result_end'])
            serialized_prompt.append(unified_tags.get('eos', ''))
    return "".join(serialized_prompt)

def remove_commands_roles(cleaned_text): # TODO: перепроверь работоспособность, учти отступы и что-то напоминающее команды
    if not filter_generations: return cleaned_text
    # Проверяем все переменные по порядку
    for var_content in clean_variables_content:
        if var_content:
            # Ищем начало содержимого переменной в тексте
            start_pos = cleaned_text.find(var_content)
            if start_pos != -1:
                # Нашли содержимое - удаляем всё начиная с этой позиции
                cleaned_text = cleaned_text[:start_pos]
                break # Прерываем после первого найденного
    # Теперь ищем маркеры команд с помощью extract_command_markers
    # Для этого нам нужен словарь команд - используем пустой словарь
    markers = extract_command_markers(cleaned_text, {})
    # Если найдено более одного маркера, обрезаем текст перед вторым маркером
    if len(markers) >= 2:
        second_marker_start = markers[1]['start']
        cleaned_text = cleaned_text[:second_marker_start]
    return cleaned_text

def split_text_with_cutting(text, min_chunk_percentage=0.8):
    if not isinstance(text, str) or not text.strip(): return None
    delimiters = ['\n\n', '\n', '. ', '! ', '? ', '; ', ': ', ' ', '']
    chunks = []
    current_pos = 0
    text_length = len(text)
    while current_pos < text_length:
        end_pos = min(current_pos + chunk_size, text_length)
        if end_pos == text_length:
            chunk = text[current_pos:end_pos]
            if chunk.strip(): chunks.append(chunk)
            break
        split_pos = None
        for delimiter in delimiters:
            candidate_pos = text.rfind(delimiter, current_pos, end_pos)
            if candidate_pos != -1:
                current_chunk_size = candidate_pos + len(delimiter) - current_pos
                if current_chunk_size >= chunk_size * min_chunk_percentage:
                    split_pos = candidate_pos + len(delimiter)
                    break
        if split_pos is None: split_pos = end_pos
        chunk = text[current_pos:split_pos]
        if chunk.strip(): chunks.append(chunk)
        current_pos = split_pos
    return chunks if chunks else None

def text_cutter(text):
    """
    Обрабатывает текст, разбивая его на части и суммируя их итеративно,
    чтобы избежать переполнения контекста модели.
    """
    let_log('ИТЕРАТИВНЫЙ КАТТЕР ВЫЗВАН')
    let_log(text)
    # Список для хранения частей, ожидающих обработки
    chunks_to_process = [text]
    # Список для хранения уже обработанных, суммированных частей
    summarized_chunks = []
    # Цикл работает, пока есть части для обработки
    while chunks_to_process:
        # Берем первый кусок из "очереди"
        current_chunk = chunks_to_process.pop(0)
        try:
            # Пытаемся суммировать текущий кусок
            summarized_part = ask_model(current_chunk, system_prompt=summarize_prompt)
            summarized_chunks.append(summarized_part)
            traceprint()
        except RuntimeError as e:
            if 'ContextOverflowError' in str(e):
                let_log(f'ошибка каттера (переполнение), делим кусок: {len(current_chunk)=}')
                let_log(e)
                # Логика разделения текста, как в оригинальной функции
                text2 = current_chunk[len(current_chunk) // 2:]
                try:
                    # Ищем "чистую" точку для разделения
                    split_pos = min(text2.find('\n'), text2.find('. '))
                    if split_pos == -1: split_pos = text2.find(' ')
                    if split_pos != -1: text2 = text2[split_pos + 2:] # +2 для \n или ". "
                except Exception as split_e: # Используем более общий Exception для отлова ошибок find
                    traceprint()
                    let_log(f"Ошибка при поиске точки разделения: {split_e}")
                    # В случае ошибки просто продолжаем с половиной текста
                    pass
                text1 = current_chunk[:current_chunk.find(text2)]
                if text2: chunks_to_process.insert(0, text2)
                if text1: chunks_to_process.insert(0, text1)
            else: sys.exit(1)
        except Exception as e:
            # Любая другая непредвиденная ошибка, останавливаемся
            traceprint()
            print(e)
            sys.exit(1)
    return ' '.join(summarized_chunks)

def load_info_loaders(info_loaders_names):
    """
    Загружает только рабочие обработчики файлов.
    Если обработчик не найден или неисправен - не добавляем его в словарь.
    Заполняет глобальный словарь input_info_loaders на основе переданного словаря.
    Если имя функции совпадает с дефолтным, то используется соответствующий дефолтный обработчик.
    Если имя другое, то функция ищется в модуле info_loaders, А ДОЖНА НЕ ТАМ.
    """
    global input_info_loaders
    input_info_loaders = {}
    # Пытаемся импортировать модуль info_loaders
    global info_loaders
    try:
        import info_loaders
        let_log("✅ Модуль info_loaders успешно импортирован")
    except Exception as e:
        let_log(f"❌ ФАТАЛЬНО: Не удалось импортировать модуль info_loaders: {e}")
        return input_info_loaders
    # Загружаем только рабочие обработчики
    for ext, func_name in info_loaders_names.items():
        try:
            # Получаем обработчик из модуля
            handler = getattr(info_loaders, func_name)
            # Проверяем что это функция/метод
            if not callable(handler):
                let_log(f"⚠ '{func_name}' для .{ext} не является вызываемым объектом. Пропускаем.")
                continue
            # Проверяем сигнатуру (должен принимать хотя бы 1 аргумент)
            sig_params = list(inspect.signature(handler).parameters.values())
            if len(sig_params) < 1:
                let_log(f"⚠ '{func_name}' имеет неверную сигнатуру для .{ext}. Пропускаем.")
                continue
            input_info_loaders[ext] = handler
            let_log(f"✅ Загружен обработчик для .{ext}: {func_name}")
        except AttributeError: let_log(f"✗ Обработчик '{func_name}' не найден в модуле info_loaders. Пропускаем .{ext}")
        except Exception as e: let_log(f"✗ Ошибка при загрузке обработчика '{func_name}' для .{ext}: {e}. Пропускаем.")
    if not input_info_loaders: let_log("⚠ ВНИМАНИЕ: Не загружено ни одного обработчика файлов!")
    else: let_log(f"Итог: загружено {len(input_info_loaders)} обработчиков")
    return input_info_loaders

def upload_user_data(files_list):
    """
    Обрабатывает файлы, пропуская те, для которых нет обработчика или возникла ошибка.
    Возвращает только успешно обработанные результаты.
    """
    all_results = []  # Только успешные результаты
    if not files_list:
        let_log("Список файлов для загрузки пуст")
        return all_results
    # Проверяем, загружены ли обработчики
    if not input_info_loaders:
        let_log("⚠ Обработчики файлов не загружены. Пропускаем загрузку файлов.")
        send_output_message(text="Система обработки файлов недоступна", command='warning')
        return all_results
    for filename in files_list:
        file_basename = os.path.basename(filename)
        let_log(f"Обработка файла: {file_basename}")
        try:
            # 1. Проверка существования файла
            if not os.path.exists(filename): raise FileNotFoundError(f"Файл не существует: {filename}")
            # 2. Получение расширения
            _, file_extension = os.path.splitext(filename)
            extension = file_extension[1:].lower() if file_extension else ''
            if not extension: raise ValueError(f"Файл не имеет расширения: {filename}")
            # 3. Проверка наличия обработчика
            if extension not in input_info_loaders: raise KeyError(f"Нет обработчика для расширения .{extension}")
            handler = input_info_loaders[extension]
            # 5. Использование кэша (если включено)
            result = read_cache()
            if result == [False]:
                # Вызов обработчика
                let_log(f"  Вызов обработчика {handler.__name__}...")
                result_content = handler(filename, input_info_loaders)
                write_cache(result_content)
            else:
                let_log(f"  Используется кэшированный результат")
                result_content = result[1]
            # 6. Успешная обработка
            if file_extension[1:].lower() == 'zip' and isinstance(result_content, list):
                # Обработка ZIP-архива
                for file_data in result_content:
                    if file_data['type'] in ['file', 'unsupported']:
                        content = file_data['content']
                        if isinstance(content, str):
                            let_log(f"  Разбиение файла из ZIP: {file_data['filename']}")
                            chunks = split_text_with_cutting(content)
                            if not chunks:
                                continue
                            for t, chunk in enumerate(chunks):
                                set_common_save_id()
                                coll_exec(
                                    action="add",
                                    coll_name="user_collection",
                                    ids=[get_common_save_id()],
                                    embeddings=[get_embs(chunk)],
                                    metadatas=[{
                                        'name': f"{filename}/{file_data['filename']}",
                                        'part': t + 1,
                                        'source': 'zip'
                                    }],
                                    documents=[chunk]
                                )
            else:
                # Обработка обычного файла
                if isinstance(result_content, str):
                    let_log(f"  Разбиение файла на чанки...")
                    chunks = split_text_with_cutting(result_content)
                    if chunks:
                        for t, chunk in enumerate(chunks):
                            set_common_save_id()
                            coll_exec(
                                action="add",
                                coll_name="user_collection",
                                ids=[get_common_save_id()],
                                embeddings=[get_embs(chunk)],
                                metadatas=[{
                                    'name': filename,
                                    'part': t + 1,
                                    'source': 'file'
                                }],
                                documents=[chunk]
                            )
            # Добавляем в список успешных
            all_results.append({
                'filename': filename,
                'content': result_content,
                'extension': extension,
                'size': file_size
            })
            let_log(f"✅ Файл {file_basename} успешно обработан")
        except (FileNotFoundError, ValueError, KeyError) as e:
            # Ожидаемые ошибки - пропускаем файл
            let_log(f"⏭ Пропускаем {file_basename}: {e}")
            send_output_message(text=f"Пропущен файл {file_basename}: {e}", command='info')
        except Exception as e:
            # Неожиданные ошибки - пропускаем с логированием
            error_msg = f"Ошибка обработки {file_basename}: {type(e).__name__}"
            let_log(f"⏭ Пропускаем {file_basename} из-за ошибки: {error_msg}")
            let_log(f"  Детали: {str(e)[:200]}")
            send_output_message(text=f"Ошибка при обработке {file_basename}", command='warning')
    # Аннотация только успешно обработанных файлов
    if all_results:
        try:
            annotation_text = ""
            for result in all_results:
                if isinstance(result['content'], str):
                    annotation_text += f"\n\n--- {os.path.basename(result['filename'])} ---\n{result['content'][:5000]}"
                elif isinstance(result['content'], list):
                    for file_data in result['content']:
                        if isinstance(file_data.get('content'), str):
                            annotation_text += f"\n\n--- {os.path.basename(result['filename'])}/{file_data['filename']} ---\n{file_data['content'][:5000]}"
            if annotation_text.strip():
                try:
                    global_state.summ_attach = annotation_available_prompt + ask_model(
                        annotation_text, 
                        system_prompt=summarize_text_some_phrases
                    )
                except:
                    try:
                        global_state.summ_attach = annotation_available_prompt + ask_model(
                            text_cutter(annotation_text), 
                            system_prompt=summarize_text_some_phrases
                        )
                    except: global_state.summ_attach = annotation_failed_text
            else: global_state.summ_attach = annotation_failed_text
        except Exception as e:
            let_log(f"Ошибка при создании аннотации: {e}")
            global_state.summ_attach = annotation_failed_text
    else:
        let_log("Нет успешно обработанных файлов для аннотации")
        global_state.summ_attach = annotation_failed_text
    # Очистка ресурсов
    try:
        if hasattr(info_loaders, 'cleanup_image_models'): info_loaders.cleanup_image_models()
    except Exception as e: let_log(f"⚠ Ошибка при очистке ресурсов: {e}")
    return all_results

def set_common_save_id(): global_state.common_save_id += 1

def get_common_save_id(): return str(global_state.common_save_id)

def reset_common_save_id(): global_state.common_save_id = 1

def down_hierarchy():
    """Добавить новый уровень иерархии (делегирование)"""
    parts = global_state.now_try.strip('/').split('/')
    # Определяем номер нового уровня
    if not parts or parts[0] == '': new_level = 1
    else:
        # Находим последнюю пару и увеличиваем уровень
        last_part = parts[-1]
        if ':' in last_part:
            # Получаем часть до двоеточия
            level_part = last_part.split(':')[0]
            # Если часть перед двоеточием пустая, уровень = 0
            if level_part == '': last_level = 0
            else:
                try: last_level = int(level_part)
                except ValueError: last_level = 0
        else:
            # Если двоеточия нет, пробуем преобразовать всю часть в число
            try: last_level = int(last_part) if last_part.isdigit() else 0
            except ValueError: last_level = 0
        new_level = last_level + 1
    # Добавляем новый уровень с исполнителем=0
    if global_state.now_try == '/' or global_state.now_try == '': global_state.now_try = f"/{new_level}:0"
    else: global_state.now_try += f"/{new_level}:0"
    let_log(f"[HIERARCHY] Down: {global_state.now_try}")

def up_hierarchy():
    """Подняться на уровень выше"""
    if global_state.now_try == '/' or global_state.now_try == '': return  # Уже на корневом уровне
    parts = global_state.now_try.strip('/').split('/')
    if len(parts) > 1:
        # Удаляем последнюю пару
        parts = parts[:-1]
        if parts: global_state.now_try = '/' + '/'.join(parts)
        else: global_state.now_try = '/'
    else: global_state.now_try = '/'
    let_log(f"[HIERARCHY] Up: {global_state.now_try}")

def next_executor():
    """Создать/пересоздать исполнителя на текущем уровне"""
    if global_state.now_try == '/' or global_state.now_try == '': global_state.now_try = f"/1:1"
    else:
        parts = global_state.now_try.strip('/').split('/')
        last_part = parts[-1]
        if ':' in last_part:
            level, executor = last_part.split(':')
            new_executor = int(executor) + 1
            parts[-1] = f"{level}:{new_executor}"
        else:
            # Формат без : (старый формат)
            level = int(last_part) if last_part.isdigit() else 1
            parts[-1] = f"{level}:1"
        global_state.now_try = '/' + '/'.join(parts)
    let_log(f"[HIERARCHY] Next executor: {global_state.now_try}")
    return get_executor_number()

def get_executor_number():
    """Получить номер текущего исполнителя"""
    if global_state.now_try == '/' or global_state.now_try == '': return 0
    parts = global_state.now_try.strip('/').split('/')
    last_part = parts[-1]
    if ':' in last_part:
        _, executor = last_part.split(':')
        try: return int(executor) if executor != '' else 0
        except ValueError: return 0
    return 0

def get_level():
    """Получить номер текущего уровня"""
    if global_state.now_try == '/' or global_state.now_try == '': return 1
    parts = global_state.now_try.strip('/').split('/')
    last_part = parts[-1]
    if ':' in last_part:
        level, _ = last_part.split(':')
        try: return int(level) if level != '' else 1
        except ValueError: return 1
    elif last_part.isdigit():
        try: return int(last_part)
        except ValueError: return 1
    return 1

def get_operator_id():
    """ID оператора (без номера исполнителя)"""
    if global_state.now_try == '/' or global_state.now_try == '': return '/'
    parts = global_state.now_try.strip('/').split('/')
    # Преобразуем каждый уровень в формат без исполнителя
    clean_parts = []
    for part in parts:
        if ':' in part:
            level, _ = part.split(':')
            clean_parts.append(level)
        else: clean_parts.append(part)
    return '/' + '/'.join(clean_parts)

def get_executor_id():
    """Полный ID исполнителя"""
    if get_executor_number() == 0: return None  # Исполнитель не создан
    return global_state.now_try

def save_emb_dialog(tag, dialog_type='operator', result_text='', result=False):
    """
    Сохраняет диалог с новой системой ID
    dialog_type: 'operator' или 'executor'
    overwrite: True - перезаписать существующие записи, False - добавить новые
    """
    let_log(f"\n{'='*60}")
    
    let_log(f"Current ID: {global_state.now_try}")
    def _parse_dialog_to_messages(t):
        """Парсит текст диалога на отдельные сообщения."""
        messages_list = []
        # Используем глобальные переменные из cross_gpt.py
        roles_to_find = [
            ('operator', operator_role_text),
            ('worker', worker_role_text),
            ('function', func_role_text)
        ]
        positions = []
        for role_type, role_marker in roles_to_find:
            start_idx = 0
            while True:
                pos = t.find(role_marker, start_idx)
                if pos == -1: break
                positions.append((pos, role_marker, role_type))
                start_idx = pos + len(role_marker)
        if not positions: return []
        positions.sort(key=lambda x: x[0])
        for i, (pos, role_marker, role_type) in enumerate(positions):
            start_pos = pos
            end_pos = len(t)
            for j in range(i + 1, len(positions)):
                next_pos, _, _ = positions[j]
                if next_pos > start_pos:
                    end_pos = next_pos
                    break
            message_full_text = t[start_pos:end_pos]
            messages_list.append({'role': role_type, 'content': message_full_text})
        let_log(f"Распарсено {len(messages_list)} сообщений")
        return messages_list
    def _get_content_without_role(message_full_text):
        """Извлекает контент, удаляя маркер роли."""
        for role_marker in [operator_role_text, worker_role_text, func_role_text]:
            if message_full_text.startswith(role_marker):
                content = message_full_text.replace(role_marker, '', 1)
                return content 
        return message_full_text 
    def _create_numbered_messages_text(msgs):
        """Создает нумерованный текст БЕЗ ролей для LLM."""
        numbered_text = ""
        for i, msg in enumerate(msgs, 1):
            content_only = _get_content_without_role(msg['content'])
            if len(content_only) > 300: message_preview = content_only[:300] + "..."
            else: message_preview = content_only
            numbered_text += f"\n{i}. {message_preview}"
        return numbered_text
    def _create_grouping_prompt(numbered_messages_text, total_messages): return grouping_prompt_1 + numbered_messages_text + grouping_prompt_2
    def _parse_ranges_from_response(response):
        """Парсит ответ модели в список диапазонов."""
        cleaned = re.sub(r'[^\d,\-]', '', response)
        ranges = []
        for part in cleaned.split(','):
            if not part: continue
            if '-' in part:
                try:
                    start, end = map(int, part.split('-'))
                    if 1 <= start <= end: ranges.append((start, end))
                except ValueError: continue
            else:
                try:
                    num = int(part)
                    if 1 <= num: ranges.append((num, num))
                except ValueError: continue
        ranges.sort(key=lambda x: x[0])
        return ranges
    def _create_groups_from_ranges(msgs, ranges, offset=0):
        """Создает группы, очищенные от ролей и объединенные."""
        groups = []
        for start, end in ranges:
            if start < 1 or end > len(msgs) or start > end: continue
            cleaned_messages = []
            for i in range(start - 1, end):
                cleaned_messages.append(_get_content_without_role(msgs[i]['content']))
            group_text = "\n".join(cleaned_messages)
            groups.append({
                'global_start': offset + start,
                'global_end': offset + end,
                'text': group_text
            })
        # Fallback: если LLM не сгруппировал все сообщения, то одиночные сообщения тоже сохраняем
        if not groups:
            for i, msg in enumerate(msgs, 1):
                groups.append({
                    'global_start': offset + i,
                    'global_end': offset + i,
                    'text': _get_content_without_role(msg['content'])
                })
        return groups
    def _calculate_batch_size(msgs, start_index):
        """Рассчитывает оптимальный размер батча."""
        max_batch_size = 20
        total_messages = len(msgs)
        token_limit = get_token_limit()
        for batch_size in range(max_batch_size, 0, -1):
            end_index = min(start_index + batch_size, total_messages)
            batch_messages = msgs[start_index:end_index]
            numbered_text = _create_numbered_messages_text(batch_messages)
            prompt = _create_grouping_prompt(numbered_text, len(batch_messages))
            estimated_tokens = len(prompt) * get_text_tokens_coefficient()
            if estimated_tokens <= (token_limit - 1000): return batch_size
        return 1
    def _process_messages_batch(msgs, batch_offset):
        """Обрабатывает один батч сообщений."""
        let_log(f"  Внутри батча (смещение: {batch_offset}): Создание промпта и вызов ask_model...")
        numbered_text = _create_numbered_messages_text(msgs)
        prompt = _create_grouping_prompt(numbered_text, len(msgs))
        try:
            # Используем ask_model из cross_gpt.py с all_user=True
            response = ask_model(prompt, all_user=True)
            ranges = _parse_ranges_from_response(response)
            return _create_groups_from_ranges(msgs, ranges, batch_offset)
        except Exception as e:
            let_log(f"  Ошибка при обработке батча: {e}")
            # Пустые ranges - сохранит все одиночно
            return _create_groups_from_ranges(msgs, [], batch_offset)
    # --- 1. Определяем doc_id в зависимости от типа ---
    doc_id = None
    need_ov = False
    if dialog_type == 'operator':
        doc_id = get_operator_id()
        let_log(f"Operator ID: {doc_id}")
        minus_convs = 0
        if global_state.conversations % 2 == 0: minus_convs = 1
        _, history_to_save = get_chat_context(global_state.conversations - minus_convs)
        if not result and global_state.need_owerwrite_operator:
            need_ov = True
            global_state.need_owerwrite_operator = False
    elif dialog_type == 'executor':
        executor_id = get_executor_id()
        if not executor_id:
            let_log("Executor not created yet, skipping save")
            return
        doc_id = executor_id
        let_log(f"Executor ID: {doc_id}")
        _, history_to_save = get_chat_context(global_state.conversations)
        if global_state.need_owerwrite_executor:
            need_ov = True
            global_state.need_owerwrite_executor = False
    # --- 2. Если нужно перезаписать - удаляем старые записи ---
    let_log(f"SAVE_EMB_DIALOG: tag={tag}, type={dialog_type}, need_ov={need_ov}")
    if need_ov:
        let_log('\nПЕРЕЗАПИСЬ ЧАТА\n')
        existing_ids = coll_exec(
            action="query",
            coll_name="milana_collection",
            query_embeddings=[[]],
            filters={
                "doc_id": doc_id,
                "dialog_type": dialog_type,
                "result": result
            },
            fetch="ids",
            n_results=100,
            flatten=True
        )
        if existing_ids:
            coll_exec(
                action="delete",
                coll_name="milana_collection",
                ids=existing_ids
            )
            let_log(f"Deleted {len(existing_ids)} old records for {doc_id}")
    else: let_log('\nНЕТ ПЕРЕЗАПИСИ ЧАТА\n')
    if result:
        if result_text == '':
            let_log("Текст результата пустой")
            return
        let_log("Сохранение результата отдельно")
        set_common_save_id()
        metadata = {
            "doc_id": doc_id,
            "dialog_type": dialog_type,
            "done": tag,
            "result": result,
            "hierarchy": global_state.now_try,
            "timestamp": time.time()
        }
        coll_exec(
            action="add",
            coll_name="milana_collection",
            ids=[get_common_save_id()],
            embeddings=[get_embs(result_text)],
            metadatas=[metadata],
            documents=[result_text]
        )
        return
    # --- 4. Парсинг и группировка диалога ---
    messages = _parse_dialog_to_messages(history_to_save)
    if not messages:
        let_log("Не удалось распарсить сообщения.")
        return
    all_groups = []
    total_messages = len(messages)
    current_index = 0
    chunks_info = []
    # Проверяем, помещаются ли все сообщения в один запрос
    test_prompt = _create_grouping_prompt(_create_numbered_messages_text(messages), total_messages)
    estimated_tokens = len(test_prompt) * get_text_tokens_coefficient()
    if estimated_tokens <= (get_token_limit() - 1000):
        let_log("Все сообщения помещаются в один запрос")
        all_groups = _process_messages_batch(messages, 0)
    else:
        let_log(f"Сообщения не помещаются (оценка: {estimated_tokens:.0f} токенов), разбиваем на батчи")
        batch_number = 0
        while current_index < total_messages:
            batch_size = _calculate_batch_size(messages, current_index)
            batch_messages = messages[current_index:current_index + batch_size]
            chunks_info.append({
                'batch_number': batch_number + 1, 
                'start_idx': current_index + 1, 
                'end_idx': min(current_index + batch_size, total_messages)
            })
            let_log(f"\n--- Обработка батча {batch_number + 1} (сообщения {current_index + 1}-{current_index + batch_size}) ---")
            batch_groups = _process_messages_batch(batch_messages, current_index)
            all_groups.extend(batch_groups)
            current_index += batch_size
            batch_number += 1
        if chunks_info:
            let_log(f"--- ИНФОРМАЦИЯ О ЧАНКАХ (для LLM) ---\n" + 
                   "\n".join([f"Чанк {c['batch_number']}: сообщения {c['start_idx']}-{c['end_idx']}" 
                             for c in chunks_info]))
    # --- 5. Сохранение всех групп ---
    let_log(f"\n--- Сохранение {len(all_groups)} групп для {dialog_type} ---")
    for i, group in enumerate(all_groups):
        set_common_save_id()
        metadata = {
            "doc_id": doc_id,
            "dialog_type": dialog_type,
            "done": tag,
            "result": result,
            "group_index": i,
            "total_groups": len(all_groups),
            "hierarchy": global_state.now_try,
            "timestamp": time.time()
        }
        let_log(f"\n### ГРУППА {i+1} (Сохраняемый документ {get_common_save_id()}) ###")
        let_log(f"Диапазон: {group['global_start']}-{group['global_end']}")
        let_log(f"СОХРАНЯЕМЫЙ ТЕКСТ (первые 500 символов):\n---START---\n{group['text'][:500]}...\n---END---")
        coll_exec(
            action="add", 
            coll_name="milana_collection", 
            ids=[get_common_save_id()],
            embeddings=[get_embs(group['text'])], 
            metadatas=[metadata], 
            documents=[group['text']]
        )
    let_log(f"Всего сохранено {len(all_groups)} групп сообщений для {doc_id}")
    let_log(f"{'='*60}")

def gigo(base_task, roles=[]):
    gigo_now_questions = gigo_questions
    try: questions = ask_model(gigo_now_questions + base_task + global_state.summ_attach, all_user=True)
    except:
        base_task = text_cutter(base_task)
        questions = ask_model(gigo_now_questions + base_task + text_cutter(global_state.summ_attach), all_user=True)
    additional_info = librarian(questions)
    if additional_info != found_info_1: additional_info = gigo_found_info + additional_info
    else: additional_info = gigo_not_found_info; let_log(found_info_1)
    minds = ''
    if not roles: roles = [gigo_dreamer, gigo_realist, gigo_critic]
    for role in roles:
        try: minds += '\n' + ask_model(gigo_role_answer_1 + role + gigo_role_answer_2 + base_task + '\n' + additional_info, all_user=True)
        except: minds += '\n' + ask_model(gigo_role_answer_1 + role + gigo_role_answer_2 + base_task + '\n' + text_cutter(additional_info), all_user=True)
    try: plan = ask_model(gigo_make_plan + base_task + '\n' + gigo_reaction + minds + '\n' + additional_info, all_user=True)
    except: plan = ask_model(gigo_make_plan + base_task + text_cutter(minds) + '\n' + text_cutter(additional_info), all_user=True)
    return gigo_return_1 + base_task + '\n' + gigo_return_2 + plan

def critic(task: str, result: str) -> int | str:
    # TODO: может обработку текста и результат сюда засунуть?
    """
    Оценивает результат.
    - Возвращает 1, если результат приемлем, критик не уверен или произошла ошибка.
    - Возвращает строку с новой, доработанной задачей для исполнителя.
    """
    # TODO: добавь текст каттер
    if global_state.conversations % 2 == 0:
        if global_state.conversations != 0: num_critic_reaction = 1
        else: num_critic_reaction = global_state.conversations - 1
    else: num_critic_reaction = global_state.conversations
    try:
        now_critic_reactions = global_state.critic_reactions[num_critic_reaction]
        let_log(f"Текущие реакции для диалога {num_critic_reaction}: {now_critic_reactions}")
        let_log(f"Реакций ({now_critic_reactions}), максимум ({global_state.max_critic_reactions}), запускаем критика")
    except:
        now_critic_reactions = 0
        global_state.critic_reactions[num_critic_reaction] = 0
        let_log(f"Нет реакций для диалога {num_critic_reaction}, установлено 0")
    if now_critic_reactions == global_state.max_critic_reactions:
        let_log(f"Достигнут максимум реакций ({now_critic_reactions}), пропускаем критика")
        del global_state.critic_reactions[num_critic_reaction]
        return 1
    # --- Этап 1 ---
    try:
        prompt_stage1 = f"{prompt_decomposition_1}\n{task}\n{prompt_decomposition_2}"
        criteria_text = ask_model(prompt_stage1, all_user=True)
    except:
        task = text_cutter(task)
        prompt_stage1 = f"{prompt_decomposition_1}\n{task}\n{prompt_decomposition_2}"
        criteria_text = ask_model(prompt_stage1, all_user=True)
    if not criteria_text or not criteria_text.strip(): return 1
    # --- Этап 2 ---
    prompt_stage2 = (f"{prompt_evaluation_1}\n"
                     f"{prompt_evaluation_2} {task}\n"
                     f"{prompt_evaluation_3} {result}\n"
                     f"{prompt_evaluation_4} {criteria_text}\n"
                     f"{prompt_evaluation_5}")
    evaluation_text = ask_model(prompt_stage2, all_user=True)
    if not evaluation_text or not evaluation_text.strip(): return 1
    # --- Этап 2.5: Обращение к Библиотекарю (закомментировано) ---
    """
    # Создаем промпт для генерации вопросов к библиотекарю
    prompt_librarian_q = (f"{prompt_librarian_questions_1}\\n{task}\\n"
                          f"{prompt_librarian_questions_2}\\n{result}\\n"
                          f"{prompt_librarian_questions_3}\\n{evaluation_text}\\n"
                          f"{prompt_librarian_questions_4}")
    # Получаем от модели текст с вопросами (и возможным мусором)
    questions_text = ask_model(prompt_librarian_q)
    librarian_context = ""
    if questions_text and questions_text.strip():
        # Отправляем ВЕСЬ текст в библиотекаря, он сам отфильтрует нужное
        print("Критик -> Библиотекарь: Запрос на проверку информации...")
        librarian_answers = librarian(questions_text)
        # Если библиотекарь вернул какие-то ответы, формируем контекст для финального решения
        if librarian_answers and librarian_answers.strip():
            librarian_context = f"{prompt_decision_librarian_context}{librarian_answers}\\n"
    """
    # Временно устанавливаем пустой контекст, пока логика выше закомментирована
    librarian_context = ""
    # --- Этап 3 и 4 ---
    prompt_stage3 = (f"{prompt_decision_1}\n"
                     f"{prompt_decision_2} {task}\n"
                     f"{prompt_decision_3} {result}\n"
                     f"{prompt_decision_4} {evaluation_text}\n"
                     f"{librarian_context}" # <-- Сюда будет подставлен контекст от библиотекаря
                     f"{prompt_decision_5}")
    decision_response = ask_model(prompt_stage3, all_user=True)
    if marker_decision_revise in decision_response:
        try:
            start_index = decision_response.index(marker_new_task) + len(marker_new_task)
            new_task = decision_response[start_index:].strip()
            if new_task:
                print("Критик: Требуется доработка. Сформулирована новая задача.")
                global_state.critic_wants_retry = True
                global_state.critic_comment = new_task
                global_state.critic_reactions[num_critic_reaction] += 1
                return new_task
        except Exception as e:
            let_log(e)
            del global_state.critic_reactions[num_critic_reaction]
            return 1
    elif marker_decision_approve in decision_response:
        print("Критик: Задача выполнена успешно.")
        del global_state.critic_reactions[num_critic_reaction]
        return 3
    elif marker_decision_unsure in decision_response:
        print("Критик: Не уверен в результате, требуется проверка человеком.")
        del global_state.critic_reactions[num_critic_reaction]
        return 2
    return 1 # Если вердикт не распознан или ответ пустой

def find_all_commands(text: str, available_commands: list[str], cutoff: float = 0.75) -> list[str]:
    """
    Находит все уникальные команды из списка, которые нечетко соответствуют словам в тексте.
    Эта функция идеально подходит для выбора нескольких инструментов на основе
    описания задачи, сгенерированного моделью.
    Args:
        text (str): Входной текст для поиска (например, ответ модели).
        available_commands (list[str]): Список всех доступных имен команд/инструментов.
        cutoff (float): Порог схожести для difflib (от 0 до 1). Чем выше, тем строже соответствие.
    Returns: list[str]: Список уникальных имен команд, которые были найдены в тексте.
    """
    # Используем множество (set) для автоматического сбора только уникальных значений.
    found_commands_set = set()
    # Регулярное выражение для поиска "слов", которые могут быть именами команд
    # (буквы, цифры и знак подчеркивания).
    word_pattern = re.compile(r'[\w_]+')
    # Итерируемся по каждому слову, найденному в тексте.
    for match_obj in word_pattern.finditer(text):
        word_from_text = match_obj.group(0)
        # Используем difflib, чтобы найти наилучшее совпадение для этого конкретного слова.
        close_matches = difflib.get_close_matches(
            word_from_text,
            available_commands,
            n=1,  # Ищем только одно, самое лучшее, совпадение для данного слова
            cutoff=cutoff
        )
        if close_matches:
            # Если для слова найдено достаточно близкое совпадение,
            # добавляем соответствующую команду из списка `available_commands` в наше множество.
            matched_command = close_matches[0]
            found_commands_set.add(matched_command)
    # Преобразуем множество обратно в список перед возвратом.
    return list(found_commands_set)

def find_and_match_command(text, commands_dict):
    """
    Ищет маркер !!!name!!! в тексте и пытается сопоставить имя с commands_dict.
    Возвращает (found_key, content_str) или None.
    commands_dict ожидается в формате: {'name': ('description', func), ...}
    Поиск: сначала простое in / substring, потом fuzzy через difflib.get_close_matches.
    """
    if not text: return None
    # Попытка найти маркер с разным количеством восклицательных знаков и возможными ошибками
    # Ищем в первых 5 символах начало маркера (2-4 восклицательных знака или их вариации)
    # Сначала проверяем первые 5 символов на наличие возможных восклицательных знаков
    first_5 = text[:5] if len(text) >= 5 else text
    # Набор символов, которые могут быть восклицательными знаками (с ошибками)
    exclamation_chars = {'!', '¡', '|', '1', 'i', 'l', 'I'}  # добавляем похожие символы
    # Ищем позицию первого символа, который может быть восклицательным знаком
    first = -1
    for i, char in enumerate(first_5):
        if char in exclamation_chars:
            first = i
            break
    # Если не нашли восклицательных знаков в первых 5 символах
    if first == -1 or first > 4: return None
    # Теперь ищем завершающий маркер после первого
    # Ищем последовательность из 1-4 символов, которые могут быть восклицательными знаками
    marker_start = first
    marker_end = marker_start
    # Определяем длину открывающего маркера (1-4 символа)
    while marker_end < len(text) and marker_end - marker_start < 4:
        if text[marker_end] in exclamation_chars: marker_end += 1
        else: break
    # Если маркер слишком короткий (меньше 1 символа)
    if marker_end - marker_start < 1: return None
    # Теперь ищем закрывающий маркер
    # Сначала пропускаем возможное имя команды (ищем следующий набор восклицательных знаков)
    search_start = marker_end
    second = -1
    second_end = -1
    # Ищем следующий набор символов, которые могут быть восклицательными знаками
    i = search_start
    while i < len(text):
        if text[i] in exclamation_chars:
            second = i
            # Определяем длину закрывающего маркера
            j = i
            while j < len(text) and j - i < 4:
                if text[j] in exclamation_chars:
                    j += 1
                else: break
            # Закрывающий маркер должен быть хотя бы из 1 символа
            if j - i >= 1:
                second_end = j
                break
            else:
                i += 1
                continue
        i += 1
    if second == -1: return None
    # Извлекаем имя команды и контент
    raw_name = text[marker_end:second].strip()
    content = text[second_end:].strip() if second_end < len(text) else ""
    # защититься, если имя пустое
    if not raw_name: return None
    # построить словарь ключей (строки)
    key_map = {}
    try:
        if isinstance(commands_dict, dict):
            for k, v in commands_dict.items(): key_map[str(k).strip()] = v
        else:
            # если передан список/итерируемый
            for item in commands_dict:
                try:
                    k = str(item[0]).strip()
                    key_map[k] = item[1] if len(item) > 1 else item
                except: continue
    except: return None
    # попытка простого поиска
    found_key = None
    for k in key_map:
        if raw_name in k or k in raw_name:
            found_key = k
            break
    if not found_key and key_map:
        try:
            matches = difflib.get_close_matches(
                raw_name, 
                list(key_map.keys()), 
                n=1, 
                cutoff=0.7
            )
            if matches: found_key = matches[0]
        except: found_key = None
    if not found_key: return None
    return (found_key, content)

def tools_selector(text, sid):
    """
    Вызывает инструменты, используя поиск маркеров и словарь команд в global_state.tools_commands_dict[sid].
    Возвращает результат выполнения команды или None.
    """
    let_log("=== [TOOLS_SELECTOR ЗАПУЩЕН (НОВАЯ ВЕРСИЯ)] ===")
    let_log(f"[TOOLS_SELECTOR] входной текст (начало 200):\n{text[:200]}")
    # 1) получить кэш
    cached = read_cache()
    let_log(f"[TOOLS_SELECTOR] кэш: {cached}")
    if cached == [True, [False, False]]: return None
    # 2) получить словарь команд для сессии
    try: now_commands = global_state.tools_commands_dict.get(sid, {})
    except Exception: now_commands = {}
    # 3) system keys
    try: sys_keys = [str(k) for k in global_state.system_tools_keys]
    except Exception: sys_keys = []
    # 4) найти маркер и сопоставить с командами
    match = find_and_match_command(text, now_commands)
    if not match:
        let_log("[TOOLS_SELECTOR] маркер не найден или команда не сопоставилась")
        let_log("=== [TOOLS_SELECTOR ЗАВЕРШЁН] ===")
        write_cache([False, False])
        return None
    found_key, content = match
    let_log(f"[TOOLS_SELECTOR] найден ключ: {found_key}, контент длиной: {len(content) if content else 0}")
    # 5) определить, системная ли команда
    is_system = False
    try:
        for sk in sys_keys:
            if found_key in sk or sk in found_key:
                is_system = True
                break
        if not is_system and sys_keys:
            close = difflib.get_close_matches(found_key, sys_keys, n=1, cutoff=0.7)
            if close: is_system = True
    except Exception: is_system = False
    let_log(f"[TOOLS_SELECTOR] команда системная? {is_system}")
    # 6) Обработка кэша в зависимости от типа команда
    if not is_system:
        # ТОЛЬКО для несистемных команд: проверяем кэш
        if cached != [False]:
            if cached[1] != ["SYSTEM", False]:
                let_log("[TOOLS_SELECTOR] Возвращаем не-системный результат из кэша")
                let_log("=== [TOOLS_SELECTOR ЗАВЕРШЁН] ===")
                return cached
    else:
        # Для системных команд: проверяем, не выполняем ли мы её уже (рекурсия)
        if cached != [False]:
            if cached[1] != ["SYSTEM", False]: raise RuntimeError('СБОЙ КЭШЕРА В ТУЛЗ СЕЛЕКТОРЕ')
        # Помечаем, что начинаем выполнение системной команды
        if cached == [False]: write_cache(["SYSTEM", False])
        traceprint()
    # 7) получить callable из now_commands
    try: entry = now_commands.get(found_key)
    except Exception: entry = None
    if entry == None:
        let_log("[TOOLS_SELECTOR] команда не найдена в словаре сессии (после сопоставления)")
        let_log("=== [TOOLS_SELECTOR ЗАВЕРШЁН] ===")
        write_cache([False, False])
        return None
    func_callable = None
    try:
        if isinstance(entry, tuple) or isinstance(entry, list):
            if len(entry) >= 2 and callable(entry[1]): func_callable = entry[1]
            elif len(entry) >= 3 and callable(entry[2]): func_callable = entry[2]
            elif callable(entry[0]): func_callable = entry[0]
        elif callable(entry): func_callable = entry
        else:
            try: func_callable = entry.get("func")
            except Exception: func_callable = None
    except Exception: func_callable = None
    if not func_callable:
        let_log("[TOOLS_SELECTOR] не удалось получить callable для команды")
        let_log("=== [TOOLS_SELECTOR ЗАВЕРШЁН] ===")
        write_cache([False, False])
        return None
    # 9) выполнить функцию
    let_log("[TOOLS_SELECTOR] Выполняем функцию...")
    if found_key == start_dialog_command_name: global_state.task_delegated = True
    #try: result = func_callable(content)
    #except Exception as e: result = "__TOOL_ERROR__: " + str(e)
    result = func_callable(content) # TODO:
    let_log(f"[TOOLS_SELECTOR] Результат (первые 500):\n{str(result)[:500]}")
    # 10) кэшировать результат если не системная команда
    try:
        if not is_system:
            let_log("[TOOLS_SELECTOR] Кэшируем результат (не системная команда)")
            write_cache(result)
            traceprint()
    except Exception: pass
    let_log("=== [TOOLS_SELECTOR ЗАВЕРШЁН] ===")
    return result

# ВЕРСИЯ ДЛЯ СТАНДАРТНОГО РЕЖИМА
def _standard_agent_func(text, agent_number):
    # надо сокращать ещё когда превышен не лимит а какое-то количество ибо модель может начать писать бред
    # нужно резать в первую очередь ответы инструментов ибо сторонние разработчики могут перегрузить модель
    global_state.stop_agent = False
    talk_prompt = text
    sid = global_state.conversations - agent_number
    global_state.now_agent_id = sid
    if agent_number: # 1 - Милана
        you = operator_role_text
        msg_from = worker_role_text
    else: # 0 - Иван
        you = worker_role_text
        if global_state.dialog_ended:
            msg_from = func_role_text
            global_state.dialog_ended = False
        else: msg_from = operator_role_text
    while not global_state.stop_agent:
        let_log(f"[DEBUG-STD] agent_number={agent_number}, sid={sid}")
        prompt, history = get_chat_context(sid)
        last_talk_prompt = talk_prompt
        try:
            full_prompt = prompt + history + msg_from + talk_prompt + you
            talk_prompt = ask_model(full_prompt)
        except Exception as e:
            let_log(f"Ошибка в _standard_agent_func: {e}")
            # Ваша логика text_cutter, если нужна
            history = start_dialog_history + text_cutter(history) + last_messages_marker
            full_prompt = prompt + history + msg_from + talk_prompt + you
            talk_prompt = ask_model(full_prompt)
        talk_prompt = remove_commands_roles(talk_prompt)
        # Сначала сообщение от предыдущего, потом ответ от текущего.
        update_history(sid, last_talk_prompt, msg_from)
        update_history(sid, talk_prompt, you)
        answer = tools_selector(talk_prompt, sid)
        if answer:
            talk_prompt = answer
            msg_from = func_role_text
        else: break
    global_state.stop_agent = False
    return talk_prompt

# ВЕРСИЯ ДЛЯ RAG РЕЖИМА
def _rag_agent_func(text, agent_number):
    global_state.stop_agent = False
    talk_prompt = text
    sid = global_state.conversations - agent_number
    global_state.now_agent_id = sid
    if agent_number: # 1 - Милана
        you = operator_role_text
        msg_from = worker_role_text
    else: # 0 - Иван
        you = worker_role_text
        if global_state.dialog_ended:
            msg_from = func_role_text
            global_state.dialog_ended = False
        else: msg_from = operator_role_text
    while not global_state.stop_agent:
        let_log(f"[DEBUG-RAG] agent_number={agent_number}, sid={sid}")
        # 1. Сохраняем входящее сообщение от предыдущего агента в RAG-историю
        update_history(sid, talk_prompt, msg_from)
        # 2. Вызываем RAG-конструктор. Он сам найдет системный промпт и всю историю.
        final_prompt_for_model, _ = get_chat_context(sid, talk_prompt)
        # 3. Вызываем модель, добавив роль текущего агента для корректной генерации
        try: talk_prompt = ask_model(final_prompt_for_model + you)
        except Exception as e:let_log(f"Ошибка в _rag_agent_func: {e}")
        # Здесь RAG уже должен был обработать длинный контекст
        talk_prompt = remove_commands_roles(talk_prompt)
        # 4. Сохраняем ответ самой модели в RAG-историю
        update_history(sid, talk_prompt, you)
        answer = tools_selector(talk_prompt, sid)
        if answer:
            let_log(global_state.stop_agent)
            talk_prompt = answer
            msg_from = func_role_text
        else: break
    global_state.stop_agent = False
    return talk_prompt

def get_user_feedback_and_update_task(current_task, dialog_result):
    while True: # очищаем очередь ввода
        if get_input_message() == None: break
    send_output_message(text=dialog_result, command='end')
    user_message = get_input_message(wait=True)
    if user_review_text1 in current_task or user_review_text2 in current_task or user_review_text3 in current_task or user_review_text4 in current_task:
        updated_task = current_task + user_review_text2 + dialog_result + user_review_text3 + user_message['text']
    else: updated_task = user_review_text1 + current_task + user_review_text2 + dialog_result + user_review_text3 + user_message['text']
    if not updated_task: raise # TODO: доделай тут или общение с пользователем или просто сообщение что текста нет
    if user_message['attachments']:
        send_output_message(text=start_load_attachments_text)
        if not input_info_loaders: load_info_loaders(default_handlers_names)
        upload_user_data(user_message['attachments'])
        send_output_message(text=end_load_attachments_text)
    return updated_task

def worker(really_main_task):
    while True:
        global_state.retries = []
        global_state.conversations = 0
        global_state.tools_commands_dict = {}
        global_state.dialog_state = True
        global_state.critic_wants_retry = False
        global_state.main_now_task = really_main_task
        global_state.gigo_web_search_allowed = False
        talk_prompt = start_dialog(global_state.main_now_task)
        global_state.gigo_web_search_allowed = True
        if not global_state.dialog_state:
            let_log('worker диалог завершился не начавшись')
            if global_state.critic_wants_retry:
                if user_review_text1 in really_main_task or user_review_text2 in really_main_task or user_review_text3 in really_main_task or user_review_text4 in really_main_task:
                    really_main_task = really_main_task + user_review_text2 + global_state.dialog_result + user_review_text4 + global_state.critic_comment
                else: really_main_task = user_review_text1 + really_main_task + user_review_text2 + global_state.dialog_result + user_review_text4 + global_state.critic_comment
            else: really_main_task = get_user_feedback_and_update_task(really_main_task, global_state.dialog_result)
            continue
        while True:
            if global_state.dialog_state:
                talk_prompt = agent_func(talk_prompt, 0) # ivan
                if global_state.task_delegated:
                    global_state.task_delegated = False # а вот тут чезанах
                    continue
            if global_state.dialog_state: talk_prompt = agent_func(talk_prompt, 1) # milana
            if not global_state.dialog_state:
                print(global_state.tools_commands_dict)
                let_log(global_state.dialog_result)
                let_log(global_state.conversations)
                # нужно еще задачу в хрому записать
                # TODO: вынеси эти 2 куска кода в функцию
                let_log(global_state.conversations)
                if global_state.conversations <= 0:
                    if global_state.critic_wants_retry: # TODO: в критике тоже добавь пояснения И ОН НЕ ВЕЗДЕ ДОЛЖЕН ИСПОЛЬЗОВАТЬ РИЛИ МАЙН ТАКС А ЕЩЁ НАДО УНИФИЦИРОВАТЬ ЕГО С КЛИЕНТСКИМИ ТЕКСТАМИ
                        if user_review_text1 in really_main_task or user_review_text2 in really_main_task or user_review_text3 in really_main_task or user_review_text4 in really_main_task:
                            really_main_task = really_main_task + user_review_text2 + global_state.dialog_result + user_review_text4 + global_state.critic_comment
                        else: really_main_task = user_review_text1 + really_main_task + user_review_text2 + global_state.dialog_result + user_review_text4 + global_state.critic_comment
                    else: really_main_task = get_user_feedback_and_update_task(really_main_task, global_state.dialog_result)
                    break
                else:
                    global_state.dialog_state = True
                    if global_state.critic_wants_retry:
                        if user_review_text1 in global_state.main_now_task or user_review_text2 in global_state.main_now_task or user_review_text3 in global_state.main_now_task or user_review_text4 in global_state.main_now_task:
                            global_state.main_now_task = global_state.main_now_task + user_review_text2 + global_state.dialog_result + user_review_text4 + global_state.critic_comment
                        else: global_state.main_now_task = user_review_text1 + global_state.main_now_task + user_review_text2 + global_state.dialog_result + user_review_text4 + global_state.critic_comment
                        talk_prompt = start_dialog(global_state.main_now_task) # TODO: тут тоже может быть внезапное завершение
                    else: talk_prompt = global_state.dialog_result

def initialize_work(base_dir, chat_id, input_queue, output_queue, stop_event, pause_event, log_queue):
    global memory_sql
    global actual_handlers_names, another_tools_files_addresses
    global token_limit, emb_token_limit, most_often
    global client
    global milana_collection
    global user_collection
    global rag_collection
    global ui_conn
    global cache_path
    global ask_provider_model, ask_provider_model_chat, get_provider_embs
    global language
    global chat_path
    global do_chat_construct
    global native_func_call
    global use_rag
    global agent_func
    global clean_variables_content
    global filter_generations
    global is_save_log
    ui_conn = [input_queue, output_queue, log_queue]
    # === Загружаем параметры чата ===
    chat_path = os.path.join(base_dir, "data", "chats", chat_id)  # ИСПРАВЛЕНО: используем base_dir
    cache_path = os.path.join(chat_path, "cache.db")
    # Обновляем пути для system_tools
    global folder_path, slash
    folder_path = base_dir  # Устанавливаем folder_path в base_dir для корректной загрузки system_tools
    # Обновляем sys.path для system_tools
    sys.path = [p for p in sys.path if not p.endswith(('system_tools', 'system_tools/milana', 'system_tools/ivan'))]  # Удаляем старые пути
    sys.path.append(os.path.join(folder_path, 'system_tools'))
    sys.path.append(os.path.join(folder_path, 'system_tools', 'milana'))
    sys.path.append(os.path.join(folder_path, 'system_tools', 'ivan'))
    let_log(f"Base directory: {base_dir}")
    let_log(f"Chat path: {chat_path}")
    let_log(f"Folder path: {folder_path}")
    # === Подготовка SQLite БД ===
    db_path = os.path.join(chat_path, "chatsettings.db")
    let_log(f"Database path: {db_path}")
    memory_sql = connect(db_path)
    sql_exec('''CREATE TABLE IF NOT EXISTS found_info (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        info TEXT NOT NULL
    )''')
    initial_text, fl = load_initial_data(chat_id)
    settings = load_chat_settings(chat_id)
    tool_paths = settings.get("another_tools", [])
    # === Настройки из settings ===
    token_limit = int(settings.get("token_limit", 8192))
    most_often = int(settings.get("frequent_response", 0))
    need_best_result = int(settings.get("best_response", 0))
    use_rag = int(settings.get("use_rag", 1))
    global_state.write_results = int(settings.get("write_results", 0))
    if int(settings.get("write_log", 1)) == 0: is_save_log = False
    # === Инициализация ChromaDB ===
    chroma_path = os.path.join(chat_path, "chroma_db")
    client = chromadb.PersistentClient(path=chroma_path, settings=Settings(allow_reset=True, anonymized_telemetry=False))
    client.reset()
    milana_collection = client.get_or_create_collection(name="milana_collection", metadata={"hnsw:space": "cosine"})
    user_collection = client.get_or_create_collection(name="user_collection", metadata={"hnsw:space": "cosine"})
    if use_rag == 1:
        use_rag = True
        agent_func = _rag_agent_func
        rag_collection = client.get_or_create_collection(name="rag_collection", metadata={"hnsw:space": "cosine"})
    else:
        use_rag = False
        agent_func = _standard_agent_func
    filter_generations = int(settings.get("filter_generations", 0))
    global_state.hierarchy_limit = int(settings.get("hierarchy_limit", 0)) * 2
    global initialize_schema, create_chat, get_chat_context, update_history, delete_chat
    from chat_manager import (
        initialize_schema,
        create_chat,
        get_chat_context,
        update_history,
        delete_chat
    )
    initialize_schema()
    default_tools_dir = os.path.join(base_dir, "default_tools")
    for rel_path in tool_paths:
        # Если в списке уже могут быть абсолютные пути, можно проверить:
        if os.path.isabs(rel_path): full_path = rel_path
        else: full_path = os.path.join(default_tools_dir, rel_path)
        full_path = os.path.normpath(full_path)
        another_tools_files_addresses.append(full_path)
    # === Инициализация модели ===
    model_type = settings.get("model_type", "ollama")
    language = settings.get("language", "ru")
    try:
        # Обновляем путь для импорта model_providers
        model_providers_path = os.path.join(base_dir, "model_providers")
        if model_providers_path not in sys.path: sys.path.append(model_providers_path)
        model_providers_module = importlib.import_module(f"model_providers.{model_type}")
        ask_model = model_providers_module.ask_model
        ask_model_chat = model_providers_module.ask_model_chat
        create_embeddings = model_providers_module.create_embeddings
        model_connect = model_providers_module.connect
        model_disconnect = model_providers_module.disconnect
        connect_params = settings.get("model_provider_params", "")
        # Подключение модели
        connection_result = model_connect(connect_params)
        if not connection_result or not connection_result[0]:
            let_log(f"Ошибка подключения модели: {connection_result[1] if len(connection_result) > 1 else 'Unknown error'}")
            return
        # Успешное подключение - получаем теги
        success, _, tags = connection_result
        if tags: 
            global unified_tags
            unified_tags = tags
        # Делаем функции модели глобальными
        globals().update({
            'ask_provider_model': ask_model,
            'ask_provider_model_chat': ask_model_chat,
            'get_provider_embs': create_embeddings,
        })
        provider_module_name = f"model_providers.{model_type}"
        provider_module = sys.modules.get(provider_module_name)
        setattr(provider_module, "token_limit", token_limit)
        emb_token_limit = provider_module.emb_token_limit
        do_chat_construct = provider_module.do_chat_construct
        native_func_call = provider_module.native_func_call
    except Exception as e:
        let_log(f"Ошибка инициализации модели: {str(e)}")
        traceback.print_exc()
        return
    # === Загрузка модели и инструментов ===
    globalize_language_packet(language)
    clean_variables_content = []
    if filter_generations == 1:
        clean_variables_content = [operator_role_text, worker_role_text, func_role_text, system_role_text]
        if unified_tags.get('bos') is not None: clean_variables_content.append(bos_tag)
        if unified_tags.get('user_start') is not None: clean_variables_content.append(user_start_tag)
        filter_generations = True
    else: filter_generations = False
    global_state.ivan_module_tools, global_state.milana_module_tools = system_tools_loader()
    # === Загружаем пользовательские модули ===
    let_log(f"\n=== ЗАГРУЗКА ПОЛЬЗОВАТЕЛЬСКИХ МОДУЛЕЙ ===")
    let_log(f"Всего файлов для загрузки: {len(another_tools_files_addresses)}")
    # 1. Объединенный поиск web_search и ask_user в одном цикле
    web_search_module = None
    ask_user_module = None
    other_modules_files = []
    for file_path in another_tools_files_addresses:
        file_name = os.path.basename(file_path).lower()
        let_log(f"Проверка файла: {file_name}")
        # Ищем веб-поиск (файл должен заканчиваться на web_search.py)
        if file_name.endswith('web_search.py'):
            let_log(f"✓ Найден файл веб-поиска: {file_path}")
            try:
                # Загружаем модуль веб-поиска отдельно
                module_name = os.path.splitext(file_name)[0]
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, 'main'):
                    # Создаем обертку с именем web_search
                    def web_search_wrapper(arg, func=module.main): return func(arg)
                    web_search_module = (module, web_search_wrapper, file_path)
                    let_log(f"  Модуль веб-поиска загружен: {module_name}")
                else: let_log(f"  ⚠ Файл веб-поиска не содержит функцию main")
            except Exception as e: let_log(f"  ⚠ Ошибка загрузки веб-поиска: {e}")
        elif file_name == 'ask_user.py':
            let_log(f"✓ Найден файл ask_user: {file_path}")
            try:
                module_name = os.path.splitext(file_name)[0]
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, 'main'):
                    ask_user_module = (module, module.main, file_path)
                    let_log(f"  Модуль ask_user загружен: {module_name}")
                else: let_log(f"  ⚠ Файл ask_user не содержит функцию main")
            except Exception as e: let_log(f"  ⚠ Ошибка загрузки ask_user: {e}")
        else: other_modules_files.append(file_path)
    # 2. Загружаем остальные модули через mod_loader
    loaded_tools = []
    if other_modules_files:
        let_log(f"\nЗагрузка остальных модулей ({len(other_modules_files)} файлов)")
        loaded_tools = mod_loader(other_modules_files)
    if web_search_module:
        module, web_search_wrapper, file_path = web_search_module
        # Загружаем локализацию для веб-поиска
        current_lang = globals().get('language', 'en')
        locale_data = load_locale(file_path, current_lang)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                match = re.match(r'^\s*[\'"]{3}\s*\n\s*([^\n]+)\n\s*([^\n]+)', content)
                if match:
                    command_name = match.group(1).strip()
                    description = match.group(2).strip()
                    # Применяем локализацию, если она есть
                    if locale_data and 'module_doc' in locale_data:
                        if len(locale_data['module_doc']) >= 2:
                            command_name = locale_data['module_doc'][0] or command_name
                            description = locale_data['module_doc'][1] or description
                    # Добавляем в loaded_tools
                    loaded_tools.append((command_name, description, module.main))
                    let_log(f"  Веб-поиск добавлен в список инструментов: {command_name}")
                    globals()['web_search'] = web_search_wrapper
                    let_log(f"✅ Функция web_search глобализована")
                else: let_log(f"  ⚠ Не удалось извлечь command_name из файла веб-поиска")
        except Exception as e: let_log(f"  ⚠ Ошибка обработки файла веб-поиска: {e}")
    else: let_log(f"⚠ Файл веб-поиска не найден в списке")
    if ask_user_module:
        module, ask_user_func, file_path = ask_user_module
        # Загружаем локализацию для ask_user
        current_lang = globals().get('language', 'en')
        locale_data = load_locale(file_path, current_lang)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                match = re.match(r'^\s*[\'"]{3}\s*\n\s*([^\n]+)\n\s*([^\n]+)', content)
                if match:
                    command_name = match.group(1).strip()
                    description = match.group(2).strip()
                    # Применяем локализацию, если она есть
                    if locale_data and 'module_doc' in locale_data:
                        if len(locale_data['module_doc']) >= 2:
                            command_name = locale_data['module_doc'][0] or command_name
                            description = locale_data['module_doc'][1] or description
                    # Добавляем в loaded_tools
                    loaded_tools.append((command_name, description, ask_user_func))
                    let_log(f"  Ask_user добавлен в список инструментов: {command_name}")
                    # Глобализуем ask_user
                    globals()['ask_user'] = ask_user_func
                    let_log(f"✅ Функция ask_user глобализована")
                else: let_log(f" ⚠ Не удалось извлечь command_name из файла ask_user")
        except Exception as e: let_log(f" ⚠ Ошибка обработки файла ask_user: {e}")
    # 4. Настраиваем global_state
    global_state.another_tools = loaded_tools
    let_log(f"\n=== ИТОГИ ЗАГРУЗКИ ===")
    let_log(f"Всего загружено инструментов: {len(loaded_tools)}")
    let_log(f"Веб-поиск доступен: {'web_search' in globals()}")
    let_log(f"Ask_user доступен: {'ask_user' in globals()}")
    let_log("Список инструментов:")
    for tt, t, _ in global_state.another_tools:
        global_state.tools_str += tt + ' ' + t + '\n'
        global_state.module_tools_keys.append(tt)
        let_log(tt)
    # === Загрузка пользовательских данных ===
    if fl:
        send_output_message(text=start_load_attachments_text)
        upload_user_data(fl)
        send_output_message(text=end_load_attachments_text)
    # === Запуск обработки ===
    let_log("ЗАПУСК")
    try: worker(initial_text)
    except Exception as e:
        print(f"Ошибка: {e}")
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(f"Файл: {tb.filename}, строка: {tb.lineno}")
        tb = traceback.extract_tb(e.__traceback__)[-1]
        t = f"{e} | {tb.filename}:{tb.lineno}"
        send_ui_no_cache(t)
        log_file = os.path.join(chat_path, 'log.txt')
        with open(log_file, 'a', encoding='utf-8') as f: f.write(f'{t}\n')
    try: model_disconnect()
    except: pass