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
import lzma
import json
import numpy as np
from sklearn.decomposition import PCA
import io
import base64
import inspect
from multiprocessing import queues
import shutil
import secrets

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
        self.max_critic_reactions = 2
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
        self.start_dialog_command_name = ''
        self.skip_tools_keys = []
        self.last_agent = None
        self.skip_nested_images = 0
        self.allow_ocr = 1
        self.wrong_command_messages_vector_ids = []
        self.number_of_plan_items = 0
        self.psm_operator_person = {} # TODO: потом объедини с другими в словарь с чат айди и подсловарями, также имена пространств должны быть короткими, в районе 3 символов
        self.current_agent_history_for_filesystem = ""
global_state = GlobalState()

chat_path = ''
filesystem_project_path = ''
vector_id_out = ''
do_chat_construct = False
native_func_call = False
use_rag = None
ui_conn = None
remove_loops = True
cache_path = ''
cache_can_write = False
agent_func = None
use_librarian = True
recreate_agents = False
cut_wrong_command_history = True
use_old_gigo = True
use_gigo = True
librarian_use_models = False
module_hints_for_operator = False
give_all_tools = False
critic_reuse_dialog = True
one_shot_intention_permission = False

default_handlers_names = { # это из настроек должно выгружаться, или лучше из документации хэндлеров которая внутри них
    'doc': 'process_docx',
    'docx': 'process_docx',
    'txt': 'process_text',
    'pdf': 'process_pdf',
    'png': 'process_image',
    'jpg': 'process_image',
    'jpeg': 'process_image',
    'zip': 'process_zip',
    'xlsx': 'process_excel',
    'xls': 'process_excel',}
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
    'gigo',
    'set_common_save_id',
    'reset_common_save_id',
    'down_hierarchy',
    'up_hierarchy',
    'next_executor',]
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
    "tool_result_end": "",}
use_user = False
chunk_size = 1000 # TODO:

pipeline = None
get_dependency_report = None
change_dir = None
get_project_tree_json = None
create_experiment_branch = None
status_success = None
status_failed = None
status_forbidden = None
resolve_workspace_path = None
to_posix_rel = None
allowed_actions = None
normalize_action = None

get_provider_embs = None
ask_provider_model = None
ask_provider_model_chat = None
memory_sql = None
client = None
milana_collection = None
user_collection = None
rag_collection = None
cache_counter = 1 # всегда начинается с 1
left_cache_counter = 0 # остаток
pending_write_cache_ids = []
language = ''
is_print_log = True
is_save_log = True

_cache_context_active = False

def no_cache(func):
    """Декоратор, запрещающий вызов функции внутри кэшируемой функции."""
    func._no_cache = True
    def wrapper(*args, **kwargs):
        global _cache_context_active
        if _cache_context_active:
            raise RuntimeError(
                f"Функция '{func.__name__}' помечена @no_cache и не может быть вызвана "
                f"внутри кэшируемой функции (используйте @cacher без @no_cache внутри)."
            )
            sys.exit(1)
        return func(*args, **kwargs)
    return wrapper

def cacher(func): # Декоратор для функций с кэшированием (ask_model, get_embs, coll_exec, sql_exec)
    def wrapper(*args, **kwargs):
        cached = read_cache()
        if cached != [False]:
            if isinstance(cached[1], dict) and '__exception__' in cached[1]: exc_data = cached[1]['__exception__']; exc = RuntimeError(exc_data['message']); raise exc
            let_log(f"[Используется кэшированный результат для {func.__name__}]")
            return cached[1]
        global _cache_context_active
        try:
            result = func(*args, **kwargs)
            write_cache(result)
            return result
            _cache_context_active = False
        except Exception as e: # Сохраняем исключение в кэше без traceback
            exc_data = {'__exception__': {'type': type(e).__name__, 'message': str(e), 'traceback_str': traceback.format_exc()}}
            write_cache(exc_data)
            _cache_context_active = False
            raise
    return wrapper

def load_locale(module_file, current_lang='en'): # Загружает локализацию для модуля из соответствующего файла
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
    #return
    t = str(t)
    # Получаем информацию о вызывающем коде
    stack = traceback.extract_stack()
    # stack[-1] - текущая функция let_log, stack[-2] - место вызова let_log
    # нам нужен стек на один уровень выше (кто вызвал let_log)
    if len(stack) >= 2:
        caller = stack[-2]
        filename = caller.filename.split('/')[-1]  # только имя файла
        lineno = caller.lineno
        funcname = caller.name
        caller_info = f"[{filename}:{lineno} {funcname}]"
    else: caller_info = "[unknown]"
    full_message = f"{caller_info} {t}"
    if is_print_log: print(full_message)
    if is_save_log:
        if left_cache_counter == 0: lname = 'log.txt'
        else: return#lname = 'log1.txt'
        log_file = os.path.join(chat_path, lname)
        with open(log_file, 'a', encoding='utf-8') as f: f.write(f'{full_message}\n')

# Опционально: полный стек функций + enter/exit (global_state.trace_full / trace_func_io)
_trace_func_depth = {}

def traceprint(*args, show_all=None, **kwargs):
    """Лог точки вызова. show_all=True или global_state.trace_full — вся цепочка функций."""
    stack = traceback.extract_stack()[:-1]  # без самого traceprint
    full = show_all if show_all is not None else bool(getattr(global_state, 'trace_full', False))
    if full and len(stack) > 1:
        chain = []
        for fr in stack[-12:]:
            chain.append(f"{fr.filename.split('/')[-1]}:{fr.lineno}:{fr.name}")
        prefix = " → ".join(chain)
    else:
        caller = stack[-1] if stack else None
        if caller:
            prefix = f"[{caller.filename.split('/')[-1]}:{caller.lineno}:{caller.name}]"
        else:
            prefix = "[traceprint]"
    if not args:
        let_log(prefix)
    else:
        let_log(f"{prefix}:", *args, **kwargs)

def trace_func(name=None, *, enabled=None):
    """
    Декоратор: логирует вход/выход из функции (опционально).
    Включается global_state.trace_func_io или enabled=True.
    """
    def deco(fn):
        fname = name or getattr(fn, '__name__', 'func')
        def wrapper(*a, **kw):
            on = enabled if enabled is not None else bool(getattr(global_state, 'trace_func_io', False))
            if not on:
                return fn(*a, **kw)
            depth = _trace_func_depth.get(fname, 0)
            _trace_func_depth[fname] = depth + 1
            let_log(f"[trace ENTER x{depth+1}] {fname}")
            try:
                result = fn(*a, **kw)
                let_log(f"[trace EXIT  x{depth+1}] {fname}")
                return result
            except Exception as e:
                let_log(f"[trace FAIL  x{depth+1}] {fname}: {e}")
                raise
            finally:
                _trace_func_depth[fname] = max(0, _trace_func_depth.get(fname, 1) - 1)
        wrapper.__name__ = getattr(fn, '__name__', 'wrapper')
        wrapper.__doc__ = getattr(fn, '__doc__', None)
        return wrapper
    return deco

def read_cache():
    global cache_counter, left_cache_counter
    global cache_can_write
    global pending_write_cache_ids
    cache_conn = None
    try:
        if left_cache_counter > 0:
            cache_conn = connect(cache_path)
            cache_cursor = cache_conn.cursor()
            let_log(f"Попытка чтения кэша с id: {cache_counter}")
            cache_cursor.execute('SELECT value FROM cache WHERE id = ?', (cache_counter,))
            result = cache_cursor.fetchone()
            if result != None:
                stored_data = result[0]
                marker = stored_data[0:1]
                data_part = stored_data[1:]
                if marker == b'\x00': decompressed_bytes = data_part
                elif marker == b'\x01': decompressed_bytes = lzma.decompress(data_part)
                elif marker == b'\x02':
                    let_log('маркер спуска')
                    pending_write_cache_ids.append(cache_counter)
                    cache_counter += 1
                    left_cache_counter -= 1
                    cache_conn.close()
                    cache_conn = None
                    return [False]
                try: deserialized_value = pickle.loads(decompressed_bytes)
                except Exception as pickle_error: raise
                let_log(f"[CACHE READ] id={cache_counter}")
                cache_counter += 1
                left_cache_counter -= 1
                let_log(deserialized_value)
                cache_conn.close()
                cache_conn = None
                return [True, deserialized_value]
        if cache_can_write:
            cache_cursor.execute('INSERT INTO cache (id, value) VALUES (?, ?)', (cache_counter, b'\x02'))
            cache_conn.commit()
            cache_conn.close()
            cache_conn = None
            pending_write_cache_ids.append(cache_counter)
            let_log(f'записан маркер {cache_counter}')
            cache_counter += 1
        else:
            cache_can_write = True
            let_log(f"Запись кэша с id {cache_counter} не найдена.")
        return [False]
    except Exception as e:
        if cache_conn: cache_conn.close()
        error_msg = f'{e}'
        send_ui_no_cache(error_msg)
        raise SystemExit(error_msg)

def compres_to_cache(content):
    pickled_bytes = pickle.dumps(content, protocol=pickle.HIGHEST_PROTOCOL)
    raw_size = len(pickled_bytes)
    compressed = lzma.compress(pickled_bytes, preset=9)
    size_uncompressed = 1 + raw_size
    size_compressed = 1 + len(compressed)
    if size_compressed < size_uncompressed: final_data = b'\x01' + compressed
    else: final_data = b'\x00' + pickled_bytes
    return final_data

def write_cache(content):
    global cache_counter
    global cache_can_write
    global pending_write_cache_ids
    cache_conn = None
    try:
        cache_conn = connect(cache_path)
        cache_cursor = cache_conn.cursor()
        if not cache_can_write:
            if pending_write_cache_ids == []: raise RuntimeError('Read/write sequence violation in the save system! Write command was expected.')
            cache_cursor.execute('DELETE FROM cache WHERE id >= ?', (pending_write_cache_ids[-1],))
            cache_counter = pending_write_cache_ids[-1]
            cache_cursor.execute('INSERT INTO cache (id, value) VALUES (?, ?)', (cache_counter, compres_to_cache(content)))
            cache_conn.commit()
            cache_conn.close()
            cache_conn = None
            del pending_write_cache_ids[-1]
            return True
        cache_cursor.execute('INSERT INTO cache (id, value) VALUES (?, ?)', (cache_counter, compres_to_cache(content)))
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

def send_ui_no_cache(t, attach=None, comm=''):
    message_data = {'text': t, 'attachments': attach, 'command': comm}
    try: ui_conn[1].put(message_data)
    except: pass

def send_log_to_ui(message: str):
    try: ui_conn[2].put(message)
    except Exception as e: let_log(f"Failed to send log to UI: {e}")

@cacher
def get_input_message(command=None, timeout=None, wait=False):
    answer = None
    if command: # Получаем сообщение из очереди
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
def _strip_ui_command_markers(text: str) -> str:
    """Убирает из текста для UI маркеры команд вида !!!name!!! / !!!!name!!!!."""
    if not text:
        return text or ''
    # служебные маркеры команд, не показываем пользователю
    cleaned = re.sub(r'!{2,4}\s*[\w\-]+\s*!{2,4}', '', text)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned if cleaned else text

def send_output_message(text=None, attachments=None, command=None):
    display = _strip_ui_command_markers(text) if text else ''
    message_data = {'text': display, 'attachments': attachments or None, 'command': command}
    try: ui_conn[1].put(message_data)
    except Exception as e: let_log(f"Ошибка при отправке сообщения: {e}"); return
    return True

@cacher
def global_trans_cache_exec(query, params=(), fetchone=False, fetchall=False, retries=3):
    """
    Выполняет SQL-запрос к глобальной БД settings.db.
    При блокировке БД делает повторные попытки.
    """
    if not global_trans_db_path: raise RuntimeError("global_trans_db_path не установлен. Вызовите set_cache_settings(settings_path=...)")
    conn = None
    for attempt in range(retries):
        try:
            conn = sqlite3.connect(global_trans_db_path)
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS translation_cache (src_text TEXT NOT NULL, translation TEXT NOT NULL, to_lang TEXT NOT NULL)")
            conn.commit()
            cursor.execute(query, params)
            if fetchone: result = cursor.fetchone()
            elif fetchall: result = cursor.fetchall()
            else: result = None
            conn.commit()
            conn.close()
            return result
        except sqlite3.OperationalError as e:
            if conn: conn.close()
            if "locked" in str(e) and attempt < retries - 1:
                time.sleep(0.5)
                continue
            else: raise
        except Exception:
            if conn: conn.close()
            raise
    return None

@cacher
def sql_exec(query, params=(), fetchone=False, fetchall=False, executemany=False):
    let_log('ОЧЕРЕДЬ')
    let_log(query)
    try:
        cursor = memory_sql.cursor()  # TODO: тут коннект каждый раз не создаётся, но тогда почему каждый раз создаётся курсор
        if executemany:
            cursor.executemany(query, params) # params - список кортежей
        else:
            print(query)
            print(params)
            cursor.execute(query, params)
        memory_sql.commit()
        result = None
        if fetchone:
            result = cursor.fetchone()
            if result and len(result) == 1:
                result = result[0]
        elif fetchall:
            result = cursor.fetchall()
        let_log('РЕЗУЛЬТАТ')
        let_log(result)
        return result
    except Exception as e:
        let_log(f"Ошибка SQL-запроса: {query} с {params} — {e}")
        raise

@cacher
def coll_exec(action, coll_name, *,
              query_embeddings=None,
              filters=None,
              doc_contains=None,
              ids=None,
              documents=None,
              metadatas=None,
              embeddings=None,
              fetch="documents",
              n_results=10,
              limit=None,
              offset=None,
              first=True,
              flatten=False,
              new_name=None,
              new_meta=None,
              client_override=None,
              relevance_coeff=0.9,
              **kwargs):
    """
    Универсальная обёртка для работы с коллекциями ChromaDB.
    Все вспомогательные функции вложены внутрь.
    """
    global_vars = globals() # ---- Глобальные настройки (с подстановкой значений по умолчанию) ----
    enable_compression = global_vars.get('enable_compression', True)
    compression_threshold = global_vars.get('compression_threshold', 1.0)
    # ---- Вспомогательные функции сжатия ----
    def _compress_doc_always(doc): # Всегда пытается сжать документ. Возвращает 'L' + base64(lzma(...)) или 'n' + оригинал
        if doc is None: return None
        if isinstance(doc, (bytes, bytearray)): raw_bytes = bytes(doc); original_str = doc.decode("utf-8") if hasattr(doc, 'decode') else str(doc)
        else: original_str = str(doc); raw_bytes = original_str.encode("utf-8")
        uncompressed_str = "n" + original_str
        uncompressed_size = len(uncompressed_str.encode("utf-8"))
        try:
            compressed_bytes = lzma.compress(raw_bytes, preset=9)
            compressed_b64 = base64.b64encode(compressed_bytes).decode("ascii")
            compressed_str = "L" + compressed_b64
            compressed_size = len(compressed_str.encode("utf-8"))
            if compressed_size < uncompressed_size: return compressed_str
            else: return uncompressed_str
        except Exception: return uncompressed_str
    def _decompress_doc_always(comp): # Распаковывает документ: 'L' -> lzma, 'z' -> gzip (старый формат), 'n' -> вернуть как есть
        if comp is None: return None
        if not comp: return comp
        first_char = comp[0]
        content = comp[1:]
        if first_char == 'L':
            decoded_bytes = base64.b64decode(content)
            decompressed_bytes = lzma.decompress(decoded_bytes)
            return decompressed_bytes.decode("utf-8")
        elif first_char == 'n': return content
        else: return comp
    def _compress_documents_always(coll_name, documents_list): # Сжимает список документов для коллекций, для которых включено сжатие.
        if documents_list is None: return None
        if not enable_compression or coll_name not in ("milana_collection", "user_collection"): return documents_list
        out = []
        for d in documents_list:
            if d is None: out.append(None); continue
            out.append(_compress_doc_always(d))
        return out
    def _decompress_documents_always(coll_name, documents_list): # Распаковывает список документов
        if documents_list is None: return None
        if not enable_compression or coll_name not in ("milana_collection", "user_collection"): return documents_list
        out = []
        for d in documents_list:
            if d is None: out.append(None); continue
            out.append(_decompress_doc_always(d))
        return out
    # ---- Остальные вспомогательные функции ----
    def _make_where(d): # Преобразует словарь фильтров в формат ChromaDB where.
        if not d: return None
        clauses = []
        for k, v in d.items():
            if isinstance(v, list): clauses.append({k: {"$in": v}})
            elif isinstance(v, dict) and any(op in v for op in ["$gt", "$gte", "$lt", "$lte", "$ne", "$eq", "$in", "$nin"]): clauses.append({k: v})
            else: clauses.append({k: v})
        if len(clauses) == 1: return clauses[0]
        else: return {"$and": clauses}
    def _filter_relevance(resp, coeff=0.9): # Фильтрует результаты по расстоянию: оставляет только те, чьё расстояние <= best * (1 + (1-coeff))
        if "distances" not in resp or resp["distances"] is None or not resp["distances"]: return resp
        dists = resp["distances"][0] if isinstance(resp["distances"][0], list) else resp["distances"]
        if not dists: return resp
        best = min(dists)
        threshold = best * (1.0 + (1.0 - coeff))
        keep_idx = [i for i, d in enumerate(dists) if d <= threshold]
        if not keep_idx: return {k: [] for k in resp}
        out = {}
        for k, v in resp.items():
            if isinstance(v, list) and v and isinstance(v[0], list): out[k] = [[row[i] for i in keep_idx] for row in v]
            elif isinstance(v, list): out[k] = [v[i] for i in keep_idx]
            else: out[k] = v
        return out
    def _process_in_nin_operators(coll, filters, coll_name, get_results=True):
        """
        Обрабатывает фильтры $in и $nin для поля vector_id.
        Возвращает либо список ID (если get_results=False), либо результат coll.get.
        """
        print(f"[{coll_name}] Запуск обхода (in/nin) для ID с фильтрами: {filters}")
        nin_ids = set(filters.get('$nin', {}).get('vector_id', []))
        in_ids = set(filters.get('$in', {}).get('vector_id', []))
        base_where_filter = {
            k: v for k, v in filters.items()
            if k not in ('$in', '$nin', 'vector_id')}
        all_ids = set()
        offset = 0
        batch_size = 1000
        where_for_get = _make_where(base_where_filter)
        while True:
            r = coll.get(where=where_for_get, limit=batch_size, offset=offset, include=[])
            current_ids = r.get('ids', [])
            if not current_ids: break
            all_ids.update(current_ids)
            offset += batch_size
            if len(current_ids) < batch_size: break
        print(f"[{coll_name}] Найдено {len(all_ids)} ID до фильтрации $in/$nin.")
        final_ids = all_ids
        if nin_ids: final_ids = final_ids - nin_ids
        if in_ids: final_ids = final_ids.intersection(in_ids)
        final_ids_list = list(final_ids)
        print(f"[{coll_name}] Осталось {len(final_ids_list)} ID после фильтрации $in/$nin.")
        if not get_results: return final_ids_list
        if final_ids_list: return coll.get(ids=final_ids_list, include=['metadatas', 'documents', 'embeddings'])
        return {'ids': [], 'metadatas': [], 'documents': [], 'embeddings': []}
    # ---- Получение коллекции ----
    coll = globals().get(coll_name)
    if coll is None and client_override:
        try: coll = client_override.get_collection(coll_name)
        except Exception: pass
    if coll is None and client:
        try: coll = client.get_collection(coll_name)
        except Exception: pass
    if coll is None: raise NameError(f"Collection '{coll_name}' not found")
    # ---- Вспомогательная для пустого результата ----
    def _empty_result(fetch):
        include = fetch if isinstance(fetch, list) else [fetch]
        if len(include) > 1:
            out = {}
            for key in include: out[key] = [] if key != "distances" else [[]]
            return out
        else:
            key = include[0]
            if key == "ids": return []
            elif key == "documents": return []
            elif key == "metadatas": return []
            elif key == "embeddings": return []
            elif key == "distances": return [[]]
            else: return None
    # ---- Внутренняя _extract (исправленная) ----
    def _extract(resp, include):
        if len(include) > 1:
            out = {}
            for key in include:
                data = resp.get(key, []) or []
                if key == "documents":
                    if isinstance(data, list) and data and isinstance(data[0], list): data = [_decompress_documents_always(coll_name, sub) for sub in data]
                    else: data = _decompress_documents_always(coll_name, data)
                if flatten and isinstance(data, list) and data and isinstance(data[0], list): data = [i for sub in data for i in sub]
                out[key] = data
            return out
        else:
            key = include[0]
            data = resp.get(key, []) or []
            if key == "documents":
                if isinstance(data, list) and data and isinstance(data[0], list): data = [_decompress_documents_always(coll_name, sub) for sub in data]
                else: data = _decompress_documents_always(coll_name, data)
            if first:
                if isinstance(data, list) and data and isinstance(data[0], list): return data[0][0] if data[0] else None
                else: return data[0] if data else None
            else:
                if isinstance(data, list) and data and isinstance(data[0], list): return [i for sub in data for i in sub]
                else: return data
    # ---- Проверка пустых эмбеддингов для записи ----
    if action in ("add", "update") and embeddings is not None:
        for i, emb in enumerate(embeddings):
            if emb is None or (isinstance(emb, list) and len(emb) == 0): print(f"[coll_exec] ⚠ Пустой эмбеддинг для {action}, индекс {i}"); return None
    if action == "query":
        if query_embeddings is None: return _empty_result(fetch)
        all_empty = True
        for qe in query_embeddings:
            if qe and isinstance(qe, list) and len(qe) > 0: all_empty = False; break
        if all_empty: return _empty_result(fetch)
    # ---- Проверка на специальные фильтры $in/$nin для vector_id ----
    id_filters_present = (filters and ((filters.get('$nin') and isinstance(filters.get('$nin'), dict) and 'vector_id' in filters['$nin']) or (filters.get('$in') and isinstance(filters.get('$in'), dict) and 'vector_id' in filters['$in'])))
    if action in ("query", "get") and id_filters_present:
        processed = _process_in_nin_operators(coll, filters, coll_name, get_results=True)
        if isinstance(processed, dict) and 'ids' in processed:
            resp = processed
            include = fetch if isinstance(fetch, list) else [fetch]
            if include == ["all"]: include = ["ids", "documents", "metadatas", "embeddings", "distances"]
            if action == "query": resp = _filter_relevance(resp, relevance_coeff)
            return _extract(resp, include)
    # ---- Основные действия ----
    try:
        if action == "add": docs_to_send = _compress_documents_always(coll_name, documents); return coll.add(ids=ids, documents=docs_to_send, metadatas=metadatas, embeddings=embeddings, **kwargs)
        if action == "update": docs_to_send = _compress_documents_always(coll_name, documents); return coll.update(ids=ids, documents=docs_to_send, metadatas=metadatas, embeddings=embeddings, **kwargs)
        if action == "delete": return coll.delete(ids=ids, where=_make_where(filters), **kwargs)
        if action == "count": return coll.count()
        if action == "modify": return coll.modify(name=new_name, metadata=new_meta)
        if action == "delete_collection":
            if client is None and client_override is None: raise ValueError("client required for delete_collection")
            cl = client_override or client
            return cl.delete_collection(coll_name)
        if action in ("query", "get"):
            include = fetch if isinstance(fetch, list) else [fetch]
            if include == ["all"]: include = ["ids", "documents", "metadatas", "embeddings", "distances"]
            params = {}
            if action == "query":
                params.update({"query_embeddings": query_embeddings or [], "where": _make_where(filters), "n_results": n_results})
                if doc_contains: params["where_document"] = {"$contains": doc_contains}
            else: # get
                params.update({"where": _make_where(filters), "limit": limit, "offset": offset})
                if doc_contains: params["where_document"] = {"$contains": doc_contains}
            params["include"] = include
            params.update(kwargs)
            resp = (coll.query if action == "query" else coll.get)(**params)
            if not resp.get("ids") or not any(resp["ids"]): return _extract(resp, include)
            if action == "query": resp = _filter_relevance(resp, relevance_coeff)
            return _extract(resp, include)
        raise ValueError(f"Unsupported action: {action}")
    except Exception as e: print(f"[coll_exec] Ошибка ({action}): {e}"); return None

def load_chat_settings(chat_id): # Загрузка настроек чата из SQLite БД
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

def load_initial_data(chat_id): # Загрузка начальных данных: задачи и вложений
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
    # Определяем, какой язык загружать
    lang_to_load = language
    if do_translate and local_and_tools_translate:
        # Пытаемся загрузить целевой язык, если он есть
        try:
            lang_module = __import__(f'lang.{target_lang}.system_texts', fromlist=['system_text_container'])
            container = lang_module.system_text_container()
            let_log(f"Загружена локализация для целевого языка '{target_lang}'")
            # Экспортируем напрямую без перевода
            for attr in dir(container):
                if attr.startswith('__'):
                    continue
                value = getattr(container, attr)
                if isinstance(value, str):
                    try:
                        setattr(container, attr, value)
                    except Exception as e:
                        let_log(f"Ошибка '{attr}': {e}")
                        pass
            for attr in dir(container):
                if attr.startswith('__'):
                    continue
                value = getattr(container, attr)
                globals()[attr] = value
            let_log(f"Языковой пакет '{target_lang}' загружен и экспортирован")
            return
        except ImportError:
            let_log(f"Локализация для '{target_lang}' не найдена, загружаем '{language}' и переводим")
            lang_to_load = language
    # Загружаем пакет для lang_to_load (родной язык или fallback)
    try:
        lang_module = __import__(f'lang.{lang_to_load}.system_texts', fromlist=['system_text_container'])
        container = lang_module.system_text_container()
    except ImportError as e:
        let_log(f"Ошибка загрузки языкового модуля '{lang_to_load}': {str(e)}")
        try:
            from lang.en.system_texts import system_text_container
            container = system_text_container()
            let_log(f"Используются тексты по умолчанию (en)")
            lang_to_load = 'en'
        except ImportError:
            let_log("Критическая ошибка: не найден модуль с текстами!")
            return
    # Если включён перевод и мы загрузили не целевой язык, переводим строки
    if do_translate and local_and_tools_translate and lang_to_load != target_lang:
        # Собираем все строки из контейнера
        original_strings = {}
        for attr in dir(container):
            if attr.startswith('__'):
                continue
            value = getattr(container, attr)
            if isinstance(value, str):
                original_strings[attr] = value
        if original_strings:
            # Переводим все строки одним вызовом translate_texts
            str_list = list(original_strings.values())
            translated_list = translate_texts(str_list, target_lang, from_lang=lang_to_load)
            # Обновляем контейнер и глобальные переменные
            for attr, trans in zip(original_strings.keys(), translated_list):
                setattr(container, attr, trans)
                globals()[attr] = trans
            let_log(f"Переведено {len(translated_list)} строк с '{lang_to_load}' на '{target_lang}'")
        else:
            # Если строк нет, просто экспортируем как есть
            for attr in dir(container):
                if attr.startswith('__'):
                    continue
                value = getattr(container, attr)
                if isinstance(value, str):
                    globals()[attr] = value
    else:
        # Без перевода - просто экспортируем
        for attr in dir(container):
            if attr.startswith('__'):
                continue
            value = getattr(container, attr)
            if isinstance(value, str):
                try:
                    setattr(container, attr, value)
                except Exception as e:
                    let_log(f"Ошибка '{attr}': {e}")
                    pass
        for attr in dir(container):
            if attr.startswith('__'):
                continue
            value = getattr(container, attr)
            globals()[attr] = value
    let_log(f"Языковой пакет '{lang_to_load}' загружен, переменные экспортированы")

def _check_module_uses_cross_gpt(file_contents):
    """Проверяет, использует ли модуль важные функции из cross_gpt, которые требуют кэширования."""
    # Удаляем комментарии
    lines = file_contents.split('\n')
    clean_lines = []
    for line in lines:
        if '#' in line: line = line[:line.index('#')]
        clean_lines.append(line)
    clean_content = '\n'.join(clean_lines)
    # 1. Импорт из chat_manager (всегда системный)
    if re.search(r'from\s+chat_manager\s+import', clean_content, re.IGNORECASE): return True
    if 'chat_manager.' in clean_content: return True
    # 2. Проверяем использование cross_gpt.важная_функция
    for func in important_functions:
        if f'cross_gpt.{func}' in clean_content: return True
    # 3. Ищем импорты из cross_gpt
    # Многострочный импорт: from cross_gpt import ( ... )
    multi_pattern = r'from\s+cross_gpt\s+import\s*\(([^)]+)\)'
    for match in re.finditer(multi_pattern, clean_content, re.DOTALL | re.IGNORECASE):
        imports = [imp.strip().split()[0] for imp in match.group(1).split(',') if imp.strip()]
        if '*' in imports: return True
        for imp in imports:
            if ' as ' in imp: imp = imp.split(' as ')[0].strip()
            if imp in important_functions: return True
    # Однострочный импорт: from cross_gpt import func1, func2, ...
    single_pattern = r'from\s+cross_gpt\s+import\s+([^\(\n]+)'
    for match in re.finditer(single_pattern, clean_content, re.IGNORECASE):
        imports = [imp.strip().split()[0] for imp in match.group(1).split(',') if imp.strip()]
        if '*' in imports: return True
        for imp in imports:
            if ' as ' in imp: imp = imp.split(' as ')[0].strip()
            if imp in important_functions: return True
    # 4. Импорты внутри функций (могут быть многострочными)
    all_imports = re.findall(r'from\s+cross_gpt\s+import\s+.*?(?=\n|$)', clean_content, re.DOTALL | re.IGNORECASE)
    for import_stmt in all_imports:
        import_part = import_stmt.split('import', 1)[1].strip()
        if '(' in import_part and ')' in import_part:
            start = import_part.find('(') + 1
            end = import_part.rfind(')')
            import_list = import_part[start:end]
        else: import_list = import_part
        imports = [imp.strip().split()[0] for imp in import_list.split(',') if imp.strip()]
        if '*' in imports: return True
        for imp in imports:
            if ' as ' in imp: imp = imp.split(' as ')[0].strip()
            if imp in important_functions: return True
    return False

def mod_loader(adrs):
    loaded_modules = []
    for mod_file in adrs:
        try:
            let_log(f"Processing: {mod_file}")
            if not os.path.isfile(mod_file): let_log(f"Файл {mod_file} не найден"); continue
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
            if doc_match: command_name = doc_match.group(1).strip(); description = doc_match.group(2).strip()
            # Если есть локализация, берем оттуда (теперь для любого языка)
            if locale_data:
                if 'module_doc' in locale_data and len(locale_data['module_doc']) >= 2: command_name = locale_data['module_doc'][0] or command_name; description = locale_data['module_doc'][1] or description
            # Проверяем, что получили command_name и description
            if not command_name or not description: let_log(f"Модуль {mod_file} должен содержать command_name и description (первые 2 строки файла или локализацию)"); continue
            # Загружаем основной модуль
            module_name = os.path.splitext(mod_file)[0]
            spec = importlib.util.spec_from_file_location(module_name, mod_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not hasattr(module, 'main'): let_log(f"Модуль {mod_file} должен содержать функцию main."); continue
            main_func = module.main
            if not callable(main_func): let_log(f"main в модуле {mod_file} должна быть функцией."); continue
            if main_func.__code__.co_argcount != 1: let_log(f"Функция main в модуле {mod_file} должна принимать ровно 1 аргумент."); continue
            # Инициализация атрибутов
            should_initialize = re.search(r"if\s+not\s+hasattr\s*\(\s*main\s*,\s*['\"]attr_names['\"]\s*\)", file_contents)
            if should_initialize:
                try:
                    main_func(None) # Инициализация атрибутов по умолчанию
                    attr_list = getattr(main_func, 'attr_names', []) # Применяем локализацию для любого языка, если есть данные
                    if locale_data:
                        for attr in attr_list:
                            localized_key = f'main.{attr}'
                            if localized_key in locale_data:
                                setattr(main_func, attr, locale_data[localized_key]) # Только для не-английских языков пишем лог о применении локализации
                                if current_lang != 'en': let_log(f"✓ Локализация: {attr} в {mod_file}")
                except Exception as e: let_log(f"⚠ Ошибка при инициализации {mod_file}: {e}"); continue
            else: let_log('НЕ ДОЛЖЕН')
            if os.path.basename(mod_file) == 'skip.py': global_state.skip_tools_keys.append(command_name); let_log(f"⚠ Модуль {mod_file} помечен как SKIP – команда '{command_name}' будет скрыта из описаний")
            # --- ДОБАВЛЕНО: Проверка на librarian и обновление описания, если доступен web_search ---
            if os.path.basename(mod_file) == 'librarian.py':
                if 'web_search' in globals() and callable(globals()['web_search']):
                    note = getattr(main_func, 'web_search_available', '')
                    if note: description = description + note; let_log(f"✓ Добавлено примечание о веб-поиске к описанию librarian")
            if _check_module_uses_cross_gpt(file_contents): global_state.system_tools_keys.append(command_name)
            loaded_modules.append((command_name, description, main_func))
        except Exception as e: let_log(f"⚠ Ошибка при обработке {mod_file}: {e}"); continue
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
    if found_start_dialog_index != -1: global_state.start_dialog_command_name = found_start_dialog_command
    # Формируем словари с ключом — кортеж токенов имени команды
    def to_dict(modules, files):
        d = {} # Используем глобальную переменную use_librarian
        global use_librarian
        for i, (cmd_t, desc_tokens, func) in enumerate(modules):
            filename = os.path.basename(files[i])
            # Если use_librarian == False и это librarian.py - пропускаем добавление в словарь
            if not use_librarian and filename == 'librarian.py': let_log(f"librarian.py загружен глобально, но НЕ добавлен в словарь команд (use_librarian=False)"); continue
            d[cmd_t] = (desc_tokens, func)
        return d
    common_dict = to_dict(common_modules, common_files)
    milana_dict = to_dict(milana_modules, milana_files)
    ivan_dict = to_dict(ivan_modules, ivan_files)
    # Выводим отладочную информацию
    let_log("\nЗагруженные системные команды:")
    for cmd in global_state.system_tools_keys: let_log(cmd)
    return ({**common_dict, **ivan_dict}, {**common_dict, **milana_dict})

@cacher
def get_embs(text):
    global emb_token_limit
    if not text or not text.strip():
        let_log("[get_embs] пустой текст — эмбеддинг не запрашиваем")
        return []
    current_text = text
    if len(current_text) * text_tokens_coefficient > emb_token_limit: half_len = len(current_text) // 2
    last_err = None
    for attempt in range(3):
        try:
            result = get_provider_embs(current_text)
            # Пустой/None результат — не пишем в Chroma, пробуем ещё раз
            if result is None or (isinstance(result, (list, tuple)) and len(result) == 0):
                last_err = "empty embedding vector"
                let_log(f"[get_embs] пустой embedding (попытка {attempt+1}/3), текст[:80]={current_text[:80]!r}")
                time.sleep(0.5 * (attempt + 1))
                continue
            if isinstance(result, (list, tuple)) and all(
                    (not isinstance(x, (int, float)) or x == 0) for x in (result[:8] if len(result) >= 8 else result)):
                # все нули в начале — подозрительно, но не всегда ошибка; логируем
                let_log(f"[get_embs] embedding начинается с нулей, dim={len(result)}")
            # Обновляем лимит, если пришлось урезать текст
            if len(current_text) < len(text):
                new_limit = int(len(current_text) * text_tokens_coefficient)
                let_log(f"[get_embs] Обновлён emb_token_limit: {new_limit} (был {emb_token_limit})")
                emb_token_limit = new_limit
            return result
        except Exception as e:
            last_err = e
            if 'ContextOverflowError' in str(e):
                half_len = len(current_text) // 2
                if half_len == 0: raise
                current_text = current_text[:half_len]
                continue
            else:
                let_log(f"[get_embs] Ошибка (попытка {attempt+1}/3): {e}")
                time.sleep(0.5 * (attempt + 1))
                continue
    let_log(f"[get_embs] не удалось получить embedding: {last_err}")
    return []

def get_token_limit(): return token_limit

def get_text_tokens_coefficient(): return text_tokens_coefficient

# ========== НОВЫЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (добавить в начало файла) ==========

def _retry_loop(get_response, is_valid, error_retry_delay=60, empty_retry_delay=2):
    """
    Универсальный цикл повторных попыток вызова провайдера.
    
    Аргументы:
        get_response: функция без аргументов, возвращающая ответ от провайдера
        is_valid: функция, принимающая ответ и возвращающая True, если ответ содержит контент
        error_retry_delay: пауза после исключения (сбой провайдера)
        empty_retry_delay: пауза после пустого/невалидного ответа
    
    Возвращает:
        Валидный ответ от провайдера (в том же формате, что вернул get_response)
    """
    start = time.time()
    had_error = False
    while True:
        try:
            response = get_response()
        except Exception as e:
            if 'ContextOverflowError' in str(e):
                raise RuntimeError("ContextOverflowError")
            let_log(e)
            send_ui_no_cache(f'{error_in_provider}\n{e}')
            had_error = True
            time.sleep(error_retry_delay)
            continue

        if not is_valid(response):
            # В UI не спамим «пустой ответ» — только в лог; UI — при реальных ошибках (сеть/лимит)
            let_log("[WARN] Модель вернула пустой или невалидный ответ. Повторная попытка...")
            time.sleep(empty_retry_delay)
            continue

        # Успешный ответ
        if had_error:
            send_ui_no_cache(success_in_provider)
        elapsed = time.time() - start
        let_log(f'Успешный ответ получен за {elapsed:.2f}s')
        return response

def _call_chat_with_retry(generation_params):
    """Для chat/completions: возвращает полный ответ провайдера (словарь) с проверкой наличия content или tool_calls."""
    def is_valid_chat_response(resp):
        # Проверка на наличие содержательного контента или вызова инструментов
        if "choices" in resp and resp["choices"]:
            choice = resp["choices"][0]
            message = choice.get("message", {})
            if message.get("tool_calls"):
                return True
            content = message.get("content", "")
            if content and content.strip():
                return True
        elif "message" in resp:
            msg = resp["message"]
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if content and content.strip():
                    return True
            elif isinstance(msg, str) and msg.strip():
                return True
        elif "response" in resp and resp["response"].strip():
            return True
        return False

    return _retry_loop(lambda: ask_provider_model_chat(generation_params), is_valid_chat_response)

def _call_completions_with_retry(generation_params):
    """Для completions: возвращает текстовую строку."""
    def is_valid_completions_response(resp):
        if isinstance(resp, str):
            return bool(resp.strip())
        elif isinstance(resp, dict):
            text = resp.get('response', '')
            return bool(text.strip())
        return False

    raw_response = _retry_loop(lambda: ask_provider_model(generation_params), is_valid_completions_response)
    # Извлекаем строку в зависимости от формата ответа
    if isinstance(raw_response, str):
        return raw_response
    elif isinstance(raw_response, dict):
        return raw_response.get('response', '')
    return ''

# ========== ПОЛНАЯ ФУНКЦИЯ ask_model ==========

# TODO: убрать температуру
@cacher
def ask_model(prompt_text, system_prompt: str = None, all_user: bool = False, limit: int = None, temperature: float = 0.6, **extra_params) -> str:
    let_log(system_prompt)
    let_log(prompt_text)
    let_log(f'ВХОД {len(prompt_text)} токенов')
    try: let_log(f'ПРОМПТ {len(system_prompt)} токенов')
    except: pass

    # --- Перевод входных данных, если включен и не local_and_tools_translate ---
    if do_translate and not local_and_tools_translate:
        if system_prompt is not None:
            system_prompt = translate_text(system_prompt, target_lang, from_lang=language)
        prompt_text = translate_text(prompt_text, target_lang, from_lang=language)

    # Проверка длины контекста
    if len(prompt_text) * text_tokens_coefficient > token_limit - 1000:
        raise RuntimeError("ContextOverflowError")
    '''
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
    '''
    if use_user:
        # Очищаем очередь перед отправкой запроса
        try:
            while True: ui_conn[0].get_nowait()
        except Empty: pass
        except Exception as e: let_log(f"Ошибка очистки очереди: {e}")
        # Отправляем запрос пользователю
        try: ui_conn[1].put({'text': "Пожалуйста, введите текст:", 'command': 'ask_user'})
        except Exception as e: let_log(f"Ошибка отправки запроса: {e}")
        # Ждём первое сообщение из очереди
        try:
            msg = ui_conn[0].get(timeout=None)
            if msg and msg.get('text') and msg.get('text').strip() != '###': return msg.get('text')
        except Empty: let_log("Очередь пуста, пользователь не ответил")
        except Exception as e: let_log(f"Ошибка получения ввода: {e}")
        let_log("[Пользователь не ввел текст, используется генерация моделью]")

    # --- Обработка особых случаев (system_prompt и all_user) ---
    if system_prompt:
        let_log("Режим (Особый случай): system_prompt -> chat/completions")
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_text}]
        generation_params = {"messages": messages, "temperature": temperature, "max_tokens": limit or token_limit}
        generation_params.update(extra_params)
        response = _call_chat_with_retry(generation_params)
        result = _process_chat_response(response)
        if do_translate and not local_and_tools_translate:
            result = translate_text(result, language, from_lang=target_lang)
        send_log_to_ui(result)
        return result
    
    if all_user:
        let_log("Режим (Особый случай): all_user=True -> chat/completions")
        messages = [{"role": "user", "content": prompt_text}]
        generation_params = {"messages": messages, "temperature": temperature, "max_tokens": limit or token_limit}
        generation_params.update(extra_params)
        response = _call_chat_with_retry(generation_params)
        result = _process_chat_response(response)
        if do_translate and not local_and_tools_translate:
            result = translate_text(result, language, from_lang=target_lang)
        send_log_to_ui(result)
        return result
    
    # --- Определение режима работы на основе do_chat_construct (1, 2, 3) ---
    if not do_chat_construct:
        # Режим 1: Подача строки (completions)
        let_log("Режим 1 (do_chat_construct=1): Подача строки -> completions")
        if not native_func_call:
            parsed_msgs = _parse_roles_to_messages_no_functions(prompt_text)
        else:
            parsed_msgs = _parse_roles_to_messages_functions(prompt_text, global_state.now_agent_id)
        generation_params = {"prompt": _serialize_messages_to_prompt(parsed_msgs), "temperature": temperature, "max_tokens": limit or token_limit, "echo": False}
        generation_params.update(extra_params)
        result = _call_completions_with_retry(generation_params)
        if do_translate and not local_and_tools_translate:
            result = translate_text(result, language, from_lang=target_lang)
        send_log_to_ui(result)
        return result
    
    elif do_chat_construct and not native_func_call:
        # Режим 2: Парсинг чата БЕЗ function call
        let_log("Режим 2 (do_chat_construct=2): Парсинг (без функций) -> chat/completions")
        messages = _parse_roles_to_messages_no_functions(prompt_text)
        generation_params = {"messages": messages, "temperature": temperature, "max_tokens": limit or token_limit}
        generation_params.update(extra_params)
        response = _call_chat_with_retry(generation_params)
        result = _process_chat_response(response)
        if do_translate and not local_and_tools_translate:
            result = translate_text(result, language, from_lang=target_lang)
        send_log_to_ui(result)
        return result
    
    elif do_chat_construct and native_func_call:
        # Режим 3: Парсинг чата С function call
        let_log("Режим 3 (do_chat_construct=3): Парсинг (С функциями) -> chat/completions")
        let_log(global_state.now_agent_id)
        # 1. Парсим историю
        messages = _parse_roles_to_messages_functions(prompt_text, global_state.now_agent_id)
        # 2. Получаем и форматируем доступные инструменты
        now_commands = global_state.tools_commands_dict.get(global_state.now_agent_id, {})
        let_log(now_commands)
        formatted_tools = _format_tools_for_api(now_commands)
        generation_params = {"messages": messages, "temperature": temperature, "max_tokens": limit or token_limit}
        # 3. Добавляем инструменты в запрос, если они есть
        let_log(formatted_tools)
        if formatted_tools:
            let_log('ДОБАВЛЯЕМ ИНСТРУМЕНТЫ')
            generation_params["tools"] = formatted_tools
            generation_params["tool_choice"] = "auto"
        generation_params.update(extra_params)
        let_log(generation_params)
        
        # Вызов с бесконечными ретраями (включая пустые ответы)
        response = _call_chat_with_retry(generation_params)
        
        # 5. Обрабатываем ответ как словарь
        let_log(response)
        if "choices" not in response or not response["choices"]:
            let_log("ask_model: Некорректный формат ответа - нет choices")
            raise RuntimeError("Некорректный формат ответа - нет choices")
        choice = response["choices"][0]
        message = choice.get("message", {})
        response_content = message.get("content", "") or ""
        tool_calls = message.get("tool_calls")
        
        if tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call['function']['name']
                arguments_json_str = tool_call['function']['arguments']
                try:
                    args_dict = json.loads(arguments_json_str)
                    arguments_str = args_dict.get('arguments', '')
                except Exception:
                    arguments_str = arguments_json_str
                marker = f"\n!!!{function_name}!!!{arguments_str}"
                response_content = marker + response_content
            if do_translate and not local_and_tools_translate:
                response_content = translate_text(response_content, language, from_lang=target_lang)
            send_log_to_ui(result)
            return response_content
        elif response_content:
            if do_translate and not local_and_tools_translate:
                response_content = translate_text(response_content, language, from_lang=target_lang)
            send_log_to_ui(result)
            return response_content
        send_log_to_ui(result)
        return response_content

def _process_chat_response(api_response):
    """
    Обрабатывает ответ от ask_provider_model_chat (словарь) и извлекает текстовый контент
    Теперь поддерживает как OpenAI-совместимый формат, так и формат Ollama
    """
    if "choices" in api_response and api_response["choices"]: # Проверяем наличие поля choices (OpenAI-совместимый формат)
        choice = api_response["choices"][0]
        if "message" in choice and "content" in choice["message"]:
            result = choice["message"]["content"].strip()
            let_log(f"_process_chat_response: Результат: '{result}'")
            return result
    elif "message" in api_response: # Если нет choices, проверяем прямой формат Ollama
        message = api_response["message"]
        if isinstance(message, dict) and "content" in message:
            result = message["content"].strip()
            let_log(f"_process_chat_response (Ollama формат): Результат: '{result}'")
            return result
        elif isinstance(message, str):
            result = message.strip()
            let_log(f"_process_chat_response (Ollama формат строка): Результат: '{result}'")
            return result
    elif "response" in api_response: # Проверяем поле response (для обратной совместимости)
        result = api_response["response"].strip()
        let_log(f"_process_chat_response (response поле): Результат: '{result}'")
        return result
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
    if last_messages_marker in remaining_prompt: # Разделяем на часть до метки (system) и после (диалог)
        system_part, dialog_part = remaining_prompt.split(last_messages_marker, 1)
        # Часть до метки - это системный промпт
        if system_part.strip(): messages.append({"role": "system", "content": system_part.strip()})
        remaining_prompt = dialog_part
    else:
        roles_to_find = [operator_role_text, worker_role_text, func_role_text]
        first_role_pos = -1
        first_role = None
        for role in roles_to_find:
            pos = remaining_prompt.find(role)
            if pos != -1 and (first_role_pos == -1 or pos < first_role_pos): first_role_pos = pos; first_role = role
        # Если найдена роль, разделяем на system и остальное
        if first_role_pos != -1:
            system_content = remaining_prompt[:first_role_pos].strip()
            if system_content: messages.append({"role": "system", "content": system_content})
            remaining_prompt = remaining_prompt[first_role_pos:]
        else: # Если ролей нет, но текст начинается не с роли - считаем системным промптом
            if remaining_prompt.strip() and not any(remaining_prompt.strip().startswith(role) for role in roles_to_find): messages.append({"role": "system", "content": remaining_prompt.strip()}); remaining_prompt = ""
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
            found_roles.append({'pos': pos, 'role': role, 'type': 'operator' if role == operator_role_text else 'worker' if role == worker_role_text else 'function'})
            start_idx = pos + len(role)
    found_roles.sort(key=lambda x: x['pos']) # Сортируем по позиции
    # Если не найдены роли, но текст есть - считаем все пользовательским сообщением
    if not found_roles and remaining_prompt.strip(): # Удаляем только operator и worker маркеры, func_role_text оставляем как есть
        clean_content = remaining_prompt.strip()
        for role in [operator_role_text, worker_role_text]: clean_content = clean_content.replace(role, '').strip()
        messages.append({"role": "user", "content": clean_content})
        return messages
    # Обрабатываем найденные роли с учетом чередования
    # Теперь просто чередуем user/assistant после system
    current_role = "user" # Начинаем с пользователя
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
        if role_type == 'function': # Для function роли: убираем только начальный \n если есть, но оставляем сам маркер
            clean_content = content
            # Убираем начальный перевод строки если он есть
            if clean_content.startswith('\n'): clean_content = clean_content[1:].strip()
            # Добавляем func_role_text в начало содержимого
            clean_content = func_role_text.replace('\n', '') + clean_content
        else: # Для operator и worker ролей: полностью удаляем маркеры
            clean_content = content
            for role in [operator_role_text, worker_role_text]: clean_content = clean_content.replace(role, '').strip()
        if not clean_content: continue
        # Определяем роль для API на основе простого чередования
        # Первое сообщение после system - user, следующее - assistant, и т.д.
        api_role = current_role # Переключаем роль для следующего сообщения
        current_role = "assistant" if current_role == "user" else "user"
        messages.append({"role": api_role, "content": clean_content})
    if not messages and prompt.strip(): # Если вообще нет сообщений, но промпт не пустой
        clean_content = prompt.strip()
        # Удаляем только operator и worker маркеры
        for role in [operator_role_text, worker_role_text]: clean_content = clean_content.replace(role, '').strip()
        messages.append({"role": "user", "content": clean_content})
    # Проверка на четность количества сообщений (включая системное)
    # Если нечетное - удаляем последнее сообщение
    if len(messages) % 2 != 0: removed_message = messages.pop(); let_log(f"Удалено последнее сообщение (нечетное количество): {removed_message['role']} - {removed_message['content'][:100]}...")
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
    # Проверяем количество не-системных сообщений
    non_system_messages = [msg for msg in base_messages if msg.get("role") != "system"]
    if len(non_system_messages) == 1: # Если только одно не-системное сообщение
        single_message = non_system_messages[0]
        content = single_message.get("content", "")
        # Проверяем, является ли это сообщение ответом функции (содержит func_text_for_parse)
        if global_state.start_dialog_command_name != '' and global_state.now_agent_id % 2 != 0 and func_text_for_parse in content:
            let_log("Обнаружено одно сообщение с ответом функции - выполняем замену")
            # Вырезаем func_text_for_parse из начала контента (как при множественных сообщениях)
            cleaned_content = content
            if content.startswith(func_text_for_parse): cleaned_content = content[len(func_text_for_parse):].strip()
            elif func_text_for_parse in content: cleaned_content = content.replace(func_text_for_parse, '', 1).strip()
            function_response_message = {"role": "function", "name": global_state.start_dialog_command_name, "content": cleaned_content} # Создаем сообщение с ответом функции
            for i, msg in enumerate(base_messages): # Заменяем исходное сообщение на ответ функции в base_messages
                if msg.get("role") != "system" and msg.get("content") == content: base_messages[i] = function_response_message; break
    # словарь команд для сессии (формат {'name':('desc', func)})
    try: now_commands = global_state.tools_commands_dict.get(sid, {})
    except Exception: now_commands = {}
    # для каждого сообщения заранее вычислим маркеры (если есть)
    markers_for_msg = []
    for m in base_messages:
        txt = m.get("content", "") if isinstance(m, dict) else ""
        matched = find_and_match_command(txt, now_commands)
        markers_for_msg.append(matched) # либо (found_key, content) либо None
    result = []
    i = 0
    while i < len(base_messages):
        msg = dict(base_messages[i]) # shallow copy
        role = (msg.get("role") or "").lower()
        content = msg.get("content", "")
        # проверяем, похоже ли текущее сообщение на ответ функции
        content_l = (content.lstrip() or "").lower()
        is_function_like = (role == "function") or content_l.startswith(func_text_for_parse)
        if is_function_like:
            prev_index = i - 1
            if prev_index >= 0: # берем маркер, найденный в предыдущем сообщении
                prev_marker = markers_for_msg[prev_index]
                if prev_marker:
                    found_key, args_str = prev_marker
                    try: # проверяем, что такой ключ есть в now_commands
                        if found_key in now_commands: # удаляем маркер из текста предыдущего сообщения
                            prev_msg_target = result[-1] if result else base_messages[prev_index]
                            prev_txt = prev_msg_target.get("content", "")
                            markers = _find_command_markers(prev_txt, now_commands, return_all=True, start_limit=None)
                            if markers: # Находим маркер с нужным ключом
                                for marker in markers:
                                    if marker['key'] == found_key: # Удаляем этот маркер
                                        new_prev_txt = prev_txt[:marker['start']] + prev_txt[marker['end']:]
                                        new_prev_txt = new_prev_txt.strip()
                                        if result: result[-1]["content"] = new_prev_txt
                                        else: base_messages[prev_index]["content"] = new_prev_txt
                                        break
                            else: # Fallback: если не нашли маркер
                                marker_token = "!!!" + found_key + "!!!"
                                if marker_token in prev_txt:
                                    new_prev_txt = prev_txt.replace(marker_token, "", 1).strip()
                                    if result: result[-1]["content"] = new_prev_txt
                                    else: base_messages[prev_index]["content"] = new_prev_txt
                            # создаем function-role сообщение с правильным именем и очищенным контентом
                            # вырезаем func_text_for_parse из начала контента
                            cleaned_content = content
                            if content_l.startswith(func_text_for_parse): cleaned_content = content[len(func_text_for_parse):].strip()
                            elif func_text_for_parse in content: cleaned_content = content.replace(func_text_for_parse, '', 1).strip()
                            function_message = {"role": "function", "name": found_key, "content": cleaned_content}
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
            if found_key in now_commands: # создаем сообщение с tool_calls вместо обычного assistant
                tool_call_id = f"call_{i}_{len(result)}"
                tool_call = {"id": tool_call_id, "type": "function", "function": {"name": found_key, "arguments": args_str}}
                # удаляем маркер из текста
                cleaned_content = content
                markers = _find_command_markers(content, now_commands, return_all=True, start_limit=None)
                if markers:
                    for marker in markers:
                        if marker['key'] == found_key:
                            cleaned_content = content[:marker['start']] + content[marker['end']:]
                            cleaned_content = cleaned_content.strip()
                            break
                else: # Fallback
                    marker_token = "!!!" + found_key + "!!!"
                    if marker_token in content: cleaned_content = content.replace(marker_token, "", 1).strip()
                # создаем сообщение с tool_calls
                tool_message = {"role": "assistant", "content": cleaned_content, "tool_calls": [tool_call]}
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
        tool_definition = {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": {"arguments": {"type": "string", "description": "Аргументы для команды в виде единой строки (весь текст, который должен идти после !!!)."}}, "required": ["arguments"]}}}
        tools_list.append(tool_definition)
    return tools_list

def parse_prompt_response(add_prompt, info, default_value=0):
    system_prompt = add_prompt + yes_no_instruction
    try: response = ask_model(info, system_prompt=system_prompt)
    except RuntimeError as e:
        if 'ContextOverflowError' in str(e):
            try: response = ask_model(text_cutter(info), system_prompt=system_prompt)
            except Exception: return default_value
        else: raise
    if not response or not response.strip(): return default_value
    response_clean = response.strip().lower()
    candidates = [yes_word.lower(), no_word.lower(), yes_word.lower().rstrip('.'), no_word.lower().rstrip('.')] # Варианты для сравнения (с точкой и без)
    best_match = None
    best_ratio = 0.0
    for cand in candidates:
        ratio = difflib.SequenceMatcher(None, response_clean, cand).ratio()
        if ratio > best_ratio: best_ratio = ratio; best_match = cand
    if best_ratio < 0.7: return default_value
    if best_match in (yes_word.lower(), yes_word.lower().rstrip('.')): return 1
    else: return 0

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
    dialog_messages = [m for m in messages if m['role'] not in ('system', 'tools')] # Отфильтровываем служебные роли для итерации по основному диалогу
    if not dialog_messages: return ""
    if tools_definition and unified_tags.get('tool_def_start'): # 2. Добавление описания инструментов (TOOLS) - в начале промпта
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
            if system_prompt and unified_tags.get('sys_start'): # Вставка системного промпта
                serialized_prompt.append(unified_tags['sys_start'])
                serialized_prompt.append(system_prompt)
                serialized_prompt.append(unified_tags['sys_end'])
            serialized_prompt.append(content)
            serialized_prompt.append(unified_tags['user_end'])
            first_user_processed = True
        # 3.2. Обработка остальных витков
        elif role == 'assistant' and first_user_processed: # Ответ ассистента
            serialized_prompt.append(unified_tags.get('assist_start', ''))
            serialized_prompt.append(content)
            serialized_prompt.append(unified_tags.get('assist_end', ''))
            serialized_prompt.append(unified_tags.get('eos', '')) # Закрывает виток
        elif role == 'user' and first_user_processed: # Новый виток пользователя
            serialized_prompt.append(unified_tags.get('bos', ''))
            serialized_prompt.append(unified_tags['user_start'])
            serialized_prompt.append(content)
            serialized_prompt.append(unified_tags['user_end'])
        elif role == 'tool_call' and unified_tags.get('tool_call_start'): # Вызов функции, сгенерированный моделью
            serialized_prompt.append(unified_tags['tool_call_start'])
            serialized_prompt.append(content)
            serialized_prompt.append(unified_tags['tool_call_end'])
            serialized_prompt.append(unified_tags.get('eos', ''))
        elif (role == 'tool' or role == 'function') and unified_tags.get('tool_result_start'): # Результат выполнения функции
            serialized_prompt.append(unified_tags['tool_result_start'])
            serialized_prompt.append(content)
            serialized_prompt.append(unified_tags['tool_result_end'])
            serialized_prompt.append(unified_tags.get('eos', ''))
    return "".join(serialized_prompt)

def detect_and_remove_loops(text, min_len=2, max_len=50, min_repeats=3, min_fraction=0.3):
    """
    Ищет в тексте повторяющиеся последовательности символов (паттерны).
    Если найден участок, где один и тот же паттерн повторяется подряд не менее min_repeats раз,
    и длина этого участка составляет не менее min_fraction от оставшейся части текста,
    то текст обрезается до начала первого повторения.
    """
    n = len(text)
    if n < min_len * min_repeats: return text
    for start in range(n - min_len * min_repeats + 1):
        for length in range(min_len, min(max_len, (n - start) // min_repeats + 1)):
            pattern = text[start:start+length]
            repeats = 1
            pos = start + length
            while pos + length <= n and text[pos:pos+length] == pattern:
                repeats += 1
                pos += length
                if repeats >= min_repeats:
                    total_loop_len = repeats * length
                    remaining_len = n - start
                    if total_loop_len / remaining_len >= min_fraction: return text[:start]
                    break
    return text

def remove_commands_roles(cleaned_text): # TODO: перепроверь работоспособность, учти отступы и что-то напоминающее команды
    if not filter_generations: return cleaned_text
    for var_content in clean_variables_content:
        if var_content:
            start_pos = cleaned_text.find(var_content)
            if start_pos != -1: cleaned_text = cleaned_text[:start_pos]; break
    markers = _find_command_markers(cleaned_text, global_state.tools_commands_dict.get(sid, {}), return_all=True, start_limit=None)
    if len(markers) >= 2: second_marker_start = markers[1]['start']; cleaned_text = cleaned_text[:second_marker_start]
    if remove_loops: cleaned_text = detect_and_remove_loops(cleaned_text)
    return cleaned_text

def split_text_with_cutting(text, min_chunk_percentage=0.8):
    if not isinstance(text, str) or not text.strip(): return None
    delimiters = ['\n\n', '\n', '. ', '! ', '? ', '; ', ': ', ' ', '']
    chunks = []
    current_pos = 0
    text_length = len(text)
    while current_pos < text_length:
        end_pos = int(min(current_pos + chunk_size, text_length))
        if end_pos == text_length:
            chunk = text[current_pos:end_pos]
            if chunk.strip(): chunks.append(chunk)
            break
        split_pos = None
        for delimiter in delimiters:
            candidate_pos = text.rfind(delimiter, current_pos, end_pos)
            if candidate_pos != -1:
                current_chunk_size = candidate_pos + len(delimiter) - current_pos
                if current_chunk_size >= chunk_size * min_chunk_percentage: split_pos = candidate_pos + len(delimiter); break
        if split_pos is None: split_pos = end_pos
        chunk = text[current_pos:split_pos]
        if chunk.strip(): chunks.append(chunk)
        current_pos = split_pos
    return chunks if chunks else None

def text_cutter(text, cut_message=False):
    let_log('ИТЕРАТИВНЫЙ КАТТЕР ВЫЗВАН')
    let_log(text)
    chunks_to_process = [text]
    summarized_chunks = []
    while chunks_to_process:
        current_chunk = chunks_to_process.pop(0)
        try:
            if cut_message: summarized_part = ask_model(current_chunk, system_prompt=cut_message_prompt)
            else: summarized_part = ask_model(current_chunk, system_prompt=summarize_prompt + '\n' + no_markdown_instruction)
            summarized_chunks.append(summarized_part)
            traceprint()
        except RuntimeError as e:
            if 'ContextOverflowError' in str(e):
                let_log(f'ошибка каттера (переполнение), делим кусок: {len(current_chunk)=}')
                let_log(e)
                text2 = current_chunk[len(current_chunk) // 2:]
                try:
                    split_pos = min(text2.find('\n'), text2.find('. '))
                    if split_pos == -1: split_pos = text2.find(' ')
                    if split_pos != -1: text2 = text2[split_pos + 2:]
                except Exception as split_e:
                    traceprint()
                    let_log(f"Ошибка при поиске точки разделения: {split_e}")
                    pass
                text1 = current_chunk[:current_chunk.find(text2)]
                if text2: chunks_to_process.insert(0, text2)
                if text1: chunks_to_process.insert(0, text1)
            else: sys.exit(1)
        except Exception as e:
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
    input_info_loaders = {}# Пытаемся импортировать модуль info_loaders
    global info_loaders
    try: import info_loaders; let_log("✅ Модуль info_loaders успешно импортирован")
    except Exception as e: let_log(f"❌ ФАТАЛЬНО: Не удалось импортировать модуль info_loaders: {e}"); return input_info_loaders
    for ext, func_name in info_loaders_names.items(): # Загружаем только рабочие обработчики
        try: # Получаем обработчик из модуля
            handler = getattr(info_loaders, func_name)
            if not callable(handler): let_log(f"⚠ '{func_name}' для .{ext} не является вызываемым объектом. Пропускаем."); continue # Проверяем что это функция/метод
            sig_params = list(inspect.signature(handler).parameters.values()) # Проверяем сигнатуру (должен принимать хотя бы 1 аргумент)
            if len(sig_params) < 1: let_log(f"⚠ '{func_name}' имеет неверную сигнатуру для .{ext}. Пропускаем."); continue
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
    if not input_info_loaders: # Проверяем, загружены ли обработчики
        let_log("⚠ Обработчики файлов не загружены. Загружаем...")
        try: load_info_loaders(default_handlers_names)
        except Exception as e:
            let_log(f"❌ Ошибка загрузки обработчиков файлов: {e}")
            send_output_message(text="Система обработки файлов недоступна", command='warning')
            return []
    all_results = [] # Только успешные результаты
    processed_chunks = [] # Собираем все чанки из всех файлов
    if not files_list: let_log("Список файлов для загрузки пуст"); return all_results
    for filename in files_list:
        file_basename = os.path.basename(filename)
        let_log(f"Обработка файла: {file_basename}")
        try: # 1. Проверка существования файла
            if not os.path.exists(filename): raise FileNotFoundError(f"Файл не существует: {filename}")
            _, file_extension = os.path.splitext(filename) # 2. Получение расширения
            extension = file_extension[1:].lower() if file_extension else ''
            if not extension: raise ValueError(f"Файл не имеет расширения: {filename}")
            # 3. Проверка наличия обработчика
            if extension not in input_info_loaders: # Пытаемся обработать как текстовый файл
                handler = info_loaders.process_unknown
                let_log(f"  Для расширения .{extension} нет обработчика, попытка открыть как текст")
            else: handler = input_info_loaders[extension]
            file_size = os.path.getsize(filename) # 4. Получаем размер файла
            result = read_cache() # 5. Использование кэша (если включено)
            if result == [False]: # Вызов обработчика
                let_log(f"  Вызов обработчика {handler.__name__}...")
                result_content = handler(filename, input_info_loaders)
                write_cache(result_content)
            else: let_log(f"  Используется кэшированный результат"); result_content = result[1]
            # 6. Обработка результата (разбиение на чанки и сбор для последующего сохранения)
            if file_extension[1:].lower() == 'zip' and isinstance(result_content, list): # Обработка ZIP-архива (включая вложенные)
                for file_data in result_content:
                    if file_data['type'] in ['file', 'unsupported']:
                        content = file_data['content']
                        if isinstance(content, str):
                            let_log(f"  Разбиение файла из ZIP: {file_data['filename']}")
                            chunks = split_text_with_cutting(content)
                            if chunks:
                                for t, chunk in enumerate(chunks): processed_chunks.append({'chunk': chunk, 'metadata': {'name': f"{filename}/{file_data['filename']}", 'part': t + 1, 'source': 'zip'}})
            else: # Обработка обычного файла
                if isinstance(result_content, str):
                    let_log(f"  Разбиение файла на чанки...")
                    chunks = split_text_with_cutting(result_content)
                    if chunks:
                        for t, chunk in enumerate(chunks): processed_chunks.append({'chunk': chunk, 'metadata': {'name': filename, 'part': t + 1, 'source': 'file'}})
            # Добавляем в список успешных
            all_results.append({'filename': filename, 'content': result_content, 'extension': extension, 'size': file_size})
            let_log(f"✅ Файл {file_basename} успешно обработан")
        except (FileNotFoundError, ValueError, KeyError) as e: # Ожидаемые ошибки - пропускаем файл
            let_log(f"⏭ Пропускаем {file_basename}: {e}")
            send_output_message(text=f"Пропущен файл {file_basename}: {e}", command='info')
        except Exception as e: # Неожиданные ошибки - пропускаем с логированием
            error_msg = f"Ошибка обработки {file_basename}: {type(e).__name__}"
            let_log(f"⏭ Пропускаем {file_basename} из-за ошибки: {error_msg}")
            let_log(f"  Детали: {str(e)[:200]}")
            send_output_message(text=f"Ошибка при обработке {file_basename}", command='warning')
    if processed_chunks: # 7. Сохраняем ВСЕ чанки из ВСЕХ успешно обработанных файлов в базу
        let_log(f"Сохраняем {len(processed_chunks)} чанков в базу...")
        for chunk_info in processed_chunks:
            set_common_save_id()
            coll_exec(action="add", coll_name="user_collection", ids=[get_common_save_id()], embeddings=[get_embs(chunk_info['chunk'])], metadatas=[chunk_info['metadata']], documents=[chunk_info['chunk']])
        let_log(f"✅ Все чанки сохранены в базу")
    else: let_log("⚠ Нет чанков для сохранения в базу")
    if all_results: # 8. Аннотация только успешно обработанных файлов
        try:
            annotation_text = ""
            for result in all_results:
                if isinstance(result['content'], str): annotation_text += f"\n\n--- {os.path.basename(result['filename'])} ---\n{result['content'][:5000]}"
                elif isinstance(result['content'], list):
                    for file_data in result['content']:
                        if isinstance(file_data.get('content'), str): annotation_text += f"\n\n--- {os.path.basename(result['filename'])}/{file_data['filename']} ---\n{file_data['content'][:5000]}"
            if annotation_text.strip():
                try: global_state.summ_attach = annotation_available_prompt + ask_model(annotation_text, system_prompt=summarize_text_some_phrases)
                except:
                    try: global_state.summ_attach = annotation_available_prompt + ask_model(text_cutter(annotation_text), system_prompt=summarize_text_some_phrases)
                    except: global_state.summ_attach = annotation_failed_text
            else: global_state.summ_attach = annotation_failed_text
        except Exception as e: let_log(f"Ошибка при создании аннотации: {e}"); global_state.summ_attach = annotation_failed_text
    else: let_log("Нет успешно обработанных файлов для аннотации"); global_state.summ_attach = annotation_failed_text
    try: # 9. Очистка ресурсов - выгружаем модель обработки изображений из ОЗУ
        if hasattr(info_loaders, 'cleanup_image_models'): info_loaders.cleanup_image_models()
    except Exception as e: let_log(f"⚠ Ошибка при очистке ресурсов: {e}")
    return all_results

@no_cache
def set_common_save_id(): global_state.common_save_id += 1

def get_common_save_id(): return str(global_state.common_save_id)

@no_cache
def reset_common_save_id(): global_state.common_save_id = 1

@no_cache
def down_hierarchy(): # Добавить новый уровень иерархии (делегирование)
    parts = global_state.now_try.strip('/').split('/')
    if not parts or parts[0] == '': new_level = 1 # Определяем номер нового уровня
    else: # Находим последнюю пару и увеличиваем уровень
        last_part = parts[-1]
        if ':' in last_part: # Получаем часть до двоеточия
            level_part = last_part.split(':')[0]
            if level_part == '': last_level = 0 # Если часть перед двоеточием пустая, уровень = 0
            else:
                try: last_level = int(level_part)
                except ValueError: last_level = 0
        else: # Если двоеточия нет, пробуем преобразовать всю часть в число
            try: last_level = int(last_part) if last_part.isdigit() else 0
            except ValueError: last_level = 0
        new_level = last_level + 1
    # Добавляем новый уровень с исполнителем = 0
    if global_state.now_try == '/' or global_state.now_try == '': global_state.now_try = f"/{new_level}:0"
    else: global_state.now_try += f"/{new_level}:0"
    let_log(f"[HIERARCHY] Down: {global_state.now_try}")

@no_cache
def up_hierarchy(): # Подняться на уровень выше
    if global_state.now_try == '/' or global_state.now_try == '': return  # Уже на корневом уровне
    parts = global_state.now_try.strip('/').split('/')
    if len(parts) > 1: # Удаляем последнюю пару
        parts = parts[:-1]
        if parts: global_state.now_try = '/' + '/'.join(parts)
        else: global_state.now_try = '/'
    else: global_state.now_try = '/'
    let_log(f"[HIERARCHY] Up: {global_state.now_try}")

@no_cache
def next_executor(): # Создать/пересоздать исполнителя на текущем уровне
    if global_state.now_try == '/' or global_state.now_try == '': global_state.now_try = f"/1:1"
    else:
        parts = global_state.now_try.strip('/').split('/')
        last_part = parts[-1]
        if ':' in last_part:
            level, executor = last_part.split(':')
            new_executor = int(executor) + 1
            parts[-1] = f"{level}:{new_executor}"
        else: # Формат без : (старый формат)
            level = int(last_part) if last_part.isdigit() else 1
            parts[-1] = f"{level}:1"
        global_state.now_try = '/' + '/'.join(parts)
    let_log(f"[HIERARCHY] Next executor: {global_state.now_try}")
    return get_executor_number()

def get_executor_number(): # Получить номер текущего исполнителя
    if global_state.now_try == '/' or global_state.now_try == '': return 0
    parts = global_state.now_try.strip('/').split('/')
    last_part = parts[-1]
    if ':' in last_part:
        _, executor = last_part.split(':')
        try: return int(executor) if executor != '' else 0
        except ValueError: return 0
    return 0

def get_level(): # Получить номер текущего уровня
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

def get_operator_id(): # ID оператора (без номера исполнителя)
    if global_state.now_try == '/' or global_state.now_try == '': return '/'
    parts = global_state.now_try.strip('/').split('/')
    clean_parts = [] # Преобразуем каждый уровень в формат без исполнителя
    for part in parts:
        if ':' in part: level, _ = part.split(':'); clean_parts.append(level)
        else: clean_parts.append(part)
    return '/' + '/'.join(clean_parts)

def get_executor_id():
    if get_executor_number() == 0: return None
    return global_state.now_try

def save_emb_dialog(tag, dialog_type='operator', result_text='', result=False):
    """
    Сохраняет диалог с новой системой ID
    dialog_type: 'operator' или 'executor'
    overwrite: True - перезаписать существующие записи, False - добавить новые
    """
    let_log(f"\n{'='*60}")
    let_log(f"Current ID: {global_state.now_try}")
    def _parse_dialog_to_messages(t): # Парсит текст диалога на отдельные сообщения
        messages_list = []
        roles_to_find = [('operator', operator_role_text), ('worker', worker_role_text), ('function', func_role_text)]
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
                if next_pos > start_pos: end_pos = next_pos; break
            message_full_text = t[start_pos:end_pos]
            messages_list.append({'role': role_type, 'content': message_full_text})
        let_log(f"Распарсено {len(messages_list)} сообщений")
        return messages_list
    def _get_content_without_role(message_full_text):
        """Извлекает контент, удаляя маркер роли."""
        for role_marker in [operator_role_text, worker_role_text, func_role_text]:
            if message_full_text.startswith(role_marker): content = message_full_text.replace(role_marker, '', 1); return content 
        return message_full_text 
    def _create_numbered_messages_text(msgs): # Создает нумерованный текст БЕЗ ролей для LLM
        numbered_text = ""
        for i, msg in enumerate(msgs, 1):
            content_only = _get_content_without_role(msg['content'])
            if len(content_only) > 300: message_preview = content_only[:300] + "..."
            else: message_preview = content_only
            numbered_text += f"\n{i}. {message_preview}"
        return numbered_text
    def _parse_ranges_from_response(response): # Парсит ответ модели в список диапазонов.
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
            for i in range(start - 1, end): cleaned_messages.append(_get_content_without_role(msgs[i]['content']))
            group_text = "\n".join(cleaned_messages)
            groups.append({'global_start': offset + start, 'global_end': offset + end, 'text': group_text})
        # Fallback: если LLM не сгруппировал все сообщения, то одиночные сообщения тоже сохраняем
        if not groups:
            for i, msg in enumerate(msgs, 1): groups.append({'global_start': offset + i, 'global_end': offset + i, 'text': _get_content_without_role(msg['content'])})
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
            # Оцениваем размер промпта: системный (группировка) + пользовательский (numbered_text)
            system_prompt = grouping_prompt_1 + grouping_prompt_2
            estimated_tokens = (len(system_prompt) + len(numbered_text)) * get_text_tokens_coefficient()
            if estimated_tokens <= (token_limit - 1000): return batch_size
        return 1
    def _process_messages_batch(msgs, batch_offset):
        """Обрабатывает один батч сообщений."""
        let_log(f"  Внутри батча (смещение: {batch_offset}): Создание промпта и вызов ask_model...")
        numbered_text = _create_numbered_messages_text(msgs)
        # Изменено: grouping_prompt_1 и grouping_prompt_2 теперь в system_prompt
        system_prompt = grouping_prompt_1 + grouping_prompt_2
        try:
            # Используем ask_model с system_prompt
            response = ask_model(numbered_text, system_prompt=system_prompt)
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
        if not result and global_state.need_owerwrite_operator: need_ov = True; global_state.need_owerwrite_operator = False
    elif dialog_type == 'executor':
        executor_id = get_executor_id()
        if not executor_id: let_log("Executor not created yet, skipping save"); return
        doc_id = executor_id
        let_log(f"Executor ID: {doc_id}")
        _, history_to_save = get_chat_context(global_state.conversations)
        if global_state.need_owerwrite_executor: need_ov = True; global_state.need_owerwrite_executor = False
    # --- 2. Если нужно перезаписать - удаляем старые записи ---
    let_log(f"SAVE_EMB_DIALOG: tag={tag}, type={dialog_type}, need_ov={need_ov}")
    if need_ov:
        let_log('\nПЕРЕЗАПИСЬ ЧАТА\n')
        existing_ids = coll_exec(action="query", coll_name="milana_collection", query_embeddings=[[]], filters={"doc_id": doc_id, "dialog_type": dialog_type, "result": result}, fetch="ids", n_results=100, flatten=True)
        if existing_ids: coll_exec(action="delete", coll_name="milana_collection", ids=existing_ids); let_log(f"Deleted {len(existing_ids)} old records for {doc_id}")
    else: let_log('\nНЕТ ПЕРЕЗАПИСИ ЧАТА\n')
    if result:
        if result_text == '': let_log("Текст результата пустой"); return
        let_log("Сохранение результата отдельно")
        set_common_save_id()
        metadata = {"doc_id": doc_id, "dialog_type": dialog_type, "done": tag, "result": result, "hierarchy": global_state.now_try, "timestamp": time.time()}
        coll_exec(action="add", coll_name="milana_collection", ids=[get_common_save_id()], embeddings=[get_embs(result_text)], metadatas=[metadata], documents=[result_text])
        return
    # --- 4. Парсинг и группировка диалога ---
    messages = _parse_dialog_to_messages(history_to_save)
    if not messages: let_log("Не удалось распарсить сообщения."); return
    all_groups = []
    total_messages = len(messages)
    current_index = 0
    chunks_info = []
    # Проверяем, помещаются ли все сообщения в один запрос
    numbered_text_all = _create_numbered_messages_text(messages)
    system_prompt_all = grouping_prompt_1 + grouping_prompt_2
    estimated_tokens = (len(system_prompt_all) + len(numbered_text_all)) * get_text_tokens_coefficient()
    if estimated_tokens <= (get_token_limit() - 1000): let_log("Все сообщения помещаются в один запрос"); all_groups = _process_messages_batch(messages, 0)
    else:
        let_log(f"Сообщения не помещаются (оценка: {estimated_tokens:.0f} токенов), разбиваем на батчи")
        batch_number = 0
        while current_index < total_messages:
            batch_size = _calculate_batch_size(messages, current_index)
            batch_messages = messages[current_index:current_index + batch_size]
            chunks_info.append({'batch_number': batch_number + 1, 'start_idx': current_index + 1, 'end_idx': min(current_index + batch_size, total_messages)})
            let_log(f"\n--- Обработка батча {batch_number + 1} (сообщения {current_index + 1}-{current_index + batch_size}) ---")
            batch_groups = _process_messages_batch(batch_messages, current_index)
            all_groups.extend(batch_groups)
            current_index += batch_size
            batch_number += 1
        if chunks_info: let_log(f"--- ИНФОРМАЦИЯ О ЧАНКАХ (для LLM) ---\n" + "\n".join([f"Чанк {c['batch_number']}: сообщения {c['start_idx']}-{c['end_idx']}" for c in chunks_info]))
    # --- 5. Сохранение всех групп ---
    let_log(f"\n--- Сохранение {len(all_groups)} групп для {dialog_type} ---")
    for i, group in enumerate(all_groups):
        set_common_save_id()
        metadata = {"doc_id": doc_id, "dialog_type": dialog_type, "done": tag, "result": result, "group_index": i, "total_groups": len(all_groups), "hierarchy": global_state.now_try, "timestamp": time.time()}
        let_log(f"\n### ГРУППА {i+1} (Сохраняемый документ {get_common_save_id()}) ###")
        let_log(f"Диапазон: {group['global_start']}-{group['global_end']}")
        let_log(f"СОХРАНЯЕМЫЙ ТЕКСТ (первые 500 символов):\n---START---\n{group['text'][:500]}...\n---END---")
        coll_exec(action="add", coll_name="milana_collection", ids=[get_common_save_id()], embeddings=[get_embs(group['text'])], metadatas=[metadata], documents=[group['text']])
    let_log(f"Всего сохранено {len(all_groups)} групп сообщений для {doc_id}")
    let_log(f"{'='*60}")

@cacher
def gigo(base_task: str, settings: dict = None) -> str:
    """
    Классический GIGO (версия may_fixes до advanced gigo):
    dreamer/realist/critic → plan с no_markdown_instruction и числом пунктов плана.
    Librarian намеренно закомментирован (как в эталоне).
    """
    if not use_gigo:
        try: return gigo_return_1 + base_task
        except NameError: return base_task
    # librarian в классическом gigo закомментирован — так и надо
    # try: questions = ask_model(base_task + global_state.summ_attach, system_prompt=gigo_questions)
    # except RuntimeError as e:
    #     if 'ContextOverflowError' in str(e):
    #         base_task = text_cutter(base_task)
    #         questions = ask_model(text_cutter(base_task + global_state.summ_attach), system_prompt=gigo_questions)
    #     else: raise
    # additional_info = librarian(questions)
    # if additional_info != found_info_1: additional_info = '\n' + gigo_found_info + '\n' + additional_info
    # else: additional_info = ''; let_log(found_info_1)
    additional_info = ''
    minds_text = ''
    minds = []
    roles = [gigo_dreamer, gigo_realist, gigo_critic]
    ents_roles = ', '.join(roles) + '\n'
    role_notes = [gigo_dreamer_note, gigo_realist_note, gigo_critic_note]
    for role, role_note in zip(roles, role_notes):
        try: minds.append(ask_model(base_task + additional_info, system_prompt=gigo_role_answer_1 + role + role_note + gigo_role_answer_2 + '\n' + no_markdown_instruction))
        except RuntimeError as e:
            if 'ContextOverflowError' in str(e): minds.append(ask_model(text_cutter(base_task + additional_info), system_prompt=gigo_role_answer_1 + role + role_note + gigo_role_answer_2 + '\n' + no_markdown_instruction))
            else: raise
    for role, mind in zip(roles, minds):
        minds_text += worker_role_text + mind
        if role == roles[-1]: minds_text += '\n' * 2 + gigo_final_role_2 + operator_role_text
        else:
            minds_text += operator_role_text + gigo_next_role + role
            if len(roles) != 1 and role == roles[-2]: minds_text += gigo_final_role
    # Пункты плана: gigo_plan_items (настройка GIGO), иначе number_of_plan_items
    n_items = 0
    try:
        if gigo_plan_items and int(gigo_plan_items) > 0:
            n_items = int(gigo_plan_items)
        elif getattr(global_state, 'number_of_plan_items', 0) and int(global_state.number_of_plan_items) > 0:
            n_items = int(global_state.number_of_plan_items)
    except Exception:
        n_items = 0
    try:
        plan_num_prefix = gigo_make_plan_num
    except NameError:
        plan_num_prefix = '\nPlan items required: '
    num_plan_items = (plan_num_prefix + str(n_items)) if n_items > 0 else ''
    try: plan = ask_model(system_role_text + gigo_make_plan_1 + no_markdown_instruction + num_plan_items + gigo_make_plan_2 + ents_roles + gigo_return_1 + base_task + additional_info + minds_text)
    except RuntimeError as e:
        if 'ContextOverflowError' in str(e):
            minds_text = ''
            for role, mind in zip(roles, minds):
                minds_text += worker_role_text + text_cutter(mind)
                if role == roles[-1]: minds_text += '\n' * 2 + gigo_final_role_2 + operator_role_text
                else:
                    minds_text += operator_role_text + gigo_next_role + role
                    if len(roles) != 1 and role == roles[-2]: minds_text += gigo_final_role
            plan = ask_model(system_role_text + gigo_make_plan_1 + no_markdown_instruction + num_plan_items + gigo_make_plan_2 + ents_roles + gigo_return_1 + base_task + text_cutter(additional_info) + minds_text)
        else: raise
    return gigo_return_1 + base_task + '\n' + gigo_return_2 + plan

def gigo_adv(task: str, settings: dict = None) -> str:
    """Продвинутый GIGO (идеи, фильтр, dreamer/realist/critic, синтез)."""
    if not use_gigo: return gigo_label_task + task
    # 1. Анализ намерения
    try: intention_text = ask_model(task, system_prompt=gigo_intention_prompt + '\n' + warn_command_text_8 + '\n' + global_state.tools_str)
    except RuntimeError as e:
        if 'ContextOverflowError' in str(e): intention_text = ask_model(text_cutter(task), system_prompt=gigo_intention_prompt)
        else: raise
    # 2. Библиотекарь
    library = ""
    if gigo_use_librarian:
        try: questions = ask_model(task + global_state.summ_attach, system_prompt=gigo_questions)
        except RuntimeError as e:
            if 'ContextOverflowError' in str(e): questions = ask_model(text_cutter(task + global_state.summ_attach), system_prompt=gigo_questions)
            else: raise
        if questions and questions.strip():
            library = librarian(questions)
            if library and library != found_info_1:
                if len(library) > 3000: library = text_cutter(library)
                library = gigo_found_info + '\n' + library
            else: library = ""
    # 3. Генерация идей
    role = ""
    if gigo_use_random_roles: # TODO: опционально случайная строка для множества ролей
        try: role = ask_model(task, system_prompt=gigo_role_generation_prompt).strip()
        except RuntimeError as e:
            if 'ContextOverflowError' in str(e): role = ask_model(text_cutter(task), system_prompt=gigo_role_generation_prompt).strip()
            else: raise
    ideas = []
    for _ in range(gigo_idea_count):
        entropy = ""
        if gigo_use_entropy: entropy = secrets.token_hex(32)
        
        # нужна переработка фантазии в реализм иначе будет получаться бред
        idea_prompt = gigo_idea_generation_prompt_1 + (role if role else 'expert') + gigo_idea_generation_prompt_2
        if entropy: idea_prompt += '\n' + gigo_entropy_instruction + ' ' + entropy
        idea_input = gigo_label_task + task + '\n' + gigo_label_intention + intention_text + '\n'
        if library: idea_input += gigo_label_additional_info + library + '\n'
        try: idea = ask_model(idea_input, system_prompt=idea_prompt)
        except RuntimeError as e:
            if 'ContextOverflowError' in str(e): idea = ask_model(text_cutter(idea_input), system_prompt=idea_prompt)
            else: raise
        ideas.append(idea)
    # 4. Фильтрация
    if gigo_use_filter and len(ideas) > 1:
        ideas_text = ''
        for idx, idea in enumerate(ideas): ideas_text += gigo_label_idea + str(idx+1) + gigo_label_colon + idea + '\n'
        try: filter_response = ask_model(ideas_text, system_prompt=gigo_filter_ideas_prompt)
        except RuntimeError as e:
            if 'ContextOverflowError' in str(e): filter_response = ask_model(text_cutter(ideas_text), system_prompt=gigo_filter_ideas_prompt)
            else: raise
        if filter_response.strip():
            selected = []
            for part in filter_response.split(','):
                part = part.strip()
                if part.isdigit():
                    idx = int(part) - 1
                    if 0 <= idx < len(ideas): selected.append(idx)
            if selected: ideas = [ideas[i] for i in selected]
    # 5. Развитие идей
    developed_ideas = []
    for idea in ideas: # TODO: верни старую последовательность сообщений
        versions = []
        try: dream = ask_model(idea, system_prompt=gigo_dreamer_prompt)
        except RuntimeError as e:
            if 'ContextOverflowError' in str(e): dream = ask_model(text_cutter(idea), system_prompt=gigo_dreamer_prompt)
            else: raise
        versions.append((gigo_dreamer, dream))
        try: real = ask_model(idea, system_prompt=gigo_realist_prompt)
        except RuntimeError as e:
            if 'ContextOverflowError' in str(e): real = ask_model(text_cutter(idea), system_prompt=gigo_realist_prompt)
            else: raise
        versions.append((gigo_realist, real))
        try: critic = ask_model(idea, system_prompt=gigo_critic_prompt)
        except RuntimeError as e:
            if 'ContextOverflowError' in str(e): critic = ask_model(text_cutter(idea), system_prompt=gigo_critic_prompt)
            else: raise
        versions.append((gigo_critic, critic))
        if not versions: developed_ideas.append(idea); continue
        synthesis_input = gigo_label_original_idea + idea + '\n\n'
        for name, text in versions: synthesis_input += name.capitalize() + ':\n' + text + '\n\n'
        try: synthesis = ask_model(synthesis_input, system_prompt=gigo_synthesize_prompt)
        except RuntimeError as e:
            if 'ContextOverflowError' in str(e): synthesis = ask_model(text_cutter(synthesis_input), system_prompt=gigo_synthesize_prompt)
            else: raise
        developed_ideas.append(synthesis)
    # 6. Выбор лучшей
    if len(developed_ideas) > 1:
        ideas_text = ''
        for idx, idea in enumerate(developed_ideas): ideas_text += gigo_label_idea + str(idx+1) + gigo_label_colon + idea + '\n'
        try: choice_response = ask_model(ideas_text, system_prompt=gigo_choose_best_prompt)
        except RuntimeError as e:
            if 'ContextOverflowError' in str(e): choice_response = ask_model(text_cutter(ideas_text), system_prompt=gigo_choose_best_prompt)
            else: raise
        best_idx = 0
        if choice_response.strip().isdigit():
            idx = int(choice_response.strip()) - 1
            if 0 <= idx < len(developed_ideas): best_idx = idx
        best_idea = developed_ideas[best_idx]
    else: best_idea = developed_ideas[0] if developed_ideas else ""
    # 7. Построение ответа)
    if gigo_plan_items > 0: plan_items = gigo_plan_items
    else: plan_items = global_state.number_of_plan_items if global_state.number_of_plan_items > 0 else 5
    build_prompt = gigo_build_answer_prompt_1 + str(plan_items) + gigo_build_answer_prompt_2
    answer_input = gigo_label_task + task + '\n' + gigo_label_intention + intention_text + '\n' + gigo_label_best_idea + best_idea
    try: answer = ask_model(answer_input, system_prompt=build_prompt + '\n' + no_markdown_instruction)
    except RuntimeError as e:
        if 'ContextOverflowError' in str(e): answer = ask_model(text_cutter(answer_input), system_prompt=build_prompt + '\n' + no_markdown_instruction)
        else: raise
    return answer

@cacher
def critic(task: str, result: str) -> int | str:
    """
    Оценивает результат.
    - Возвращает 1, если результат приемлем, критик не уверен или произошла ошибка.
    - Возвращает строку с новой, доработанной задачей для исполнителя.
    """
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
    # --- Этап 1: Декомпозиция задачи на критерии ---
    try: system_prompt1 = prompt_decomposition_1 + "\n" + prompt_decomposition_2; criteria_text = ask_model(task, system_prompt=system_prompt1)
    except: task_cut = text_cutter(task); criteria_text = ask_model(task_cut, system_prompt=system_prompt1)
    if not criteria_text or not criteria_text.strip(): return 1
    # --- Этап 2: Оценка по критериям ---
    user_prompt2 = f"{prompt_evaluation_2}\n{task}\n{prompt_evaluation_3}\n{result}\n{prompt_evaluation_4}\n{criteria_text}"
    system_prompt2 = prompt_evaluation_1 + "\n" + prompt_evaluation_5
    try: evaluation_text = ask_model(user_prompt2, system_prompt=system_prompt2)
    except: user_prompt2_cut = text_cutter(user_prompt2); evaluation_text = ask_model(user_prompt2_cut, system_prompt=system_prompt2)
    if not evaluation_text or not evaluation_text.strip(): return 1
    # --- Этап 2.5: Обращение к Библиотекарю (закомментировано) ---
    """
    # Формируем запрос к модели для генерации вопросов к библиотекарю
    system_prompt_librarian = prompt_librarian_questions_1 + "\n" + prompt_librarian_questions_4
    user_prompt_librarian = f"{prompt_librarian_questions_2}\n{task}\n{prompt_librarian_questions_3}\n{result}\nEvaluation report:\n{evaluation_text}"
    try: questions_text = ask_model(user_prompt_librarian, system_prompt=system_prompt_librarian)
    except: questions_text = ask_model(text_cutter(user_prompt_librarian), system_prompt=system_prompt_librarian)
    librarian_context = ""
    if questions_text and questions_text.strip():
        print("Критик -> Библиотекарь: Запрос на проверку информации...")
        librarian_answers = librarian(questions_text)
        if librarian_answers and librarian_answers.strip():
            librarian_context = f"{prompt_decision_librarian_context}{librarian_answers}\n"
    """
    # --- Этап 3 и 4: Принятие решения ---
    user_prompt3 = f"{prompt_decision_2}\n{task}\n{prompt_decision_3}\n{result}\n{prompt_decision_4}\n{evaluation_text}"
    # Если блок выше раскомментирован, можно добавить librarian_context:
    # user_prompt3 += f"\n{librarian_context}"
    system_prompt3 = prompt_decision_1 + "\n" + prompt_decision_5
    if use_magical_prompt: system_prompt3 += prompt_critic_principles
    try: decision_response = ask_model(user_prompt3, system_prompt=system_prompt3)
    except: user_prompt3_cut = text_cutter(user_prompt3); decision_response = ask_model(user_prompt3_cut, system_prompt=system_prompt3)
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
    found_commands_set = set()
    # Регулярное выражение для поиска "слов", которые могут быть именами команд
    # (буквы, цифры и знак подчеркивания).
    word_pattern = re.compile(r'[\w_]+')
    # Итерируемся по каждому слову, найденному в тексте.
    for match_obj in word_pattern.finditer(text):
        word_from_text = match_obj.group(0)
        # Используем difflib, чтобы найти наилучшее совпадение для этого конкретного слова.
        close_matches = difflib.get_close_matches(word_from_text, available_commands, n=1, cutoff=cutoff)
        if close_matches: matched_command = close_matches[0]; found_commands_set.add(matched_command)
    return list(found_commands_set)

def _find_command_markers(text, commands_dict, return_all=False, start_limit=None):
    """
    Ищет маркеры команд вида !!!name!!! (с вариациями символов '!' и '¡').
    Возвращает:
        если return_all=False: кортеж (key, content, start, end) для первого найденного маркера или None.
        если return_all=True: список словарей с ключами 'key', 'content', 'start', 'end', 'full_match'.
    Параметр start_limit ограничивает поиск маркеров, начинающихся не дальше этой позиции.
    """
    if not text or not commands_dict: return [] if return_all else None
    exclamation_chars = {'!', '¡'}
    pattern = r'[!¡]{1,3}\s*([\w\s]+?)\s*[!¡]{1,3}'
    markers = []
    for match in re.finditer(pattern, text):
        start = match.start()
        if start_limit is not None and start > start_limit: continue
        raw_name = match.group(1).strip()
        # Нормализуем имя: заменяем пробелы на подчёркивания, приводим к нижнему регистру
        normalized_name = re.sub(r'\s+', '_', raw_name).lower()
        # Поиск соответствия в commands_dict
        found_key = None
        # 1. Точное совпадение по нормализованному имени
        for key in commands_dict:
            if normalized_name == key.lower(): found_key = key; break
        # 2. Нечёткое сравнение, если точного нет
        if not found_key:
            best_match = None
            best_ratio = 0.0
            threshold = 0.8
            for key in commands_dict:
                key_lower = key.lower()
                if abs(len(normalized_name) - len(key_lower)) > 2: continue
                ratio = difflib.SequenceMatcher(None, normalized_name, key_lower).ratio()
                if ratio > best_ratio and ratio >= threshold: best_ratio = ratio; best_match = key
            if best_match: found_key = best_match
        if not found_key: continue
        # Определяем конец маркера (позиция после закрывающих символов)
        end = match.end()
        # Содержимое после маркера (до следующего маркера или конца текста)
        next_match = re.search(pattern, text[end:])
        if next_match: content_end = end + next_match.start()
        else: content_end = len(text)
        content = text[end:content_end].strip()
        marker_info = {'key': found_key, 'content': content, 'start': start, 'end': end, 'full_match': text[start:end]}
        if not return_all: return (found_key, content, start, end)
        markers.append(marker_info)
    if return_all: return markers
    return None

def find_and_match_command(text, commands_dict): # Ищет в тексте первый маркер команды, начинающийся в первых 5 символах. Возвращает (найденный_ключ, содержимое_после_маркера) или None.
    result = _find_command_markers(text, commands_dict, return_all=False, start_limit=5)
    if result: key, content, _, _ = result;  return (key, content)
    return None

def _find_formatting_ranges(text):
    """
    Определяет интервалы текста, которые находятся внутри markdown-форматирования
    (жирный **, курсив *, подчёркивание _, зачёркнутый ~~, инлайн-код `).
    Учитывает экранирование обратным слешем и старается не ловить звёздочки внутри слов.
    Возвращает список кортежей (start, end) для всех таких участков.
    """
    n = len(text)
    stack = []
    ranges = []
    i = 0
    # Множество символов, которые могут быть частью слова (буквы, цифры, подчёркивание)
    word_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_')
    # Все возможные маркеры с их длиной и признаком, нужно ли проверять границы слов
    marker_defs = [
        ('**', 2, False),  # жирный — два символа, не бывает внутри слов
        ('__', 2, False),
        ('~~', 2, False),
        ('*', 1, True),    # курсив — может быть внутри слова, нужна проверка
        ('_', 1, True),    # подчёркивание — аналогично
        ('`', 1, False),   # код — внутри слов обычно не используется как маркер, но бывает; оставим без проверки
    ]
    while i < n:
        # Экранирование: пропускаем следующий символ
        if text[i] == '\\' and i + 1 < n: i += 2; continue
        matched = False
        for mark, length, check_word in marker_defs:
            if i + length <= n and text[i:i+length] == mark:
                # Для маркеров, требующих проверки на границы слов
                if check_word:
                    prev_char = text[i-1] if i > 0 else None
                    next_char = text[i+length] if i+length < n else None
                    # Если до или после буква/цифра/подчёркивание — считаем, что это не маркер, а часть слова
                    if (prev_char and prev_char in word_chars) or (next_char and next_char in word_chars):
                        i += 1
                        matched = True
                        break
                # Решаем, открывающий или закрывающий
                if stack and stack[-1][1] == mark:
                    # Закрываем последний такой же
                    start_pos, _ = stack.pop()
                    ranges.append((start_pos, i + length))
                else: stack.append((i, mark)) # Открываем новый
                i += length
                matched = True
                break
        if not matched: i += 1
    # Оставляем только корректно закрытые пары; незакрытые игнорируем
    # Объединяем пересекающиеся интервалы
    if ranges:
        ranges.sort()
        merged = []
        cur_start, cur_end = ranges[0]
        for start, end in ranges[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = start, end
        merged.append((cur_start, cur_end))
        return merged
    return []

def _find_any_markers(text):
    """
    Ищет в тексте все маркеры вида !!!name!!! (с вариациями символов '!' и '¡').
    Возвращает список словарей с ключами: 'start', 'end', 'raw_name'.
    """
    exclamation_chars = {'!', '¡'}
    pattern = r'[!¡]{1,3}\s*([\w\s]+?)\s*[!¡]{1,3}'
    markers = []
    for match in re.finditer(pattern, text): markers.append({'start': match.start(), 'end': match.end(), 'raw_name': match.group(1).strip()})
    return markers

def remove_wrong_command_messages():
    if global_state.wrong_command_messages_vector_ids != []:
        sql_exec("DELETE FROM rag_messages WHERE vector_id IN ({})".format(','.join('?' * len(global_state.wrong_command_messages_vector_ids))), global_state.wrong_command_messages_vector_ids)
        coll_exec(action="delete", coll_name="rag_collection", ids=global_state.wrong_command_messages_vector_ids)
        global_state.wrong_command_messages_vector_ids = []

def add_wrong_command_message_id(vid):
    if cut_wrong_command_history: global_state.wrong_command_messages_vector_ids.append(vid)

def analyze_protocol(text, now_commands={}):
    """
    Анализирует текст ответа модели на соответствие протоколу вызова инструментов.
    Возвращает:
        None - нет команд (можно завершить цикл и отправить сообщение пользователю)
        str - предупреждение о нарушении (нужно отправить модели и повторить цикл)
    """
    if global_state.stop_agent: return None
    sid = global_state.now_agent_id
    commands_dict = global_state.tools_commands_dict.get(sid, {})
    raw_markers = _find_any_markers(text)
    if not raw_markers: return None
    violations = []
    if len(raw_markers) > 1: violations.append(warn_command_text_2) # множественность
    if not native_func_call: # позиция первого маркера (если не native)
        first_marker = raw_markers[0]
        if first_marker['start'] > 5: violations.append(warn_command_text_3)
    # markdown-блоки (```)
    markdown_ranges = []
    md_positions = [m.start() for m in re.finditer(r'```', text)]
    for i in range(0, len(md_positions), 2):
        if i + 1 < len(md_positions): markdown_ranges.append((md_positions[i], md_positions[i+1] + 3))
    # json-блоки (грубо)
    json_ranges = []
    stack = []
    for i, ch in enumerate(text):
        if ch == '{': stack.append(i)
        elif ch == '}':
            if stack: start = stack.pop(); json_ranges.append((start, i + 1))
    # инлайн-форматирование
    formatting_ranges = _find_formatting_ranges(text)
    for marker in raw_markers:
        pos = marker['start']
        for start, end in markdown_ranges: # проверка попадания в markdown-блоки
            if start <= pos <= end: violations.append(warn_command_text_4); break
        for start, end in json_ranges: # проверка попадания в json-блоки
            if start <= pos <= end: violations.append(warn_command_text_5); break
        for start, end in formatting_ranges: # проверка попадания в инлайн-форматирование
            if start <= pos <= end: violations.append(warn_command_text_6); break
    # проверка известности команды
    unknown_command = False
    for marker in raw_markers:
        raw_name = marker['raw_name']
        normalized_name = re.sub(r'\s+', '_', raw_name).lower()
        found = False
        # точное совпадение
        for key in commands_dict:
            if normalized_name == key.lower(): found = True; break
        # нечёткое сравнение, если точного нет
        if not found and commands_dict:
            best_match = None
            best_ratio = 0.0
            threshold = 0.8
            for key in commands_dict:
                key_lower = key.lower()
                if abs(len(normalized_name) - len(key_lower)) > 2: continue
                ratio = difflib.SequenceMatcher(None, normalized_name, key_lower).ratio()
                if ratio > best_ratio and ratio >= threshold: best_ratio = ratio; best_match = key
            if best_match: found = True
        if not found: unknown_command = True; break
    if unknown_command:
        known_commands_str = warn_command_text_7
        for known_tool in now_commands: known_commands_str + '\n' + ' (' + now_commands[known_tool][0] + ')'
        violations.append(wrong_command + known_commands_str)
    if not violations:
        remove_wrong_command_messages()
        return None # если нарушений нет, команды корректны, но мы не будем их выполнять (т.к. find_and_match_command не сработал) TODO: ТУТ МОЖЕТ БЫТЬ ОШИБКА
    add_wrong_command_message_id(vector_id_out)
    violations = list(dict.fromkeys(violations)) # TODO: переработай циклы
    violations.insert(0, warn_command_text_1)
    violations.append(warn_command_text_7)
    return "\n".join(violations) # формируем сообщение

def tools_selector(text, sid):
    """
    Вызывает инструменты, используя поиск маркеров и словарь команд в global_state.tools_commands_dict[sid].
    Возвращает результат выполнения команды или None.
    """
    if do_translate and local_and_tools_translate: text = translate_text(text, language, from_lang=target_lang)
    let_log("=== [TOOLS_SELECTOR ЗАПУЩЕН (НОВАЯ ВЕРСИЯ)] ===")
    let_log(f"[TOOLS_SELECTOR] входной текст (начало 200):\n{text[:200]}")
    # 1) получить кэш
    cached = read_cache()
    let_log(f"[TOOLS_SELECTOR] кэш: {cached}")
    if cached == [True, [False, False]]: return None
    try:
        if isinstance(cached[1][1], str): return cached[1][1]
    except: pass
    #is_warn = analyze_protocol(text) тут возврат не делаем потому что это только сильнее путает модель как оказалось
    # 2) получить словарь команд для сессии
    try: now_commands = global_state.tools_commands_dict.get(sid, {})
    except Exception: now_commands = {}
    # 3) system keys
    try: sys_keys = [str(k) for k in global_state.system_tools_keys]
    except Exception: sys_keys = []
    # 4) найти маркер и сопоставить с командами
    match = find_and_match_command(text, now_commands)
    if not match: # TODO: может разделить случаи когда маркер не найден или команда не сопоставилась
        let_log("[TOOLS_SELECTOR] маркер не найден или команда не сопоставилась")
        let_log(text)
        let_log(now_commands)
        is_warn = analyze_protocol(text, now_commands)
        if is_warn != None:
            let_log(is_warn)
            write_cache([False, is_warn])
            let_log("=== [TOOLS_SELECTOR ЗАВЕРШЁН С ИНФОРМАЦИЕЙ ОБ ОШИБКЕ] ===")
            return is_warn
        write_cache([False, False])
        let_log("=== [TOOLS_SELECTOR ЗАВЕРШЁН] ===")
        return None
    found_key, content = match
    let_log(f"[TOOLS_SELECTOR] найден ключ: {found_key}, контент длиной: {len(content) if content else 0}")
    is_system = False
    try: # 5) определить, системная ли команда
        for sk in sys_keys:
            if found_key in sk or sk in found_key: is_system = True; break
        if not is_system and sys_keys:
            if difflib.get_close_matches(found_key, sys_keys, n=1, cutoff=0.7): is_system = True
    except Exception: is_system = False
    let_log(f"[TOOLS_SELECTOR] команда системная? {is_system}")
    # 6) Обработка кэша в зависимости от типа команды
    if not is_system: # ТОЛЬКО для несистемных команд: проверяем кэш
        if cached != [False]:
            if cached[1] != ["SYSTEM", False]:
                let_log("[TOOLS_SELECTOR] Возвращаем не-системный результат из кэша")
                let_log("=== [TOOLS_SELECTOR ЗАВЕРШЁН] ===")
                return cached[1]
    else: # Для системных команд: проверяем, не выполняем ли мы её уже (рекурсия)
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
        is_warn = analyze_protocol(text, now_commands)
        if is_warn != None:
            let_log(is_warn)
            write_cache([False, is_warn])
            let_log("=== [TOOLS_SELECTOR ЗАВЕРШЁН] ===")
            return is_warn
        let_log("=== [TOOLS_SELECTOR ЗАВЕРШЁН] ===")
        write_cache([False, wrong_command])
        return wrong_command
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
        # Неинициализированный system tool — сразу стоп
        try:
            if found_key in global_state.system_tools_keys or any(
                    found_key in sk or sk in found_key for sk in global_state.system_tools_keys):
                let_log(f"[FATAL] Системный инструмент '{found_key}' не инициализирован")
                try: send_ui_no_cache(f"FATAL: system tool not initialized: {found_key}")
                except Exception: pass
                sys.exit(1)
        except Exception:
            pass
        let_log("=== [TOOLS_SELECTOR ЗАВЕРШЁН] ===")
        write_cache([False, False])
        return None
    # 9) выполнить функцию
    let_log("[TOOLS_SELECTOR] Выполняем функцию...")
    if found_key == global_state.start_dialog_command_name: global_state.task_delegated = True
    remove_wrong_command_messages()
    # Настоящий system tool = в system_tools_keys (модули из system_tools, не сторонние)
    is_real_system = False
    try:
        is_real_system = found_key in global_state.system_tools_keys or any(
            found_key in sk or sk in found_key for sk in global_state.system_tools_keys)
    except Exception:
        is_real_system = is_system
    try:
        result = func_callable(content)
    except Exception as e:
        if is_real_system:
            let_log(f"[FATAL] Ошибка в системном инструменте '{found_key}': {e}")
            try: send_ui_no_cache(f"FATAL system tool error: {found_key}: {e}")
            except Exception: pass
            sys.exit(1)
        result = "__TOOL_ERROR__: " + str(e)
    if not isinstance(result, str):
        if is_real_system:
            let_log(f"[FATAL] Системный инструмент '{found_key}' вернул не str")
            sys.exit(1)
        raise RuntimeError('FUNCTION ANSWER MUST BE STR')
    let_log(f"[TOOLS_SELECTOR] Результат (первые 500):\n{str(result)[:500]}")
    try: # 10) кэшировать результат если не системная команда
        if not is_system:
            let_log("[TOOLS_SELECTOR] Кэшируем результат (не системная команда)")
            write_cache([True, result])
            traceprint()
    except Exception: pass
    let_log("=== [TOOLS_SELECTOR ЗАВЕРШЁН] ===")
    if do_translate and local_and_tools_translate: result = translate_text(result, target_lang)
    return result

def agent_func(text, agent_number):
    global vector_id_out, vector_id_in #?
    global_state.last_agent = agent_number
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
        # Вызываем RAG-конструктор. Он сам найдет системный промпт и всю историю.
        final_prompt_for_model, _ = get_chat_context(sid, talk_prompt)
        # Вызываем модель, добавив роль текущего агента для корректной генерации
        talk_prompt = ask_model(final_prompt_for_model + you)
        talk_prompt = remove_commands_roles(talk_prompt)
        # Сохраняем ответ самой модели в RAG-историю
        set_common_save_id()
        vector_id_out = str(get_common_save_id())
        embedding = get_embs(text)
        #coll_exec(action="add", coll_name="rag_collection", ids=[vector_id_out], metadatas=[{'chat_id': str(sid), 'role': you, 'relevance_score': 0}], embeddings=[embedding])
        # TODO: вот тут локал мессадж
        #let_log(f"Сообщение {vector_id_out} векторизовано и добавлено в RAG")
        answer = tools_selector(talk_prompt, sid)
        if answer:
            let_log(global_state.stop_agent)
            talk_prompt = answer
            msg_from = func_role_text
            update_history(sid, talk_prompt, you, vector_id=vector_id_out)
            # посмотри как вектор айди ин работает, где создавать
            vector_id_in = update_history(sid, talk_prompt, func_role_text)
            # тут надо сохранять от функции но сначала от агента
        else:
            update_history(sid, talk_prompt, you, vector_id=vector_id_out, local_message=False)
            break
        # Сохраняем входящее сообщение от предыдущего агента в RAG-историю
        # а тут только исходящее от агента
        if global_state.wrong_command_messages_vector_ids != []: add_wrong_command_message_id(vector_id_in)
    global_state.stop_agent = False
    return talk_prompt

def get_user_feedback(current_task, dialog_result):
    while True:
        ims = get_input_message()
        if ims == None: break
    # --- Перевод финального ответа на язык пользователя (если local_and_tools_translate) ---
    if do_translate and local_and_tools_translate:
        dialog_result_to_user = translate_text(dialog_result, language, from_lang=target_lang)
    else:
        dialog_result_to_user = dialog_result
    send_output_message(text=dialog_result_to_user, command='end')
    user_message = get_input_message(wait=True)
    # --- Перевод текста пользователя на целевой язык (если local_and_tools_translate) ---
    if do_translate and local_and_tools_translate:
        user_text = translate_text(user_message['text'], target_lang, from_lang=language)
    else:
        user_text = user_message['text']
    updated_task = current_task + user_review_text2 + dialog_result + user_review_text3 + user_text
    if user_message['attachments']:
        send_output_message(text=start_load_attachments_text)
        upload_user_data(user_message['attachments'])
        send_output_message(text=end_load_attachments_text)
    return updated_task

def get_user_or_critic_feedback(rmt):
    if global_state.critic_wants_retry:
        rmt = really_main_task + user_review_text2 + global_state.dialog_result + user_review_text4 + global_state.critic_comment
        let_log(f"[WORKER] Updating really_main_task after critic retry: {rmt[:100]}")
    else:
        rmt = get_user_feedback(rmt, global_state.dialog_result)
        let_log(f"[WORKER] Updating really_main_task after user feedback: {really_main_task[:100]}")
    return rmt

# ===== ИЗМЕНЁННАЯ ФУНКЦИЯ worker (ПОЛНАЯ) =====
def worker(really_main_task):
    let_log(f"[WORKER] START: really_main_task={really_main_task[:100]}, conversations={global_state.conversations}, now_agent_id={global_state.now_agent_id}")
    # --- Перевод задачи на целевой язык (если local_and_tools_translate) ---
    if do_translate and local_and_tools_translate:
        really_main_task = translate_text(really_main_task, target_lang, from_lang=language)
        let_log(f"[WORKER] Задача переведена на целевой язык: {really_main_task[:100]}")
    while True:
        global_state.retries = []
        global_state.conversations = 0
        global_state.tools_commands_dict = {}
        global_state.dialog_state = True
        global_state.critic_wants_retry = False
        global_state.main_now_task = really_main_task
        global_state.gigo_web_search_allowed = False
        let_log(f"[WORKER] Before start_dialog: main_now_task={global_state.main_now_task[:100]}, dialog_state={global_state.dialog_state}")
        talk_prompt = start_dialog(global_state.main_now_task)
        let_log(f"[WORKER] After start_dialog: talk_prompt={talk_prompt[:100] if talk_prompt else 'None'}, dialog_state={global_state.dialog_state}")
        global_state.gigo_web_search_allowed = True
        if not global_state.dialog_state:
            let_log(f"[WORKER] Dialog finished without starting")
            really_main_task = get_user_or_critic_feedback(really_main_task)
            continue
        while True:
            let_log(f"[WORKER] Main loop start. dialog_state={global_state.dialog_state}, conversations={global_state.conversations}, now_agent_id={global_state.now_agent_id}")
            if global_state.dialog_state:
                let_log(f"[WORKER] Calling agent_func(0) with talk_prompt={talk_prompt[:100]}")
                talk_prompt = agent_func(talk_prompt, 0)
                let_log(f"[WORKER] agent_func(0) returned talk_prompt={talk_prompt[:100]}, task_delegated={global_state.task_delegated}")
                if global_state.task_delegated:
                    let_log(f"[WORKER] Task delegated, resetting flag and continue")
                    global_state.task_delegated = False
                    continue
            if global_state.dialog_state:
                let_log(f"[WORKER] Calling agent_func(1) with talk_prompt={talk_prompt[:100]}")
                talk_prompt = agent_func(talk_prompt, 1)
                let_log(f"[WORKER] agent_func(1) returned talk_prompt={talk_prompt[:100]}")
            if not global_state.dialog_state:
                let_log(f"[WORKER] Dialog state became false. tools_commands_dict={global_state.tools_commands_dict}, dialog_result={global_state.dialog_result[:100]}, conversations={global_state.conversations}")
                print(global_state.tools_commands_dict)
                let_log(global_state.dialog_result)
                let_log(global_state.conversations)
                if global_state.conversations <= 0:
                    let_log(f"[WORKER] conversations<=0, handling user feedback/critic")
                    really_main_task = get_user_or_critic_feedback(really_main_task)
                    break
                else:
                    let_log(f"[WORKER] conversations>0, resetting dialog_state and possibly retrying with critic or new prompt")
                    global_state.dialog_state = True
                    if global_state.critic_wants_retry:
                        global_state.main_now_task = global_state.main_now_task + user_review_text2 + global_state.dialog_result + user_review_text4 + global_state.critic_comment
                        let_log(f"[WORKER] critic retry: updated main_now_task={global_state.main_now_task[:100]}")
                        talk_prompt = start_dialog(global_state.main_now_task)
                        let_log(f"[WORKER] after start_dialog (critic): talk_prompt={talk_prompt[:100]}, dialog_state={global_state.dialog_state}")
                    else:
                        talk_prompt = global_state.dialog_result
                        let_log(f"[WORKER] using dialog_result as new talk_prompt={talk_prompt[:100]}")

def init_chromadb(chroma_path, use_rag, max_attempts=3):
    def _try_init(remove_on_failure=False): # Вспомогательная функция для попытки инициализации
        try:
            client = chromadb.PersistentClient(path=chroma_path, settings=Settings(allow_reset=True, anonymized_telemetry=False))
            milana = client.get_or_create_collection(name="milana_collection", metadata={"hnsw:space": "cosine"})
            user = client.get_or_create_collection(name="user_collection", metadata={"hnsw:space": "cosine"})
            rag = None
            if use_rag: rag = client.get_or_create_collection(name="rag_collection", metadata={"hnsw:space": "cosine"})
            return client, milana, user, rag, True
        except Exception as e:
            error_msg = str(e)
            let_log(f"ChromaDB init error: {error_msg}")
            if remove_on_failure:
                if os.path.exists(chroma_path):
                    shutil.rmtree(chroma_path, ignore_errors=True)
                    let_log(f"Removed entire ChromaDB directory: {chroma_path}")
                return None, None, None, None, False
            if "already exists" in error_msg.lower():
                collections_to_try = ["milana_collection", "user_collection", "rag_collection"]
                try: # Создаём временный клиент для удаления коллекций
                    temp_client = chromadb.PersistentClient(path=chroma_path, settings=Settings(allow_reset=True))
                    for coll_name in collections_to_try:
                        try:
                            temp_client.delete_collection(coll_name)
                            let_log(f"Deleted collection '{coll_name}' to resolve conflict")
                        except Exception: pass
                except Exception: pass
            return None, None, None, None, False
    for attempt in range(max_attempts):
        let_log(f"ChromaDB init attempt {attempt+1}/{max_attempts}")
        client, milana, user, rag, success = _try_init(remove_on_failure=False) # Сначала пробуем без удаления папки
        if success: return client, milana, user, rag
        if attempt == max_attempts - 1: # Если не удалось, на последней попытке удаляем папку
            let_log("Final attempt: deleting entire ChromaDB directory and retrying")
            client, milana, user, rag, success = _try_init(remove_on_failure=True)
            if success: return client, milana, user, rag
            else: raise RuntimeError("Cannot initialize ChromaDB even after deleting the database folder")
    raise RuntimeError("Unexpected: failed to initialize ChromaDB after all attempts")

def initialize_work(base_dir, chat_id, input_queue, output_queue, log_queue, session_passwords=None, settings_override=None):
    """
    settings_override: dict настроек при первом запуске чата (не читать chatsettings с диска).
    При повторных запусках (resume) settings_override=None — читаем из БД как раньше.
    """
    global actual_handlers_names, another_tools_files_addresses
    global token_limit, emb_token_limit, chunk_size, left_cache_counter
    global client, milana_collection, user_collection, rag_collection
    global ui_conn
    global cache_path, chat_path, memory_sql, folder_path, slash, filesystem_project_path, global_trans_db_path
    global ask_provider_model, ask_provider_model_chat, get_provider_embs
    global create_chat, get_chat_context, update_history, delete_chat
    global language, do_translate, target_lang, local_and_tools_translate, use_local_cache, use_global_cache
    global do_chat_construct, native_func_call
    global use_rag, clean_variables_content, filter_generations, is_save_log, use_librarian, recreate_agents, cut_wrong_command_history, use_psm
    global pipeline, get_dependency_report, change_dir, get_project_tree_json, create_experiment_branch, status_success, status_failed, status_forbidden, resolve_workspace_path, to_posix_rel, allowed_actions, normalize_action
    global use_magical_prompt, use_gigo, use_old_gigo, gigo_idea_count, gigo_plan_items, gigo_use_entropy, gigo_use_random_roles, gigo_use_filter, gigo_use_librarian
    global librarian_use_models, module_hints_for_operator, give_all_tools, critic_reuse_dialog, one_shot_intention_permission
    if session_passwords: import encryption_utils; encryption_utils.SESSION_PASSWORDS.update(session_passwords) # Загружаем пароли из родительского процесса UI в память этого процесса
    ui_conn = [input_queue, output_queue, log_queue]
    # пометка: settings_override применится после открытия db_path
    # === Загружаем параметры чата ===
    global_trans_db_path = os.path.join(base_dir, "data", "settings.db")
    # Корень чатов из глобальных настроек (chats_dir), fallback data/chats
    chats_root = os.path.join(base_dir, "data", "chats")
    try:
        _gs_conn = connect(global_trans_db_path)
        _gs_cur = _gs_conn.cursor()
        _gs_cur.execute("SELECT value FROM settings WHERE key = 'chats_dir'")
        _row = _gs_cur.fetchone()
        _gs_conn.close()
        if _row and _row[0] and str(_row[0]).strip():
            _cd = os.path.expanduser(str(_row[0]).strip())
            chats_root = _cd if os.path.isabs(_cd) else os.path.join(base_dir, _cd)
    except Exception:
        pass
    chat_path = os.path.join(chats_root, chat_id)
    filesystem_project_path = os.path.join(chat_path, 'files')
    cache_path = os.path.join(chat_path, "cache.db")
    folder_path = base_dir # Обновляем пути для system_tools
    sys.path = [p for p in sys.path if not p.endswith(('system_tools', 'system_tools/milana', 'system_tools/ivan'))]
    sys.path.append(os.path.join(folder_path, 'system_tools'))
    sys.path.append(os.path.join(folder_path, 'system_tools', 'milana'))
    sys.path.append(os.path.join(folder_path, 'system_tools', 'ivan'))
    let_log(f"Base directory: {base_dir}")
    let_log(f"Chat path: {chat_path}")
    let_log(f"Folder path: {folder_path}")
    # === Подготовка SQLite БД ===
    init_cache_conn = connect(cache_path)
    init_cache_cursor = init_cache_conn.cursor()
    init_cache_cursor.execute('CREATE TABLE IF NOT EXISTS cache (id INTEGER PRIMARY KEY, value BLOB)')
    init_cache_cursor.execute('SELECT COUNT(*) FROM cache')
    left_cache_counter = init_cache_cursor.fetchone()[0]
    init_cache_conn.commit()
    init_cache_conn.close()
    db_path = os.path.join(chat_path, "chatsettings.db")
    let_log(f"Database path: {db_path}")
    memory_sql = connect(db_path)
    sql_exec('''CREATE TABLE IF NOT EXISTS found_info (id INTEGER PRIMARY KEY AUTOINCREMENT, info TEXT NOT NULL)''')
    let_log("##### Инициализация таблиц базы данных RAG... #####")
    sql_exec('''
        CREATE TABLE IF NOT EXISTS rag_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT,
            role TEXT,
            full_text TEXT,
            is_vectorized BOOLEAN DEFAULT FALSE,
            vector_id TEXT,
            relevance_score INTEGER DEFAULT 0,
            is_compressed BOOLEAN DEFAULT FALSE,
            global_summary TEXT DEFAULT NULL,
            recent_summary TEXT DEFAULT NULL
            );''')
    sql_exec('CREATE TABLE IF NOT EXISTS system_prompts (chat_id INTEGER PRIMARY KEY, system_prompt TEXT)''')
    initial_text, fl = load_initial_data(chat_id)
    # При первом запуске UI может передать settings_override, чтобы не читать БД повторно
    if settings_override and isinstance(settings_override, dict):
        settings = dict(settings_override)
        let_log("[initialize_work] settings from override dict (first start)")
    else:
        settings = load_chat_settings(chat_id)
        let_log("[initialize_work] settings from chatsettings.db")
    tool_paths = settings.get("another_tools", [])
    token_limit = int(settings.get("token_limit", 8192))
    global_state.allow_ocr = int(settings.get("allow_ocr", 0)) == 1
    global_state.hierarchy_limit = int(settings.get("hierarchy_limit", 0))
    global_state.write_results = int(settings.get("write_results", 0)) == 1
    global_state.number_of_plan_items = int(settings.get("number_of_plan_items", 0))
    global_state.max_critic_reactions = int(settings.get("max_critic_reactions", 2))
    global_state.skip_nested_images = int(settings.get("skip_nested_images", 0)) == 1
    use_rag = int(settings.get("use_rag", 1)) == 1
    use_psm = int(settings.get("use_psm", 0)) == 1
    is_save_log = int(settings.get("write_log", 1)) == 1
    use_librarian = int(settings.get("use_librarian", 1)) == 1
    recreate_agents = int(settings.get("recreate_agents", 0)) == 1
    filter_generations = int(settings.get("filter_generations", 0)) == 1
    use_magical_prompt = int(settings.get("use_magical_prompt", 0)) == 1
    cut_wrong_command_history = int(settings.get("cut_wrong_command_history", 1)) == 1

    do_translate = int(settings.get("do_translate", 0)) == 1
    target_lang = settings.get("target_lang", "None")
    local_and_tools_translate = int(settings.get("local_and_tools_translate", 0)) == 1
    use_local_cache = int(settings.get("use_local_cache", 1)) == 1
    use_global_cache = int(settings.get("use_global_cache", 0)) == 1

    use_gigo = int(settings.get("use_gigo", 1)) == 1
    use_old_gigo = int(settings.get("use_old_gigo", 1)) == 1
    gigo_idea_count = int(settings.get("gigo_idea_count", 2))
    gigo_plan_items = int(settings.get("gigo_plan_items", 10))
    gigo_use_entropy = int(settings.get("gigo_use_entropy", 0)) == 1
    gigo_use_random_roles = int(settings.get("gigo_use_random_roles", 1)) == 1
    gigo_use_filter = int(settings.get("gigo_use_filter", 1)) == 1
    gigo_use_librarian = int(settings.get("gigo_use_librarian", 1)) == 1
    librarian_use_models = int(settings.get("librarian_use_models", 0)) == 1
    module_hints_for_operator = int(settings.get("module_hints_for_operator", 0)) == 1
    give_all_tools = int(settings.get("give_all_tools", 0)) == 1
    critic_reuse_dialog = int(settings.get("critic_reuse_dialog", 1)) == 1
    one_shot_intention_permission = int(settings.get("one_shot_intention_permission", 0)) == 1
    # optional trace flags (entry in settings optional)
    try:
        global_state.trace_full = int(settings.get("trace_full", 0)) == 1
        global_state.trace_func_io = int(settings.get("trace_func_io", 0)) == 1
    except Exception:
        pass

    chroma_path = os.path.join(chat_path, "chroma_db") # === Инициализация ChromaDB ===
    client, milana_collection, user_collection, rag_collection = init_chromadb(chroma_path, use_rag)
    from chat_manager import create_chat, get_chat_context, update_history, delete_chat
    default_tools_dir = os.path.join(base_dir, "default_tools")
    for rel_path in tool_paths:
        if os.path.isabs(rel_path): full_path = rel_path
        else: full_path = os.path.join(default_tools_dir, rel_path)
        full_path = os.path.normpath(full_path)
        another_tools_files_addresses.append(full_path)
    # === Инициализация модели ===
    model_type = settings.get("model_type", "ollama")
    try:
        model_providers_path = os.path.join(base_dir, "model_providers")
        if model_providers_path not in sys.path: sys.path.append(model_providers_path)
        model_providers_module = importlib.import_module(f"model_providers.{model_type}")
        ask_model = model_providers_module.ask_model
        ask_model_chat = model_providers_module.ask_model_chat
        create_embeddings = model_providers_module.create_embeddings
        model_connect = model_providers_module.connect
        model_disconnect = model_providers_module.disconnect
        connect_params = settings.get("model_provider_params", "")
        params_dict = {}
        for part in connect_params.split(";"):
            if "=" not in part: continue
            k, v = part.split("=", 1)
            params_dict[k.strip().lower()] = v.strip()
        decrypted_token = None
        if params_dict.get("password") == "set": # Если в параметрах указано, что пароль установлен
            # Пытаемся получить пароль из кэша (сначала по chat_id, потом по model_type)
            password = None
            if session_passwords: password = session_passwords.get(chat_id) or session_passwords.get(model_type)
            if not password: raise RuntimeError(f"Password not found in session for chat {chat_id} or model {model_type}")
            # Определяем, какой параметр содержит зашифрованный токен
            encrypted_token = params_dict.get("api_token") or params_dict.get("token", "")
            if encrypted_token:
                try: decrypted_token = encryption_utils.decrypt_token(encrypted_token, password)
                except Exception as e: raise RuntimeError(f"Failed to decrypt token: {e}")
            else: raise RuntimeError("Encrypted token (api_token/token) not found in connection string")
        elif params_dict.get("password") == "empty": decrypted_token = params_dict.get("api_token") or params_dict.get("token", "")
        sig = inspect.signature(model_connect) # Вызываем connect провайдера, передавая расшифрованный токен отдельным параметром
        if '_decrypted_token' in sig.parameters: connection_result = model_connect(connect_params, _decrypted_token=decrypted_token)
        else: connection_result = model_connect(connect_params)
        if not connection_result or not connection_result[0]: let_log(f"Ошибка подключения модели: {connection_result[1] if len(connection_result) > 1 else 'Unknown error'}"); return
        success = connection_result[0]
        tags = connection_result[2] if len(connection_result) > 2 else {}
        if tags: global unified_tags; unified_tags = tags
        globals().update({'ask_provider_model': ask_model, 'ask_provider_model_chat': ask_model_chat, 'get_provider_embs': create_embeddings,})
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
    # === Загрузка языка, модели и инструментов ===
    if do_translate:
        if use_local_cache:
            sql_exec("CREATE TABLE translation_cache (src_text TEXT NOT NULL, translation TEXT NOT NULL, to_lang TEXT NOT NULL)")
            # выгрузка из глобального с целевым языком (тогда можно и в модуль импортировать функцию работы с глобальными настройками (только на самом деле надо отдельный файл))
            if use_global_cache:
                global_cache_records = global_trans_cache_exec("SELECT src_text, translation FROM translation_cache WHERE to_lang IN (?, ?)", (target_lang, language), fetchall=True)
                if global_cache_records:
                    sql_exec("INSERT OR IGNORE INTO translation_cache (src_text, translation, to_lang) VALUES (?, ?, ?)", global_cache_records, executemany=True)
                    print(f"Загружено {len(global_cache_records)} записей")
        from simple_translator import translate_text, translate_texts
    language = settings.get("language", "ru")
    globalize_language_packet(language)
    chunk_size = provider_module.emb_token_limit * text_tokens_coefficient
    clean_variables_content = []
    if filter_generations == 1:
        clean_variables_content = [operator_role_text, worker_role_text, func_role_text, system_role_text]
        if unified_tags.get('bos') is not None: clean_variables_content.append(unified_tags.get('bos'))
        if unified_tags.get('user_start') is not None: clean_variables_content.append(unified_tags.get('user_start'))
        filter_generations = True
    else: filter_generations = False

    from filesystem import (
        pipeline,
        get_dependency_report,
        change_dir,
        get_project_tree_json,
        create_experiment_branch,
        #status_success,
        #status_failed,
        #status_forbidden,
        resolve_workspace_path,
        to_posix_rel,
        allowed_actions,
        normalize_action)

    let_log(f"\n=== ЗАГРУЗКА СПЕЦИАЛЬНЫХ МОДУЛЕЙ (до системных) ===")
    special_files = {'web_search': None, 'ask_user': None}
    other_files = []
    for file_path in another_tools_files_addresses:
        file_name = os.path.basename(file_path).lower()
        if file_name.endswith('web_search.py'): special_files['web_search'] = file_path
        elif file_name == 'ask_user.py': special_files['ask_user'] = file_path
        else: other_files.append(file_path)
    loaded_tools = []
    for module_type, file_path in special_files.items():
        if file_path:
            let_log(f"Загрузка {module_type}: {file_path}")
            module_result = mod_loader([file_path])
            if module_result:
                for command_name, description, func in module_result:
                    loaded_tools.append((command_name, description, func))
                    if module_type == 'web_search':
                        globals()['web_search'] = func
                        let_log(f"✅ Функция web_search глобализована")
                    elif module_type == 'ask_user':
                        globals()['ask_user'] = func
                        let_log(f"✅ Функция ask_user глобализована")
            else: let_log(f"⚠ Не удалось загрузить {module_type}")
    global_state.ivan_module_tools, global_state.milana_module_tools = system_tools_loader()
    if other_files:
        let_log(f"\nЗагрузка обычных модулей ({len(other_files)} файлов)")
        other_loaded = mod_loader(other_files)
        loaded_tools.extend(other_loaded)
    else: let_log("Нет обычных модулей для загрузки")
    global_state.another_tools = loaded_tools
    let_log(f"\n=== ИТОГИ ЗАГРУЗКИ ===")
    let_log(f"Всего загружено инструментов: {len(loaded_tools)}")
    let_log(f"Веб-поиск доступен: {'web_search' in globals()}")
    let_log(f"Ask_user доступен: {'ask_user' in globals()}")
    let_log("Список инструментов:")
    for tt, t, _ in global_state.another_tools:
        global_state.module_tools_keys.append(tt)
        if tt not in global_state.skip_tools_keys: global_state.tools_str += tt + ' (' + t + ')\n'
        let_log(tt)
    if fl:
        send_output_message(text=start_load_attachments_text)
        upload_user_data(fl)
        send_output_message(text=end_load_attachments_text)
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