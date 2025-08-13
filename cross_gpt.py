import os
from sqlite3 import connect, OperationalError
import chromadb
from chromadb.config import Settings
import importlib.util
import sys
import re
import traceback
import time
import difflib
from multiprocessing import queues

Empty = queues.Empty # Чтобы ловить исключение Empty у multiprocessing.Queue

def traceprint(*args, **kwargs):
    stack = traceback.extract_stack()
    caller = stack[-2]  # фрейм, откуда вызвана функция
    line_number = caller.lineno
    filename = caller.filename.split("/")[-1]  # только имя файла
    if not args:  # если аргументов нет, печатаем только номер строки
        let_log(f"[{filename}:{line_number}]")
    else: # иначе печатаем номер строки + переданные аргументы
        let_log(f"[{filename}:{line_number}]:", *args, **kwargs)

class GlobalState:
    def __init__(self):
        self.stop_agent = False
        self.dialog_state = True
        self.dialog_result = ''
        self.conversations = 0
        self.tools_commands_dict = {}
        self.last_task_for_specialist = []
        self.now_try = '/'
        self.common_save_id = 0
        self.retries = []
        self.retrying = False
        self.main_now_task = ''
        self.parents_id = '/'
        self.another_tools = []
        self.tools_str = []
        self.milana_module_tools = []
        self.ivan_module_tools = []
        self.module_tools_keys = []
        self.task_retry = 0
        self.max_attempts = 3
        self.system_tools_keys = []
        self.an_t_str = {}

global_state = GlobalState()

def set_now_try(value=None):
    if global_state.now_try[-1] == '/': 
        global_state.now_try = global_state.now_try + '1'
    else:
        parts = global_state.now_try.split('/')
        parts[-1] = str(int(parts[-1]) + 1)
        global_state.now_try = '/'.join(parts)

def get_now_try():
    return global_state.now_try

def up_now_try():
    parts = global_state.now_try.split('/')
    del parts[-1]
    global_state.now_try = '/'.join(parts)
    if global_state.now_try == '': 
        global_state.now_try = '/'

def down_now_try(): 
    global_state.now_try = global_state.now_try + '/'

def reset_now_try(): 
    global_state.now_try = 1

def set_common_save_id(): 
    global_state.common_save_id += 1

def get_common_save_id(): 
    return str(global_state.common_save_id)

def reset_common_save_id(): 
    global_state.common_save_id = 1

def find_work_folder(file_name):
    real_path = os.path.realpath(file_name)
    if real_path.find('\\'): slash = '\\'
    else: slash = '/'
    return file_name[:file_name.rfind(slash)], slash

folder_path, slash = find_work_folder(__file__)
sys.path.append(os.path.join(folder_path, 'system_tools'))
sys.path.append(os.path.join(folder_path, 'system_tools', 'milana'))
sys.path.append(os.path.join(folder_path, 'system_tools', 'ivan'))

is_print_log = True # TODO: перед релизом сделай false
is_save_log = True

def let_log(t):
    t = str(t)
    # Печать в консоль
    if is_print_log:
        print(t)

    # Сохранение лога
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
        
        # Получаем текущий максимальный id в таблице cache
        cursor.execute('SELECT MAX(id) FROM cache')
        row = cursor.fetchone()
        max_id = row[0] if row and row[0] is not None else -1
        
        conn.close()
        
        # Записываем в файл только если max_id >= cache_counter
        if max_id >= cache_counter:
            log_file = os.path.join(chat_path, 'log.txt')
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f'{t}\n')

memory_sql = None  # глобальная переменная для SQL-функций

client = None
milana_collection = None
user_collection = None
max_tasks = 0
most_often = False
need_best_result = False
little_model = True
retry_lowest = False # нужны функции проверки нижайшего и высшего диалога
retry_highest = False
retry_all = False
may_use_user_files = False
may_use_internet = False
cache_counter = 1 # всегда начинается с 1
language = ''

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
              **kwargs):
    """
    Универсальная обёртка с кэшированием, обратно-совместимая
    и с поддержкой add/update/delete/query/get/count/modify.
    """
    # Вызов кэша
    cached = cacher(); traceprint()
    if cached is not None:
        # старый coll_exec возвращал строки "None"/"True"/"False"
        if cached == "None":
            return None
        if cached == "True":
            return True
        if cached == "False":
            return False
        return eval(cached)

    # Получаем объект коллекции
    coll = globals().get(coll_name) or (client and client.get_collection(coll_name))
    if coll is None and action != "delete_collection":
        raise NameError(f"Collection '{coll_name}' not found")

    def _make_where(d):
        if not d: return None
        clauses = []
        for k, v in d.items():
            if isinstance(v, list):
                clauses.append({k: {"$in": v}})
            else:
                clauses.append({k: v})
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def _extract(resp, include):
        # множественный fetch → dict
        if len(include) > 1:
            out = {}
            for key in include:
                data = resp.get(key, []) or []
                if flatten and isinstance(data, list) and data and isinstance(data[0], list):
                    data = [i for sub in data for i in sub]
                out[key] = data
            return out
        # одиночный fetch → плоский список или первый элемент
        key = include[0]
        data = resp.get(key, []) or []
        if first:
            return data[0] if data else None
        if isinstance(data, list) and data and isinstance(data[0], list):
            return [i for sub in data for i in sub]
        return data

    try:
        # add
        if action == "add":
            out = coll.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
                **kwargs
            )
            cacher("True"); traceprint()
            return out

        # update
        if action == "update":
            out = coll.update(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
                **kwargs
            )
            cacher("True"); traceprint()
            return out

        # delete
        if action == "delete":
            out = coll.delete(
                ids=ids,
                where=_make_where(filters),
                **kwargs
            )
            cacher("True"); traceprint()
            return out

        # count
        if action == "count":
            out = coll.count()
            cacher(str(out)); traceprint()
            return out

        # modify коллекции
        if action == "modify":
            out = coll.modify(name=new_name, metadata=new_meta)
            cacher("True"); traceprint()
            return out

        # delete_collection
        if action == "delete_collection":
            if client is None:
                raise ValueError("client required for delete_collection")
            out = client.delete_collection(coll_name)
            cacher("True"); traceprint()
            return out

        # query или get
        if action in ("query", "get"):
            include = fetch if isinstance(fetch, list) else [fetch]
            if include == ["all"]:
                include = ["ids", "documents", "metadatas", "embeddings", "distances"]

            params = {}
            if action == "query":
                params.update({
                    "query_embeddings": query_embeddings or [],
                    "where": _make_where(filters),
                    "n_results": n_results
                })
                if doc_contains:
                    params["where_document"] = {"$contains": doc_contains}
            else:  # get
                params.update({
                    "where": _make_where(filters),
                    "limit": limit,
                    "offset": offset
                })
                if doc_contains:
                    params["where_document"] = {"$contains": doc_contains}

            # общий include
            params["include"] = include
            # любые доп. kwargs
            params.update(kwargs)

            resp = (coll.query if action == "query" else coll.get)(**params)
            out = _extract(resp, include)
            # Закешировать строковое представление, как раньше
            cacher(str(out)); traceprint()
            return out

        raise ValueError(f"[coll_exec] Unsupported action: {action}")

    except Exception as e:
        # повторяем поведение старого coll_exec
        let_log(f"[coll_exec] Ошибка ({action}): {e}")
        cacher("None"); traceprint()
        return None

def sql_exec(query, params=(), fetchone=False, fetchall=False):
    let_log('ОЧЕРЕДЬ')
    let_log(query)
    """Выполняет SQL-запрос с поддержкой кэширования"""
    try:
        # Пытаемся получить из кэша
        cached = cacher()
        traceprint()
        let_log('КЭШ')
        let_log(type(cached))
        let_log(cached)
        if cached is not None:
            if cached == "None":
                return None
            # Разбираем кэшированный ответ
            type_part, _, value = cached.partition(":")
            let_log(type_part)
            let_log(value)
            if type_part == 'str': return value
            if value == '': return []
            if value == '[]': return []
            if value == '()': return ()
            if value == '{{}}': return {}
            return eval(value) # TODO: замени
            
        cursor = memory_sql.cursor()
        # Всегда передаем params в execute, даже если они пустые
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
        # Сохраняем в кэш с указанием типа
        if result is not None:
            if isinstance(result, int):
                cacher(f"int:{result}"); traceprint()
            elif isinstance(result, float):
                cacher(f"float:{result}"); traceprint()
            elif isinstance(result, bool):
                cacher(f"bool:{result}"); traceprint()
            elif isinstance(result, list):
                cacher(f"list:{','.join(map(str, result))}"); traceprint()
            elif isinstance(result, tuple):
                cacher(f"tuple:{','.join(map(str, result))}"); traceprint()
            else:
                cacher(f"str:{str(result)}"); traceprint()
        else:
            cacher("None"); traceprint()
            
        return result
        
    except Exception as e:
        let_log(f"Ошибка SQL-запроса: {query} с {params} — {e}")
        sys.exit(1)
        return None

def list_to_db(lst):
    """Преобразует список чисел в строку для хранения в SQLite."""
    if not lst:
        return ""
    return ",".join(map(str, lst)) # Пример: [1.23, -0.1, 0, 4] → "1.23,-0.1,0,4"

def db_to_list(db_str):
    """Преобразует строку из SQLite обратно в список чисел."""
    if not db_str:
        return []
    return [float(x) if "." in x else int(x) for x in db_str.split(",")]

def send_log_to_ui(message: str):
    try: ui_conn[2].put(message)
    except Exception as e: let_log(f"Failed to send log to UI: {e}")

chunk_size = 1000 # TODO:

get_provider_embs = None
ask_provider_model = None

def get_embs(text: str):
    # 1) кеширование
    send_log_to_ui('embeddings:\n' + text)
    cached = cacher()
    traceprint()
    if cached is not None:
        let_log("[Используются кэшированные эмбеддинги]")
        if cached == 'empty_embs': cached = []
        else: cached = list(eval(cached))
        let_log(cached)
        return cached

    start = time.time()

    # 2) начинаем с одного куска — весь текст
    pieces = [text]

    while True:
        try:
            # 3) пытаемся получить эмбеддинги для каждого куска
            #    предполагаем, что get_provider_embs принимает строку и возвращает List[float]
            all_embs = [get_provider_embs(p) for p in pieces]
            break
        except RuntimeError as e:
            if not 'ContextOverflowError' in str(e): raise
            let_log('дробление для ембеддингов')
            # 4) при переполнении -> дробим куски
            if all(len(p) == 1 for p in pieces):
                # все куски длины 1 — дальше дробить нечего
                raise
                sys.exit(1)
            new_pieces: List[str] = []
            for p in pieces:
                if len(p) > 1:
                    mid = len(p) // 2
                    new_pieces.append(p[:mid])
                    new_pieces.append(p[mid:])
                else:
                    new_pieces.append(p)
            pieces = new_pieces
        except Exception as e:
            traceprint()
            let_log(e)
            # если какая-то другая ошибка — пробрасываем
            raise
            sys.exit(1)

    elapsed = time.time() - start
    let_log(f'Получение эмбеддингов выполнено за {elapsed:.2f} секунд')

    # 5) объединяем списки эмбеддингов в один flat-список
    flat_embs: List[float] = []
    for emb in all_embs:
        flat_embs.extend(emb)
    # 6) сохраняем в кеш (преобразуем список в строку)
    if flat_embs == [] or flat_embs is None: cacher('empty_embs'); traceprint()
    else: cacher(str(flat_embs)); traceprint()
    return flat_embs

inst_start_tag = "[INST]"
inst_end_tag = "[/INST]"
sys_start_tag = "<sys>"
sys_end_tag = "</sys>"

def get_inst_tags():
    """Возвращает реальные INST теги из модели или дефолтные"""
    global inst_start_tag, inst_end_tag
    return inst_start_tag, inst_end_tag

def get_sys_tags():
    """Возвращает реальные системные теги из модели или дефолтные"""
    global sys_start_tag, sys_end_tag
    return sys_start_tag, sys_end_tag

def prepare_prompt_with_tags(prompt_text, is_system_prompt=False):
    """Подготавливает промпт с нужными тегами"""

    # Получаем теги
    inst_start, inst_end = get_inst_tags()
    sys_start, sys_end = get_sys_tags()

    # Формируем итоговый промпт
    if is_system_prompt:
        final_prompt = f"{sys_start}{prompt_text}{sys_end}"
    else:
        final_prompt = f"{inst_start}{prompt_text}{inst_end}"

    return final_prompt

    # Токенизированная версия
    '''
    if is_system_prompt:
        sys_start_tokens, sys_end_tokens = get_sys_tags()
        return sys_start_tokens + prompt_tokens + sys_end_tokens
    else:
        inst_start_tokens, inst_end_tokens = get_inst_tags()
        return inst_start_tokens + prompt_tokens + inst_end_tokens
    '''

def ask_model(prompt: str,
              limit: int = None,
              temperature: float = 0,
              **extra_params) -> str:
    prompt_text = prepare_prompt_with_tags(prompt)
    let_log(prompt_text)
    let_log(f'ВХОД {len(prompt_text)} токенов')
    let_log(f'ВХОД ({len(prompt_text)}):\n{prompt_text}')

    cached = cacher(); traceprint()
    if cached is not None:
        let_log("[Используется кэшированный ответ]")
        let_log(cached)
        if cached == '[ask_model error]':
            raise RuntimeError('ContextOverflowError')
        send_log_to_ui('model:\n' + cached)
        return cached

    generation_params: Dict[str, Any] = {
        "prompt": prompt_text,
        "temperature": temperature,
        "echo": False
    }
    if limit:
        generation_params["max_tokens"] = limit

    for name, val in extra_params.items():
        generation_params[name] = val

    start = time.time()
    try:
        generated = ask_provider_model(generation_params)
        cacher(generated)  # только при успехе
    except RuntimeError as e:
        if 'ContextOverflowError' in str(e):
            cacher('[ask_model error]')  # кладём только переполнение
            raise
        else:
            sys.exit(1)  # любая другая RuntimeError — выход
    except Exception:
        sys.exit(1)  # любая другая ошибка — выход

    elapsed = time.time() - start
    send_log_to_ui('model:\n' + generated)
    traceprint()
    let_log(generated)
    let_log(f'СГЕНЕРИРОВАНО {len(generated)} токенов за {elapsed:.2f}s')
    let_log(f'Генерация заняла {elapsed:.2f}s, вывод {len(generated)} токенов')
    return generated

def globalize_language_packet(language):
    """Загружает и токенизирует системные тексты для указанного языка"""
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

    # Токенизируем все строки в контейнере
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
    
    # Экспортируем атрибуты контейнера в глобальную область видимости
    for attr in dir(container):
        if attr.startswith('__'):
            continue
        value = getattr(container, attr)
        globals()[attr] = value

    let_log(f"Языковой пакет '{language}' загружен, переменные экспортированы")

def highest_or_lowest():
    if global_state.parents_id == '/': return 'highest'
    answer = sql_exec('SELECT * FROM tries WHERE parents_id = (SELECT MAX(parents_id) FROM tries)', fetchone=True)
    if len(answer) == len(global_state.parents_id): return 'lowest'
    return False

input_info_loaders = {}
'''
default_handlers = {
    'docx': getattr(info_loaders, 'process_docx', None),
    'doc': getattr(info_loaders, 'process_docx', None),
    'pdf': getattr(info_loaders, 'process_pdf', None),
    #'png': getattr(info_loaders, 'process_image', None),
    #'jpg': getattr(info_loaders, 'process_image', None),
    #'jpeg': getattr(info_loaders, 'process_image', None),
    'txt': getattr(info_loaders, 'process_text', None)
}
'''
info_loaders = None
def load_info_loaders(info_loaders_names):
    """
    Заполняет глобальный словарь input_info_loaders на основе переданного словаря.
    Если имя функции совпадает с дефолтным, то используется соответствующий дефолтный обработчик.
    Если имя другое, то функция ищется в модуле info_loaders, А ДОЖНА НЕ ТАМ.
    """
    global input_info_loaders
    import info_loaders
    for ext, func_name in info_loaders_names.items():
        handler = getattr(info_loaders, func_name, None)
        if handler: input_info_loaders[ext] = handler
        else: let_log(f"Ошибка: функция {func_name} не найдена в модуле info_loaders.")

def split_text_with_cutting(text, min_chunk_percentage=0.8):
    if not isinstance(text, str) or not text.strip():
        return None
    
    delimiters = ['\n\n', '\n', '. ', '! ', '? ', '; ', ': ', ' ', '']
    chunks = []
    current_pos = 0
    text_length = len(text)
    
    while current_pos < text_length:
        end_pos = min(current_pos + chunk_size, text_length)
        
        if end_pos == text_length:
            chunk = text[current_pos:end_pos]
            if chunk.strip():
                chunks.append(chunk)
            break
        
        split_pos = None
        for delimiter in delimiters:
            candidate_pos = text.rfind(delimiter, current_pos, end_pos)
            if candidate_pos != -1:
                current_chunk_size = candidate_pos + len(delimiter) - current_pos
                if current_chunk_size >= chunk_size * min_chunk_percentage:
                    split_pos = candidate_pos + len(delimiter)
                    break
        
        if split_pos is None:
            split_pos = end_pos
        
        chunk = text[current_pos:split_pos]
        if chunk.strip():
            chunks.append(chunk)
        current_pos = split_pos
    
    return chunks if chunks else None

def upload_user_data(files_list):
    for filename in files_list:
        let_log(filename)
        _, file_extension = os.path.splitext(filename)
        result = cacher(); traceprint()
        if result is None:
            global input_info_loaders
            if not input_info_loaders: load_info_loaders(default_handlers_names)
            try: result = input_info_loaders.get(file_extension[1:])(filename, input_info_loaders)
            except:
                try: result = input_info_loaders.get('txt')(filename, input_info_loaders)
                except: continue
            cacher(str(result)); traceprint()
        
        let_log('разбиение')
        result = split_text_with_cutting(result) # разбиение текста
        let_log('текст разбит')
        if not result: continue
        for t in range(len(result)):
            set_common_save_id()
            coll_exec(
                action="add",
                coll_name="user_collection",
                ids=[get_common_save_id()],
                embeddings=[get_embs(result[t])],
                metadatas=[{'name': filename, 'part': t + 1}],
                documents=[result[t]]
            )

# Функция для поиска try_id с учётом полного пути (task + global_state.parents_id + try_id)
def find_try_id_with_path(task, parid, successful_try_id): return sql_exec('SELECT try_id, parents_id FROM tries WHERE task = ? AND parents_id = ? AND successful_try_id = ?', (task, parid, successful_try_id), fetchone=True)

# Функция для поиска всех try_id, у которых parents_id начинается с заданного префикса
def find_child_try_ids(parid): return sql_exec('SELECT try_id, parents_id FROM tries WHERE parents_id LIKE ?', (parid + '/%',), fetchall=True)
# Удаление записей по try_id и parents_id
def delete_try_ids_with_parents_id(try_ids_with_parents):
    for try_id, parid in try_ids_with_parents:
        ids_to_delete = coll_exec(
            action="query",
            coll_name="milana_collection",
            query_embeddings=[[]],  # пустой запрос, как [""] раньше
            filters={"try_id": try_id, "parents_id": parid},
            fetch="ids",
            n_results=None,
            flatten=True
        )
        if ids_to_delete:
            coll_exec(
                action="delete",
                coll_name="milana_collection",
                ids=ids_to_delete
            )

# Основная логика удаления с учётом полного пути
def delete_hierarchy(successful_try_ids):
    to_delete = [] # список пар (try_id, parents_id), которые нужно удалить
    # Для каждого successful_try_id ищем try_id и его parents_id
    # if not max_try_id: max_try_id = 0
    for successful_try_id in successful_try_ids:
        try_id_with_path = find_try_id_with_path(global_state.main_now_task, global_state.parents_id, successful_try_id)
        if try_id_with_path:
            try_id, current_parents_id = try_id_with_path
            stack = [(try_id, current_parents_id)]  # создаём стек для обработки всех дочерних элементов
            while stack:
                current_try_id, current_parents_id = stack.pop()
                to_delete.append((current_try_id, current_parents_id))
                # Ищем дочерние try_id по текущему parents_id (включая уровень ниже)
                child_try_ids = find_child_try_ids(f"{current_parents_id}/{current_try_id}")
                # Добавляем все найденные дочерние try_id в стек
                stack.extend(child_try_ids)
    # Теперь, когда мы собрали все try_id и их parents_id, которые нужно удалить, удалим их
    if to_delete: delete_try_ids_with_parents_id(to_delete)

def select_little_best(task, fr, sr):
    while True:
        traceprint()
        try: return int(ask_model(select_little_best_text + task + solution_text_1 + fr + solution_text_2 + sr))
        except: pass

def is_similar(task, first_result, second_result):
    while True:
        traceprint()
        try: return int(ask_model(is_similar_text + task + solution_text_1 + fr + solution_text_2 + sr))
        except: pass

def litle_model_sorting():
    # дополни ембеддингами
    first_try_id, first_result = sql_exec('SELECT try_id, result FROM tries WHERE task = ? AND parents_id = ? AND successful_try_id = 1', (global_state.main_now_task, global_state.parents_id), fetchone=True)
    first_result = first_result
    results = [[first_try_id,],] # лучше это заменить на словарь
    for i in range(global_state.task_retry - 1):
        second_try_id, second_result = sql_exec('SELECT try_id, result FROM tries WHERE task = ? AND parents_id = ? AND successful_try_id = ?', (global_state.main_now_task, global_state.parents_id, i + 1), fetchone=True)
        second_result = second_result
        results.append([second_try_id,])
        best_num = select_little_best(global_state.main_now_task, first_result, second_result)
        if best_num == 1:
            fl = results[-2]
            sl = results[-1]
            results[-2] = sl
            results[-1] = fl
        elif best_num == 2: first_try_id = second_try_id; first_result = second_result
        elif best_num == 0:
            if len(results[-2]) == 2: results[-1].append(results[-2][1])
            else: results[-1].append(first_try_id)
            first_try_id = second_try_id; first_result = second_result
    if most_often:
        similar_results = {}
        not_similar_results = []
        for i in results:
            try: lnk = i[1]
            except:
                not_similar_results.append(i[0])
                continue
            try: similar_results[lnk] += 1
            except:
                del not_similar_results[lnk]
                similar_results[lnk] = 1
        for i in similar_results:
            first_result = sql_exec('SELECT result FROM tries WHERE task = ? AND parents_id = ? AND try_id = ?', (global_state.main_now_task, global_state.parents_id, i), fetchone=True)
            # после цикла должен удалить айди из непохожих по списку
            to_del = []
            for j in not_similar_results:
                second_result = sql_exec('SELECT result FROM tries WHERE task = ? AND parents_id = ? AND try_id = ?', (global_state.main_now_task, global_state.parents_id, j), fetchone=True)
                if is_similar(global_state.main_now_task, first_result, second_result): to_del.append(j); similar_results[i] += 1
            for j in to_del: del not_similar_results[j]
        for id1 in not_similar_results:
            first_result = sql_exec('SELECT result FROM tries WHERE task = ? AND parents_id = ? AND try_id = ?', (global_state.main_now_task, global_state.parents_id, id1), fetchone=True)
            for id2 in not_similar_results:
                # Пропускаем сравнение текста с самим собой и повторные пары
                if id1 != id2:
                    second_result = sql_exec('SELECT result FROM tries WHERE task = ? AND parents_id = ? AND try_id = ?', (global_state.main_now_task, global_state.parents_id, id2), fetchone=True)
                    # Вызываем функцию is_similar для двух текстов
                    if is_similar(global_state.main_now_task, first_result, second_result):
                        try: similar_results[id1]; similar_results[id1] += 1
                        except: similar_results[id1] = 1
        most_often_result = max(similar_results, key=similar_results.get) # выбирает
        if need_best_result:
            if results[-1][0] != most_often_result:
                first_result = sql_exec('SELECT result FROM tries WHERE task = ? AND parents_id = ? AND try_id = ?', (global_state.main_now_task, global_state.parents_id, results[-1][0]), fetchone=True)
                second_result = sql_exec('SELECT result FROM tries WHERE task = ? AND parents_id = ? AND try_id = ?', (global_state.main_now_task, global_state.parents_id, most_often_result), fetchone=True)
                best_num = select_little_best(global_state.main_now_task, first_result, second_result)
                if best_num < 2: best_result = results[-1][0]
                elif best_num == 2: best_result = most_often_result
        else: best_result = most_often_result
    elif need_best_result: best_result = results[-1][0]
    if need_del_not_best:
        res_ids = sql_exec('SELECT try_id FROM tries WHERE task = ? AND parents_id = ? AND successful_try_id <> ?', (global_state.main_now_task, global_state.parents_id, best_result), fetchall=True)
        delete_hierarchy(successful_try_ids)
    else:
        sql_exec('UPDATE tries SET selected_best = ? WHERE try_id = ? AND parents_id = ?', (True, best_result, global_state.parents_id))
        # получает 2 объекта (с чатом и результатом)
        cbid = coll_exec(
            action="query",
            coll_name="milana_collection",
            query_embeddings=[[0.0] * 1536],  # или просто []
            filters={"$and": [{"try_id": try_id}, {"parents_id": global_state.parents_id}]},
            n_results=2,
            fetch=["ids", "metadatas"],
            first=False
        )

        # Обработка результатов
        document_ids = cbid['ids']
        metadata = cbid['metadatas']
        metadata[0]['selected_best'] = True
        metadata[1]['selected_best'] = True

        # Обновление
        coll_exec(
            action="update",
            coll_name="milana_collection",
            ids=document_ids,
            metadatas=metadata
        )

def mod_loader(adrs):
    loaded_modules = []
    for mod_file in adrs:
        try:
            let_log(f"Processing: {mod_file}")
            if not os.path.isfile(mod_file):
                let_log(f"Файл {mod_file} не найден")
                continue

            # Пытаемся загрузить локализацию ТОЛЬКО если язык не английский
            current_lang = globals().get('language', 'en')
            locale_data = {}
            
            if current_lang != 'en':
                lang_file = mod_file.replace('.py', '_lang.py')
                if os.path.isfile(lang_file):
                    try:
                        spec = importlib.util.spec_from_file_location('lang_module', lang_file)
                        lang_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(lang_module)
                        
                        if hasattr(lang_module, 'locales'):
                            locale_data = lang_module.locales.get(current_lang, {})
                    except Exception as e:
                        let_log(f"⚠ Ошибка при загрузке локализации {lang_file}: {e}")

            # Читаем содержимое файла для поиска первых строк
            with open(mod_file, encoding='utf-8') as f:
                file_contents = f.read()

            # Парсим первые строки из файла (многострочный комментарий в начале)
            command_name = None
            description = None
            
            # Новый улучшенный regex для обработки разных форматов многострочных комментариев
            doc_match = re.match(r'^\s*[\'"]{3}\s*\n\s*([^\n]+)\n\s*([^\n]+)', file_contents)
            if not doc_match:
                # Альтернативный вариант, если нет переноса строки после открывающих кавычек
                doc_match = re.match(r'^\s*[\'"]{3}\s*([^\n]+)\n\s*([^\n]+)', file_contents)
            
            if doc_match:
                command_name = doc_match.group(1).strip()
                description = doc_match.group(2).strip()
            
            # Если есть локализация и язык не английский, берем оттуда
            if current_lang != 'en' and locale_data:
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
            should_initialize = re.search(
                r"if\s+not\s+hasattr\s*\(\s*main\s*,\s*['\"]attr_names['\"]\s*\)", 
                file_contents
            )
            
            if should_initialize:
                try:
                    main_func(None)  # Инициализация атрибутов по умолчанию
                    attr_list = getattr(main_func, 'attr_names', [])
                    
                    # Применяем локализацию ТОЛЬКО если язык не английский и есть данные
                    if current_lang != 'en' and locale_data:
                        for attr in attr_list:
                            localized_key = f'main.{attr}'
                            if localized_key in locale_data:
                                setattr(main_func, attr, locale_data[localized_key])
                                let_log(f"✓ Локализация: {attr} в {mod_file}")
                except Exception as e:
                    let_log(f"⚠ Ошибка при инициализации {mod_file}: {e}")
                    continue

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

    # Пути
    common_path = os.path.join(folder_path, 'system_tools')
    ivan_path   = os.path.join(folder_path, f'system_tools{slash}ivan')
    milana_path = os.path.join(folder_path, f'system_tools{slash}milana')

    # Собираем списки файлов
    common_files = fpf(common_path)
    ivan_files   = fpf(ivan_path)
    milana_files = fpf(milana_path)

    # Загружаем модули
    common_modules = mod_loader(common_files)
    ivan_modules   = mod_loader(ivan_files)
    milana_modules = mod_loader(milana_files)

    # Глобализуем общие И ivan-модули
    globalize_by_filename(common_modules, common_files)
    globalize_by_filename(ivan_modules, ivan_files)
    globalize_by_filename(milana_modules, milana_files)

    # Формируем словари с ключом — кортеж токенов имени команды
    def to_dict(modules):
        d = {}
        for cmd_t, desc_tokens, func in modules:
            # Добавляем в system_tools_keys команды
            global_state.system_tools_keys.append(cmd_t)
            d[cmd_t] = (desc_tokens, func)
        return d

    common_dict = to_dict(common_modules)
    ivan_dict = to_dict(ivan_modules)
    milana_dict = to_dict(milana_modules)

    # Выводим отладочную информацию
    let_log("\nЗагруженные системные команды:")
    for cmd in global_state.system_tools_keys:
        let_log(cmd)

    return (
        {**common_dict, **ivan_dict},   # common + ivan
        {**common_dict, **milana_dict}  # common + milana
    )

def make_tools_str(td):
    ts = ''
    for t in td: ts += t[0] + ' ' + t[1] + slash_n
    return ts

def gigo(base_task, roles=[]):
    try: questions = ask_model(gigo_questions + base_task).splitlines()
    except:
        base_task = text_cutter(base_task)
        questions = ask_model(gigo_questions + base_task).splitlines()
    additional_info = ''
    for quest in questions:
        answer = librarian(quest)
        let_log(found_info_1)
        if answer != found_info_1: additional_info += '\n' + answer
    if additional_info != '': additional_info = gigo_found_info + additional_info
    else: additional_info = gigo_not_found_info
    minds = ''
    if not roles: roles = [gigo_dreamer, gigo_realist, gigo_critic]
    for role in roles:
        try: minds += '\n' + ask_model(gigo_role_answer_1 + role + gigo_role_answer_2 + base_task + '\n' + additional_info)
        except: minds += '\n' + ask_model(gigo_role_answer_1 + role + gigo_role_answer_2 + base_task + '\n' + text_cutter(additional_info))
    try: plan = ask_model(gigo_make_plan + base_task + '\n' + gigo_reaction + minds + '\n' + additional_info)
    except: plan = ask_model(gigo_make_plan + base_task + text_cutter(minds) + '\n' + text_cutter(additional_info))
    return gigo_return_1 + ':\n' + base_task + '\n' + gigo_return_2 + ':\n' + plan

def get_dialog_history(sid):
    history = sql_exec('SELECT history FROM chats WHERE chat_id=?', (sid,), fetchone=True)
    return history

def text_cutter(text):
    let_log('КАТТЕР ВЫЗВАН')
    let_log(text)
    text2 = text[len(text) // 2:]
    try:
        text2 = text2[min(text2.find(slash_n), text2.find(dot_space)) + 2:]  
    except RuntimeError as e:
        if 'ContextOverflowError' in str(e):
            let_log('ошибка каттера'); let_log(e)
            pass
        else:
            raise  # пробрасываем дальше
    except Exception as e:
        traceprint()
        let_log(e)
        raise  # пробрасываем дальше

    text1 = text[:text.find(text2)]
    traceprint()

    try:
        text1 = ask_model(summarize_prompt + text1)
    except RuntimeError as e:
        if 'ContextOverflowError' in str(e):
            text1 = text_cutter(text1)
        else:
            sys.exit(1)  # останавливаем процесс
    except Exception:
        sys.exit(1)

    traceprint()
    try:
        text2 = ask_model(summarize_prompt + text2)
    except RuntimeError as e:
        if 'ContextOverflowError' in str(e):
            text2 = text_cutter(text2)
        else:
            sys.exit(1)
    except Exception:
        sys.exit(1)

    return text1 + just_space + text2

# лучше использовать токинайзер из модели, чтобы не скачивать лишнего
# резулт тру фолс может быть излишеством
# нужен пробел после цифр во избежание проблем
# проблемы с выводом, количество токенов не вызывает ошибку, но ответ просто срезан
# нужно делить сохранение на половину контекста
# также нужно сделать ограничение по контексту ибо слишком большой сбивает модель
# квадратные скобки лучше заменить на круглые чтобы модель их не писала и добавить слово "здесь"
# текст желательно сжиматься не должен, особенно, если он не превышает saved_message_ctx, может добавить сжатие опционально

def save_emb_dialog(text, tag, result=None):
    # может здесь формулировать ответы, отсекая ненужный текст и файлы?
    # как-то изображения надо добавить
    # саксессфул должен быть bool в retries, а retries в бд
    # должен получать из retries (их добавление нужно обновить чтобы не увеличивать счёт при неудачной попытке)
    # может быть id задачи и сама задача
    # строки выше получают значения из общих переменных которые определяются при работе а не из бд
    # нужно сделать ветки чтобы отсекать неправильные подветки даже с правильными решениями внутри
    # ветка и подветка создаётся вместе с главной задачей в метаданных, внутри, или хз ещё где
    # нужно реализовать retries в бд
    parts = get_now_try().split('/')
    if len(parts) > 1:
        global_state.parents_id = '/'.join(parts[:-1])
    else:
        global_state.parents_id = '/'
    global_state.now_try = parts[-1]
    set_common_save_id()

    if result:
        coll_exec(
            action="add",
            coll_name="milana_collection",
            ids=[get_common_save_id()],
            embeddings=[get_embs(result)],
            metadatas=[{
                "done": tag,
                "parents_id": global_state.parents_id,
                "try_id": global_state.now_try,
                "result": True
            }],
            documents=[result]
        )
    else: # Поиск "Краткая история диалога"
        if save_emb_dialog_history in text:
            start_idx = text.find(save_emb_dialog_history) + len(save_emb_dialog_history) # это оптимизируй
            end_idx = text.find(slash_n, start_idx)
            if end_idx == -1: # Если строка до конца текста
                end_idx = len(text)
            history_text = text[start_idx:end_idx]
            # Получение тезисов через ask_model
            while True:
                traceprint()
                try: response = ask_model(save_emb_dialog_mark_thesis_1 + history_text + save_emb_dialog_mark_thesis_2); break
                except: history_text = text_cutter(history_text)
            theses = response.split(save_emb_dialog_thesis)
            for thesis in theses:
                set_common_save_id()
                coll_exec(
                    action="add",
                    coll_name="milana_collection",
                    ids=[get_common_save_id()],
                    embeddings=[get_embs(thesis)],
                    metadatas=[{
                        "done": tag,
                        "parents_id": global_state.parents_id,
                        "try_id": global_state.now_try,
                        "result": False
                    }],
                    documents=[thesis]
                )
        # Обработка оставшегося текста как диалога
        dialogue_start_idx = text.find(slash_n)
        if dialogue_start_idx != -1:
            dialogue_text = text[dialogue_start_idx:]
            # Разбиваем на группы с помощью ask_model
            while True:
                traceprint()
                try: response = ask_model(save_emb_dialog_mark_group + dialogue_text); break
                except: dialogue_text = text_cutter(dialogue_text)
            groups = response.split(save_emb_dialog_group)
            for group in groups:
                set_common_save_id()
                coll_exec(
                    action="add",
                    coll_name="milana_collection",
                    ids=[get_common_save_id()],
                    embeddings=[get_embs(group)],
                    metadatas=[{
                        "done": tag,
                        "parents_id": global_state.parents_id,
                        "try_id": global_state.now_try,
                        "result": False
                    }],
                    documents=[group]
                )

def get_input_message(command=None, timeout=None, wait=False):
    cached = cacher(); traceprint()
    if cached is not None: return eval(cached)
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
            try:
                answer = ui_conn[0].get(block=(timeout is not None), timeout=timeout); break
            except Empty: pass
            except Exception as e: let_log(f"Ошибка при получении сообщения: {e}"); break
    cacher(str(answer)); traceprint()
    return answer

def send_output_message(text=None, attachments=None, command=None):
    cached = cacher(); traceprint()
    if cached is not None: return
    message_data = {
        'text': text or '',
        'attachments': attachments or None,
        'command': command
    }
    try:
        ui_conn[1].put(message_data)
    except Exception as e:
        let_log(f"Ошибка при отправке сообщения: {e}")
        cacher('False'); traceprint()
        return
    cacher('True'); traceprint()

# Опционально: нормализация (если захочешь активировать потом)
def normalize_command(cmd_str):
    return cmd_str.replace('→', '').strip().lower()

def tools_selector(text, sid):
    """Выбирает инструменты на основе команды в виде строки"""
    let_log("=== [TOOLS_SELECTOR ЗАПУЩЕН] ===")
    # 2. Поиск первого !!!
    first_exclamation = text.find("!!!")
    let_log(f"[3] Первый !!! найден на позиции: {first_exclamation}")
    if first_exclamation == -1 or first_exclamation > 4:
        let_log("[ERROR] Первый !!! не найден или далеко от начала")
        let_log('первое возвращение')
        return None

    # 3. Поиск второго !!!
    second_exclamation = text.find("!!!", first_exclamation + 3)
    let_log(f"[4] Второй !!! найден на позиции: {second_exclamation}")
    if second_exclamation == -1:
        let_log("[ERROR] Второй !!! не найден")
        let_log('второе возвращение')
        return None

    # 4. Извлечение команды и содержимого
    command = text[first_exclamation + 3 : second_exclamation].strip().replace('\\_','_') # TODO: потом вообще от _ избавиться
    content = text[second_exclamation + 3 :].strip()

    let_log(f"[5] Извлечённая команда: «{command}»")
    let_log(f"[6] Содержимое после команды:\n{content}")

    # 5. Проверка кэша
    cached = cacher(); traceprint()
    let_log(f"[7] Содержимое кэша: {cached}")
    was_system_in_cache = False
    if cached is not None:
        if cached == "SYSTEM":
            was_system_in_cache = True
            let_log("[7.1] Команда системная — выполняем заново")
        else:
            let_log("[7.2] Команда не системная — возвращаем кэш")
            return cached
    
    # 6. Проверка, является ли команда системной
    sys_keys_str = [k for k in global_state.system_tools_keys]
    is_system = any(command in k for k in sys_keys_str)
    let_log(f"[8] Системные команды:\n{sys_keys_str}")
    let_log(f"[9] Команда является системной? {is_system}")

    # 7. Поиск команды среди доступных
    now_commands = global_state.tools_commands_dict.get(sid, {})
    let_log(f"[10] Команды для SID {sid}:")

    command_keys_str = {}
    for k, func in now_commands.items():
        key_str = k.strip()
        let_log(f"  →  {key_str}")
        command_keys_str[key_str] = func

    # Попытка найти подходящий ключ через in
    found_key = None
    for key in command_keys_str:
        if command in key or key in command:
            found_key = key
            break

    # Если не нашли — попытка fuzzy-поиска
    if not found_key:
        possible_matches = difflib.get_close_matches(command, list(command_keys_str.keys()), n=1, cutoff=0.7)
        if possible_matches:
            found_key = possible_matches[0]
            let_log(f"[10.1] Fuzzy match найден: {found_key}")
        else:
            let_log("[ERROR] Команда не найдена среди доступных")
            let_log('нет команды')
            return tool_selector_return_2

    command_func = command_keys_str[found_key]
    is_system = any(found_key in k for k in sys_keys_str)
    if is_system and not was_system_in_cache:
        let_log("[13] Отмечаем как системную в кэше")
        cacher("SYSTEM"); traceprint()
    # 8. Выполнение команды
    let_log("[11] Выполняем команду...")
    let_log(command_func)
    result = command_func[1](content)
    let_log(f"[12] Результат выполнения:\n{result}")

    # 9. Кэширование результата
    if not is_system:
        let_log("[13] Кэшируем результат (не системная команда)")
        cacher(str(result)); traceprint()
    let_log("=== [TOOLS_SELECTOR ЗАВЕРШЁН] ===")

    return result

now_agent_id = ''

def agent_func(text, agent_number):
    # надо сокращать ещё когда превышен не лимит а какое-то количество ибо модель может начать писать бред
    # нужно резать в первую очередь ответы инструментов ибо сторонние разработчики могут перегрузить модель
    global_state.stop_agent = False
    talk_prompt = text
    sid = global_state.conversations - agent_number
    global now_agent_id
    now_agent_id = str(sid) # для cmd
    if agent_number: # если 1 это милана!!!
        you = operator_role_text
        msg_from = worker_role_text
    else:
        you = worker_role_text
        msg_from = operator_role_text
    while not global_state.stop_agent:
        let_log(f"[DEBUG] agent_number={agent_number}, global_state.conversations={global_state.conversations}, sid={sid}")
        print(sid)
        prompt, history = sql_exec('SELECT prompt, history FROM chats WHERE chat_id=?', (sid,), fetchone=True) # TODO: ВОЗВРАЩАЕТ None
        prompt = prompt
        history = history
        last_talk_prompt = talk_prompt
        traceprint()
        try: talk_prompt = ask_model(system_role_text + prompt + history + msg_from + talk_prompt + you)
        except:
            history = start_dialog_history + text_cutter(history)
            traceprint()
            try: talk_prompt = ask_model(system_role_text + prompt + history + msg_from + talk_prompt + you)
            except:
                # что тут?
                traceprint()
                try: talk_prompt = ask_model(system_role_text + prompt + history + msg_from + talk_prompt + you)
                except: let_log('ВСЁ ЕЩЁ ПРЕВЫШЕНИЕ')
        history = history + msg_from + last_talk_prompt + talk_prompt
        sql_exec("UPDATE chats SET history = ? WHERE chat_id = ?", (history, sid))
        #let_log(talk_prompt)
        answer = tools_selector(talk_prompt, sid)
        if answer: talk_prompt = answer; msg_from = func_role_text
        else: break
        #let_log(talk_prompt)
    global_state.stop_agent = False
    return talk_prompt

def worker(really_main_task):
    while True:
        global_state.retries = []
        global_state.conversations = 0
        global_state.tools_commands_dict = {}
        global_state.dialog_state = True
        global_state.main_now_task = really_main_task
        talk_prompt = start_dialog(really_main_task)
        need_start_new_dialog = False # TODO: это осторожно выпилить
        need_work = True
        if not global_state.dialog_state:
            let_log('записи удалены')
            sql_exec('DELETE FROM chats WHERE chat_id = ?', (global_state.conversations,))
            sql_exec('DELETE FROM chats WHERE chat_id = ?', (global_state.conversations - 1,))
            let_log('Диалог завершился не начавшись')
            need_work = False
            while True:
                if get_input_message() is None: break
            send_output_message(text=global_state.dialog_result, command='end')
            user_message = get_input_message(wait=True)
            really_main_task = user_message['text']
            if not really_main_task: raise # TODO: доделай тут или общение с пользователем или просто сообщение что текста нет
            if user_message['attachments']:
                send_output_message(text=start_load_attachments_text)
                if not input_info_loaders: load_info_loaders(default_handlers_names)
                upload_user_data(user_message['attachments'])
                send_output_message(text=end_load_attachments_text)
        while need_work:
            talk_prompt = agent_func(talk_prompt, 0) # ivan
            if need_start_new_dialog: talk_prompt = make_spec_first
            talk_prompt = agent_func(talk_prompt, 1) # milana
            if not global_state.dialog_state:
                # тут поработай с инициализацией и попытками
                sql_exec('DELETE FROM chats WHERE chat_id = ?', (global_state.conversations,))
                sql_exec('DELETE FROM chats WHERE chat_id = ?', (global_state.conversations - 1,))
                global_state.conversations -= 2
                print(global_state.tools_commands_dict)
                global_state.tools_commands_dict.popitem()
                global_state.tools_commands_dict.popitem()
                #global_state.tools_commands_dict = global_state.tools_commands_dict[:-3]
                let_log(global_state.dialog_result)
                let_log(global_state.conversations)
                # ВМЕСТО ВСТАВКИ ОБНОВЛЯЕТ, И МБ ОБНОВЛЯЕТ ПОПЫТКУ ВЫШЕ ПО ИЕРАРХИИ
                # нужен id общей задачи (мб retry_id)
                # нужно еще задачу в трайс и хрому записать и извлекать из трайс если надо
                # после выбора лучшего варианта 'done' остальноых меняются но не на incorrect
                # получает кол-во успешных попыток из бд
                parts = get_now_try().split('/')
                if len(parts) > 1: global_state.parents_id = '/'.join(parts[:-1])
                else: global_state.parents_id = '/'
                # % для сравнением с началом строки, мало ли будут совпадения позже
                len_retries = sql_exec('SELECT COUNT(try_id) FROM tries WHERE task = ? AND try_id LIKE ? AND result IS NOT NULL AND result <> ""', (global_state.main_now_task, f"{global_state.parents_id}%"), fetchall=True)
                # добавь опции ретраев на определенных уровнях
                if retry_lowest or retry_highest or retry_all:
                    if retry_all: rt = True
                    else:
                        answer = highest_or_lowest()
                        if answer == 'lowest' and retry_lowest: rt = True
                        elif answer == 'highest' and retry_highest: rt = True
                    if rt:
                        if global_state.task_retry == len_retries:
                            if little_model: litle_model_sorting()
                            else: pass
                        else:
                            global_state.retrying = True
                            start_dialog('')
                            talk_prompt = make_spec_first
                            continue
                if global_state.conversations <= 0:
                    pass # что-то делает
                    while True:
                        if get_input_message() is None: break
                    send_output_message(text=global_state.dialog_result, command='end')
                    user_message = get_input_message(wait=True)
                    really_main_task = user_message['text']
                    if not really_main_task: raise # TODO: доделай тут или общение с пользователем или просто сообщение что текста нет
                    if user_message['attachments']:
                        send_output_message(text=start_load_attachments_text)
                        if not input_info_loaders: load_info_loaders(default_handlers_names)
                        upload_user_data(user_message['attachments'])
                        send_output_message(text=end_load_attachments_text)
                    let_log('тут')
                    break
                else:
                    talk_prompt = global_state.dialog_result + udp_spec_if_needed # тут ошибка, иван не создан
                    global_state.dialog_state = True
                    continue

# можно в случае ошибки ролбэк сделать тогда замуск инициалайз с трэйсбэком и траем

cache_path = ''

def cacher(info=None, rollback=False):
    """Получает значение из кэша или сохраняет новое значение.
    
    Если rollback=True, выполняет откат последних записей.
    Если происходит ошибка, инициализирует базу данных.
    """
    db_name = "cache_database.db"
    global cache_counter

    def initialize_cache_db():
        """Инициализация базы данных и создание таблицы, если она не существует."""
        cache_conn = connect(cache_path)
        cache_cursor = cache_conn.cursor()
        cache_cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                id INTEGER PRIMARY KEY,
                value TEXT
            )
        ''')
        cache_conn.commit()
        cache_conn.close()

    def rollback_cache(num_records):
        """Удаляет последние num_records записей из кэша."""
        cache_conn = connect(cache_path)
        cache_cursor = cache_conn.cursor()

        # Удаляем последние num_records записей
        cache_cursor.execute('DELETE FROM cache WHERE id IN (SELECT id FROM cache ORDER BY id DESC LIMIT ?)', (num_records,))
        cache_conn.commit()
        cache_conn.close()

    try:
        # Инициализация базы данных
        initialize_cache_db()

        cache_conn = connect(cache_path)
        cache_cursor = cache_conn.cursor()

        if info is None:
            # Попытка получить значение из кэша
            cache_cursor.execute('SELECT value FROM cache WHERE id = ?', (cache_counter,))
            result = cache_cursor.fetchone()
            
            if result is None:
                return None  # Если записи нет, возвращаем None

            cache_counter += 1  # Увеличиваем счетчик
            return result[0]  # Возвращаем значение
        else:
            # Сохраняем новое значение в кэше
            cache_cursor.execute('INSERT INTO cache (id, value) VALUES (?, ?)', (cache_counter, info))
            cache_conn.commit()
            cache_counter += 1  # Увеличиваем счетчик

        if rollback:
            rollback_cache(1)  # Выполняем откат последней записи

    except OperationalError:
        # Если произошла ошибка, инициализируем базу данных
        initialize_cache_db()
    finally:
        cache_conn.close()

default_handlers_names = { # это из настроек выгружается
    'doc': 'process_docx',
    'docx': 'process_docx',
    'txt': 'process_text',
    'pdf': 'process_pdf',
    'png': 'process_image',
    'jpg': 'process_image',
    'jpeg': 'process_image',
    }

actual_handlers_names = {}
another_tools_files_addresses = []

ui_conn = None

def load_chat_settings(chat_id):
    """Загрузка настроек чата из SQLite БД"""
    settings = {}
    
    # Загрузка основных настроек
    settings_rows = sql_exec("SELECT key, value FROM settings", fetchall=True)
    if settings_rows:
        let_log(settings_rows)
        let_log(type(settings_rows))
        settings.update({row[0]: row[1] for row in settings_rows})
    
    # Загрузка модулей
    another_tools_files = []
    
    # Дефолтные модули
    default_mods = sql_exec("SELECT adress FROM default_mods WHERE enabled=?", (1,), fetchall=True)
    if default_mods:
        another_tools_files.extend([row[0] for row in default_mods])
    
    # Кастомные модули
    custom_mods = sql_exec("SELECT adress FROM custom_mods", fetchall=True)
    if custom_mods:
        another_tools_files.extend([row[0] for row in custom_mods])
    
    settings["another_tools"] = another_tools_files
    
    return settings

def load_initial_data(chat_id):
    """Загрузка начальных данных: задачи и вложений"""
    # Загрузка задачи (первого сообщения)
    task = sql_exec("SELECT text FROM messages WHERE id=?", 
                   (1,), fetchone=True)
    task = task if task else ""
    
    # Загрузка вложений
    attachments = sql_exec("SELECT attachments FROM messages WHERE id=?", 
                         (1,), fetchone=True)
    
    if attachments and attachments:
        try:
            attachments = eval(attachments)  # Безопаснее чем json.loads для списка путей
        except:
            attachments = []
    else:
        attachments = []
    
    return task, attachments

chat_path = ''

def initialize_work(chat_id, input_queue, output_queue, stop_event, pause_event, log_queue):
    global memory_sql
    global actual_handlers_names, another_tools_files_addresses
    global token_limit, most_often, max_tasks
    global client
    global milana_collection
    global user_collection
    global ui_conn
    global cache_path
    global ask_provider_model, get_provider_embs # Для модели
    global language
    global chat_path

    ui_conn = [input_queue, output_queue, log_queue]
    global_state.task_retry

    # === Загружаем параметры чата ===
    chat_path = os.path.join(folder_path, "data", "chats", chat_id)
    cache_path = os.path.join(chat_path, "cache.db")
    let_log(chat_id)
    let_log(input_queue)
    let_log(output_queue)
    let_log(pause_event)
    let_log(chat_path)
    # === Инициализация ChromaDB ===
    chroma_path = os.path.join(chat_path, "chroma_db")
    client = chromadb.PersistentClient(path=chroma_path, settings=Settings(allow_reset=True))
    client.reset()
    milana_collection = client.get_or_create_collection(name="milana_collection")
    user_collection = client.get_or_create_collection(name="user_collection")

    # === Подготовка SQLite БД ===
    db_path = os.path.join(chat_path, "chatsettings.db")
    let_log(db_path)
    

    #try: os.remove(db_path)
    #except: pass
    
    #try: os.remove(cache_path)
    #except: pass
    #try: os.remove(os.path.join(chat_path, "chroma_db"))
    #except: pass

    memory_sql = connect(db_path)

    sql_exec('''CREATE TABLE IF NOT EXISTS chats (
        chat_id INT PRIMARY KEY,
        prompt TEXT,
        history TEXT
    )''')

    sql_exec('''CREATE TABLE IF NOT EXISTS tries (
        try_id TEXT,
        successful_try_id INT,
        parents_id TEXT,
        selected_best BOOLEAN,
        task TEXT,
        result TEXT
    )''')

    sql_exec('''CREATE TABLE IF NOT EXISTS found_info (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        info TEXT NOT NULL
    )''')

    initial_text, attachments = load_initial_data(chat_id)
    settings = load_chat_settings(chat_id)
    let_log(settings)
    # === Формируем параметры в старом формате ===
    params = {
        'chat_id': chat_id,
        'chat_path': chat_path,
        'initial_text': initial_text,
        'settings': settings,
        'tools': settings.get("another_tools", []),
        'files_list': attachments
    }
    let_log(settings)
    # === Извлекаем параметры ===
    chat_id = params.get("chat_id")
    chat_path = params.get("chat_path")
    initial_text = params.get("initial_text")
    settings = params.get("settings", {})
    tool_paths = params.get("tools", [])

    if not chat_path:
        let_log("Не передан путь к чату!")
        return

    # === Настройки из settings ===
    token_limit = int(settings.get("token_limit", 8192))
    most_often = int(settings.get("frequent_response", 0))
    need_best_result = int(settings.get("best_response", 0))
    max_tasks = int(settings.get("max_tasks", 4))
    global_state.task_retry = int(settings.get("task_repeat", 0))
    
    default_tools_dir = os.path.join(os.path.dirname(__file__), "default_tools")
    for rel_path in tool_paths:
        # Если в списке уже могут быть абсолютные пути, можно проверить:
        if os.path.isabs(rel_path):
            full_path = rel_path
        else:
            full_path = os.path.join(default_tools_dir, rel_path)
        full_path = os.path.normpath(full_path)
        another_tools_files_addresses.append(full_path)

    # === Инициализация модели ===
    model_type = settings.get("model_type", "local")
    language = settings.get("language", "ru")
    
    try:
        # Импорт провайдера в зависимости от типа модели
        match model_type:
            case "local":
                from model_providers.local_provider import ask_model, create_embeddings
                from model_providers.local_provider import connect as model_connect
                from model_providers.local_provider import disconnect as model_disconnect
                let_log(settings.get("model_path"))
                connect_params = settings.get("model_path", "")
            case "ollama":
                from model_providers.ollama_provider import ask_model, create_embeddings
                from model_providers.ollama_provider import connect as model_connect
                from model_providers.ollama_provider import disconnect as model_disconnect
                connect_params = settings.get("ollama_model", "")
            case "openai":
                from model_providers.openai_provider import ask_model, create_embeddings
                from model_providers.openai_provider import connect as model_connect
                from model_providers.openai_provider import disconnect as model_disconnect
                connect_params = settings.get("openai_api_key", "")
            case "lmstudio":
                from model_providers.lmstudio_provider import ask_model, create_embeddings
                from model_providers.lmstudio_provider import connect as model_connect
                from model_providers.lmstudio_provider import disconnect as model_disconnect
                connect_params = settings.get("lmstudio_model", "")
            case "huggingface":
                from model_providers.huggingface_provider import ask_model, create_embeddings
                from model_providers.huggingface_provider import connect as model_connect
                from model_providers.huggingface_provider import disconnect as model_disconnect
                connect_params = settings.get("huggingface_model", "")
            case "cohere":
                from model_providers.cohere_provider import ask_model, create_embeddings
                from model_providers.cohere_provider import connect as model_connect
                from model_providers.cohere_provider import disconnect as model_disconnect
                connect_params = settings.get("cohere_api_key", "")
            case "anthropic":
                from model_providers.anthropic_provider import ask_model, create_embeddings
                from model_providers.anthropic_provider import connect as model_connect
                from model_providers.anthropic_provider import disconnect as model_disconnect
                connect_params = settings.get("anthropic_api_key", "")
            case "custom":
                from model_providers.custom_provider import ask_model, create_embeddings
                from model_providers.custom_provider import connect as model_connect
                from model_providers.custom_provider import disconnect as model_disconnect

                # Поддержка строки подключения (как у других)
                host = settings.get("custom_api_ip", "127.0.0.1")
                port = settings.get("custom_api_port", "65432")
                token = settings.get("custom_api_key", "")
                connect_params = f"host={host};port={port};token={token}"
            case _:
                raise ValueError(f"Unsupported model type: {model_type}")
        
        # Подключение модели
        connection_result = model_connect(connect_params)
        if not connection_result or not connection_result[0]: # TODO: Ошибка подключения модели: Файл модели не найден
            let_log(f"Ошибка подключения модели: {connection_result[1] if len(connection_result) > 1 else 'Unknown error'}")
            return
        
        # Успешное подключение - получаем теги
        success, _, tags = connection_result
        # Устанавливаем реальные теги
        # TODO: тут проверь, заменив теги на ''
        # ВООБЩЕ ТУТ ДОДЕЛАЙ
        global inst_start_tag
        global inst_end_tag
        global sys_start_tag
        global sys_end_tag
        
        inst_start_tag = tags.get("user") or "[INST]"
        inst_end_tag = tags.get("assistant") or "[/INST]"
        sys_start_tag = tags.get("system") or "<sys>"
        sys_end_tag = tags.get("end") or "</sys>"
        
        '''
        inst_start_tag = tags.get("user") or ""
        inst_end_tag = tags.get("assistant") or ""
        sys_start_tag = tags.get("system") or ""
        sys_end_tag = tags.get("end") or ""
        if inst_start_tag == '"user")': inst_start_tag = "user"
        '''
        '''
        inst_start_tag = ""
        inst_end_tag = ""
        sys_start_tag = ""
        sys_end_tag = ""
        '''
        let_log(f"Установлены теги модели: user={inst_start_tag}, assistant={inst_end_tag}, system={sys_start_tag}, end={sys_end_tag}")
        # Делаем функции модели глобальными
        globals().update({
            'ask_provider_model': ask_model,
            'get_provider_embs': create_embeddings,
            }
        )
        provider_module_name = f"model_providers.{model_type}_provider"
        provider_module = sys.modules.get(provider_module_name)
        setattr(provider_module, "token_limit", token_limit)
    except Exception as e:
        let_log(f"Ошибка инициализации модели: {str(e)}")
        traceback.print_exc()
        return

    # === Загрузка модели и инструментов ===
    globalize_language_packet(language)
    
    global_state.ivan_module_tools, global_state.milana_module_tools = system_tools_loader()
    global_state.another_tools = mod_loader(another_tools_files_addresses)
    global_state.tools_str = make_tools_str(global_state.another_tools)
    for tt, _, _ in global_state.another_tools: global_state.an_t_str[tt] = tt # можно не детокенизировать туда-сюда
    
    let_log(global_state.tools_str)
    let_log(global_state.ivan_module_tools)
    
    global_state.module_tools_keys = [tool[0] for tool in global_state.another_tools]
    let_log(another_tools_files_addresses)
    let_log(global_state.another_tools)
    
    # === Загрузка пользовательских данных ===
    fl = params.get("files_list")
    let_log(fl)
    
    if fl:
        send_output_message(text=start_load_attachments_text)
        upload_user_data(fl)
        send_output_message(text=end_load_attachments_text)
    
    # === Запуск обработки ===
    let_log("ЗАПУСК")
    try:
        worker(initial_text)
    except Exception as e:
        # выводим сообщение об ошибке
        print(f"Ошибка: {e}")
        # получаем стек-след (traceback) и извлекаем последнюю запись
        tb = traceback.extract_tb(e.__traceback__)[-1]
        # tb.filename, tb.lineno, tb.name и tb.line содержат информацию об исключении
        print(f"Файл: {tb.filename}, строка: {tb.lineno}")
        tb = traceback.extract_tb(e.__traceback__)[-1]
        t = f"{e} | {tb.filename}:{tb.lineno}"
        message_data = {
            'text': t,
            'attachments': None,
            'command': ''
        }
        try: ui_conn[1].put(message_data)
        except: pass
        log_file = os.path.join(chat_path, 'log.txt')
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f'{t}\n')
    model_disconnect() # Отключаем модель при завершении

#initialize_work(params)

'''
Идеи для реализации:
Милана должна курировать 1 задачу но зная другие для контекста
Создание диалогов, промптов, агентов и инструментов самим ботом
Разбитие задач на сферы и создание отдельной цепочки для каждой (например, разделение медицины, физики и программирования), также с рекурсиями подсфер
Генерация файлов разных форматов в качестве ответов
Получение информации из интернета, либо уточнение у пользователя (для Википедии был модуль, но как по мне лучше дать открытый доступ в инернет)
Менеджер количества цепочек, чтобы не перегрузить компьютер, можно поручить "библиотекарю"
Проверка на ошибки в процессе генерации и после генерации. Возможно создание отдельных цепочек для проверки.
Язык в промпте добавить
Нужно добавить в настройках макс количество задач для миланы
Можно попробовать несколько собеседников в одной цепочке, например, подавать данные с ключами-именами
Нужно распределить сообщения между участниками.
нужно добавить ограничение на контекст для диалога и контекст для сохраненных данных
выбор лучшего варианта нужно доработать добавлением критика, который должен быть даже без 10000 обезьян
'''

# TODO: придумай хороший промпт объясняющий что после команды и аргумента ничего другого быть не может
# TODO: ТОЛЬКО ОДНА КОМАНДА НА ОТВЕТ И УЖЕ СОЗДАННОМУ СПЕЦИАЛИСТУ НЕ КОМАНДОВАТЬ
# TODO: поставь ccache через визуал студио
# TODO: замени рекурсию на цикл в text_cutter()
# TODO: интегрируй аск юзер и симпл сёрч если (они включены) в либрариан (вообще это концепция яслей или хз когда модели можно как можно меньше думать)
# TODO: отчёт о прогрессе
# TODO: поработай с путями функций и прочего чтобы чаты можно было переносить
# TODO: для 10000 обезьян нужна другая генерация но модель будет наверное делать тоже самое если в библиотекаре те же данные, в любом случае, нужно добавить рандомайзер
# TODO: перенеси все импорты в инициалайз
# TODO: перенести все принты в лет лог, лет лог отправляет опкционально в консоль в гуи в файл (в последний сравнивает значение кэщера с последней записью кэша)
# TODO: загрузка инфо лоадеров только когда инфы нет в кэше
# TODO: выпили цвета в консоли
# TODO: допиши в промпт чтобы милана не торопилась заканчивать диалог
# Этот код предполагает, что вы создаете уникальный индекс на поле try_id.
#memory_cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_try_id ON tries (try_id)')

# TODO: что нужно для релиза
# обезопасить обращения к бд +
# сделать чтобы аск юзер нормально работал с интерфейсом +
# добавить конечный ответ +
# вернуть кэширование при генераци текста и ембеддингов +
# обновить гиго +
# добавить английскую локализацию систем текстс +
# выгружай теги правильно +
# приделать апи +
# протестировать всё на олламе +
# протестировать искусственно вызванное завершение диалога +
# добавить кэширование инфо лоадеров +
# [coll_exec] Ошибка (query): Error binding parameter 2: type 'list' is not supported +
# наладитьь везде работу с coll_exec +
# выпилить логгирование в файл +
# добавить отображение генераций по кнопке +
# исправить работу кэшера (где ембеддинги в толк промпте) +
# минимально исправить ошибки для запуска
# исправить ошибку при пересоздании специалиста

# доработать провайдеры (отдельные библиотеки для каждого + автоустановка по надобности)
# разобраться с ембеддингами (они вроде как сохранились хз но там объект numpy) и инференсом в hf
# добавить копирование и вставку
# pаменить ответ на задачу 4 на [ответ на задачу]
# добавить загрузку локализации модов в интерфейс (опционильно) +
# добавить реакцию на сообщения из интерфейса, не забыть проверку на то является ли это первым сообщением (опционально)

# возможные проблемы кэшера
# ретёрн без кэширования в exec
# переполнение токенов в аск модел, вызов каттера, который вызывает модель и получает от туда старый кэш, или нет (может ли он так?)
# то есть каттер режет и уже там что-то сохраняется но аск модель при повторе вызывает уже результат работы текст каттера, переполнения не происходит
# текст каттер не вызывается, происходит сдвиг запросов кэширования
# сначала проверь каттер, затем добавь проверку на особое значение переполнения в кэше
# отпринтуй входные данные каттера, вызови райз, сравни входные данные с тем что потом попадёт в модель при повторе
