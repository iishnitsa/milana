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
        self.last_task_for_executor = {}
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
        self.max_critic_reactions = 3
        self.critic_reactions = {}
        self.critic_wants_retry = False
        self.critic_comment = ''
        self.task_retry = 0
        self.max_attempts = 3
        self.system_tools_keys = []
        self.an_t_str = {}
        self.summ_attach = ''
        self.now_agent_id = 1
        self.gigo_web_search_allowed = True
        self.hierarchy_limit = 0

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
    if real_path.find('\\') != -1: slash = '\\'
    else: slash = '/'
    return file_name[:file_name.rfind(slash)], slash

folder_path, slash = find_work_folder(__file__)
sys.path.append(os.path.join(folder_path, 'system_tools'))
sys.path.append(os.path.join(folder_path, 'system_tools', 'milana'))
sys.path.append(os.path.join(folder_path, 'system_tools', 'ivan'))

is_print_log = True
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
        
        # Записываем в файл только если max_id <= cache_counter
        if max_id <= cache_counter:
            log_file = os.path.join(chat_path, 'log.txt')
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f'{t}\n')

memory_sql = None  # глобальная переменная для SQL-функций

client = None
milana_collection = None
user_collection = None
rag_collection = None
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
              # фильтрация релевантности
              relevance_coeff: float = 0.9,
              **kwargs):
    """
    Универсальная обёртка с кэшированием и поддержкой add/update/delete/query/get/count/modify.
    Поддержка $in и $nin для vector_id через множественные запросы (встроена).
    ВАЖНО: в этой версии обёртка не вычисляет расстояния (нет эвклидов/косинусов).
    """

    cached = cacher(); traceprint()
    if cached is not None:
        '''
        if cached in ("None", "True", "False"):
            return None if cached == "None" else cached == "True"
        return eval(cached)
        '''
        if cached in ("None", "True", "False"):
            return None if cached == "None" else cached == "True"
        return cached

    coll = globals().get(coll_name) or (client and client.get_collection(coll_name))
    if coll is None and action != "delete_collection":
        raise NameError(f"Collection '{coll_name}' not found")

    def _make_where(d):
        if not d: return None
        clauses = []
        for k, v in d.items():
            if isinstance(v, list):
                # Если значение - список, используем $in
                clauses.append({k: {"$in": v}})
            elif isinstance(v, dict) and any(op in v for op in ["$gt", "$gte", "$lt", "$lte", "$ne", "$eq", "$in", "$nin"]):
                # Если значение - словарь с операторами, используем его
                clauses.append({k: v})
            else:
                # В противном случае, это простое сравнение ($eq)
                clauses.append({k: v})
        
        # Если есть несколько условий, объединяем их через $and
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def _extract(resp, include):
        # Логика _extract (без изменений)
        if len(include) > 1:
            out = {}
            for key in include:
                data = resp.get(key, []) or []
                if flatten and isinstance(data, list) and data and isinstance(data[0], list):
                    data = [i for sub in data for i in sub]
                out[key] = data
            return out
        key = include[0]
        data = resp.get(key, []) or []
        if first:
            return data[0] if data else None
        if isinstance(data, list) and data and isinstance(data[0], list):
            return [i for sub in data for i in sub]
        return data

    def _filter_relevance(resp, coeff: float):
        # Логика _filter_relevance (без изменений)
        if "distances" not in resp or resp["distances"] is None or not resp["distances"]:
            return resp
        dists = resp["distances"][0]
        if not dists:
            return resp
        best = min(dists)
        threshold = best * (1.0 + (1.0 - coeff))
        keep_idx = [i for i, d in enumerate(dists) if d <= threshold]
        if not keep_idx:
            return {k: [] for k in resp}
        out = {}
        for k, v in resp.items():
            if isinstance(v, list) and v and isinstance(v[0], list):
                out[k] = [[row[i] for i in keep_idx] for row in v]
            elif isinstance(v, list):
                out[k] = [v[i] for i in keep_idx]
            else:
                out[k] = v
        return out

    # ИСПРАВЛЕННАЯ ФУНКЦИЯ для обхода ограничений ChromaDB
    def _process_in_nin_operators(coll, filters, coll_name, get_results=True):
        """
        Обрабатывает операторы '$in' и '$nin' для 'vector_id',
        обходя ограничения ChromaDB, используя стратегию извлечения всех ID
        и последующей ручной фильтрации.

        Корректно извлекает 'vector_id' из top-level ключей '$in' и '$nin'.
        """
        let_log(f"[{coll_name}] Запуск обхода (in/nin) для ID с фильтрами: {filters}")
        
        # 1. Инициализация. Корректное извлечение $in и $nin для 'vector_id' из top-level операторов
        nin_ids = set(filters.get('$nin', {}).get('vector_id', []))
        in_ids = set(filters.get('$in', {}).get('vector_id', []))
        
        # 2. Создаем базовый фильтр 'where' (ВСЕ, кроме операторов $in и $nin, которые мы обрабатываем вручную)
        base_where_filter = {
            k: v for k, v in filters.items() 
            if k not in ('$in', '$nin', 'vector_id') # Исключаем top-level операторы для ID
        }
        
        all_ids = set()
        offset = 0
        batch_size = 1000 
        
        # 3. Выполняем пагинацию, чтобы получить ВСЕ ID, соответствующие базовому фильтру метаданных.
        # Используем _make_where для преобразования метаданных
        where_for_get = _make_where(base_where_filter)
        
        while True:
            r = coll.get(
                where=where_for_get,
                limit=batch_size,
                offset=offset,
                # ВАЖНОЕ ИСПРАВЛЕНИЕ: include=[] предотвращает ошибку "got ids in get"
                include=[] 
            )
            
            current_ids = r.get('ids', [])
            if not current_ids:
                break
                
            all_ids.update(current_ids)
            offset += batch_size
            
            if len(current_ids) < batch_size:
                break

        let_log(f"[{coll_name}] Найдено {len(all_ids)} ID до фильтрации $in/$nin.")
        
        # 4. Применяем ручную фильтрацию (in/nin)
        final_ids = all_ids
        
        if nin_ids:
            final_ids = final_ids - nin_ids
            
        if in_ids:
            final_ids = final_ids.intersection(in_ids)
            
        final_ids_list = list(final_ids)
        
        let_log(f"[{coll_name}] Осталось {len(final_ids_list)} ID после фильтрации $in/$nin.")

        if not get_results:
            return final_ids_list

        # 5. Выполняем финальный запрос для получения документов по отфильтрованным ID
        if final_ids_list:
            return coll.get(
                ids=final_ids_list,
                # include не содержит 'ids'
                include=['metadatas', 'documents', 'embeddings']
            )
            
        return {'ids': [], 'metadatas': [], 'documents': [], 'embeddings': []}

    try:
        # ИСПРАВЛЕННАЯ ЛОГИКА АКТИВАЦИИ обходного пути для $in/$nin на vector_id
        
        # Проверяем наличие top-level операторов $in или $nin, содержащих 'vector_id'
        id_filters_present = (
            filters and 
            (
                (filters.get('$nin') and isinstance(filters.get('$nin'), dict) and 'vector_id' in filters['$nin']) or
                (filters.get('$in') and isinstance(filters.get('$in'), dict) and 'vector_id' in filters['$in'])
            )
        )

        if action in ("query", "get") and id_filters_present:
            # Используем кастомную функцию для обхода ограничений ChromaDB
            processed = _process_in_nin_operators(
                coll, filters, coll_name, get_results=True
            )

            # Если обходной путь вернул готовые результаты, обрабатываем их и возвращаем
            if isinstance(processed, dict) and 'ids' in processed:
                resp = processed
                
                # Логика определения include (копируем из стандартного блока)
                include = fetch if isinstance(fetch, list) else [fetch]
                if include == ["all"]:
                    include = ["ids", "documents", "metadatas", "embeddings", "distances"]
                
                if action == "query":
                    resp = _filter_relevance(resp, relevance_coeff)
                
                out = _extract(resp, include)
                cacher(out); traceprint()
                return out

        if action == "add":
            out = coll.add(ids=ids, documents=documents, metadatas=metadatas,
                           embeddings=embeddings, **kwargs)
            cacher("True"); traceprint()
            return out

        if action == "update":
            out = coll.update(ids=ids, documents=documents, metadatas=metadatas,
                              embeddings=embeddings, **kwargs)
            cacher("True"); traceprint()
            return out

        if action == "delete":
            out = coll.delete(ids=ids, where=_make_where(filters), **kwargs)
            cacher("True"); traceprint()
            return out

        if action == "count":
            out = coll.count()
            cacher(out); traceprint()
            return out

        if action == "modify":
            out = coll.modify(name=new_name, metadata=new_meta)
            cacher("True"); traceprint()
            return out

        if action == "delete_collection":
            if client is None:
                raise ValueError("client required for delete_collection")
            out = client.delete_collection(coll_name)
            cacher("True"); traceprint()
            return out

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
            else:
                params.update({
                    "where": _make_where(filters),
                    "limit": limit,
                    "offset": offset
                })
                if doc_contains:
                    params["where_document"] = {"$contains": doc_contains}

            params["include"] = include
            params.update(kwargs)
            
            resp = (coll.query if action == "query" else coll.get)(**params)

            # Проверка на пустой результат
            if not resp.get("ids") or not any(resp["ids"]):
                out = _extract(resp, include)
                cacher(out); traceprint()
                return out

            if action == "query":
                resp = _filter_relevance(resp, relevance_coeff)
            
            out = _extract(resp, include)
            cacher(out); traceprint()
            return out

        raise ValueError(f"[coll_exec] Unsupported action: {action}")

    except Exception as e:
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
            return cached
            # Разбираем кэшированный ответ
            '''
            type_part, _, value = cached.partition(":")
            let_log(type_part)
            let_log(value)
            if type_part == 'str': return value
            if value == '': return []
            if value == '[]': return []
            if value == '()': return ()
            if value == '{{}}': return {}
            try: return eval(value) # TODO: замени
            except: return value
            '''
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
            cacher(result); traceprint()
            '''
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
            '''
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
import numpy as np
from sklearn.decomposition import PCA

def get_embs(text: str):# TODO: протестировать
    # 1) кеширование
    send_log_to_ui('embeddings:\n' + text)
    cached = cacher()
    traceprint()
    if cached is not None:
        let_log("[Используются кэшированные эмбеддинги]")
        if cached == 'empty_embs': # TODO: тут сбой кэшера потому что сохранений в цикле много
            cached = []
        let_log(cached)
        return cached

    start = time.time()

    # [ИСПРАВЛЕНИЕ 1 - БЕЗОПАСНОСТЬ] Добавляем ограничение на количество попыток
    max_attempts_emb = 4
    current_attempt = 0

    # 2) начинаем с одного куска — весь текст
    pieces = [text]
    
    while True:
        current_attempt += 1 # Увеличиваем счетчик
        if current_attempt > max_attempts_emb:
            let_log(f'Превышено максимальное количество попыток ({max_attempts_emb}) дробления текста для эмбеддингов. Возврат пустого списка.')
            cacher('empty_embs') 
            return []

        try:
            # 3) ПРЕДВАРИТЕЛЬНАЯ ПРОВЕРКА ПЕРЕПОЛНЕНИЯ
            for i, p in enumerate(pieces):
                estimated_tokens = len(p) * text_tokens_coefficient
                if estimated_tokens > emb_token_limit:
                    # Создаем кастомное исключение для переполнения
                    raise RuntimeError('ContextOverflowError')
                    
            pieces = [piece for piece in pieces if piece.strip()]
            
            if not pieces:
                let_log("Текст состоит из пробелов, возврат пустого эмбеддинга.")
                all_embs = []
                break

            # 4) если проверка пройдена - получаем эмбеддинги
            all_embs_raw = [get_provider_embs(p) for p in pieces]
            
            # [ИСПРАВЛЕНИЕ 2 - РАЗМЕРНОСТЬ] Разворачиваем (flatten) результат на один уровень.
            # Превращаем [[emb1], [emb2]] в [emb1, emb2]
            all_embs = [emb for sublist in all_embs_raw for emb in sublist]
            
            # === НАЧАЛО ЗАКОММЕНТИРОВАННОГО КОДА (PCA/Padding) ===
            # TODO: протестируй на разных моделях
            # emb_len = len(all_embs[0]) if all_embs else 0
            # if emb_len > 1024:
            #     let_log('Уменьшение размерности эмбеддинга с помощью PCA')
            #     # Применение PCA для уменьшения размерности (например, до 1024)
            #     # from sklearn.decomposition import PCA
            #     # pca = PCA(n_components=1024)
            #     # embs_array = np.array(all_embs)
            #     # all_embs = pca.fit_transform(embs_array).tolist()
            # elif emb_len < 1024:
            #     let_log('Дополнение эмбеддинга нулями')
            #     # Дополнение нулями до 1024
            #     # for emb in all_embs:
            #     #     emb.extend([0.0] * (1024 - emb_len))
            # === КОНЕЦ ЗАКОММЕНТИРОВАННОГО КОДА ===

            break # Успешный выход из цикла
            
        except RuntimeError as e:
            # [СТРОГАЯ ОБРАБОТКА ОШИБОК 1] Продолжаем цикл только при ContextOverflowError
            if 'ContextOverflowError' not in str(e): 
                raise # Немедленно поднимаем любую другую RuntimeError, не связанную с переполнением
            
            let_log('интеллектуальное дробление для эмбеддингов')
            
            # 5) ИНТЕЛЛЕКТУАЛЬНОЕ ДЕЛЕНИЕ С ВЫЧИСЛЕНИЕМ КОЛИЧЕСТВА ЧАСТЕЙ
            new_pieces = []
            need_retry = False 
            for p in pieces:
                estimated_tokens = len(p) * text_tokens_coefficient
                
                if estimated_tokens <= emb_token_limit:
                    new_pieces.append(p)
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
                cacher('empty_embs')
                return []
                
            pieces = new_pieces
            let_log(f'Разделено на {len(pieces)} частей')
            
        except Exception as e:
            # [СТРОГАЯ ОБРАБОТКА ОШИБОК 2] Любая другая критическая ошибка
            traceprint()
            let_log(f'Критическая ошибка (не ContextOverflowError) при получении эмбеддингов: {e}')
            raise # Немедленно поднимаем

    elapsed = time.time() - start
    let_log(f'Получение эмбеддингов выполнено за {elapsed:.2f} секунд')

    # 6) УСРЕДНЯЕМ эмбеддинги
    if not all_embs:
        flat_embs = []
    elif len(all_embs) == 1:
        flat_embs = all_embs[0]
    else:
        let_log(f'Усреднение {len(all_embs)} эмбеддингов')
        
        # Работает корректно, так как all_embs теперь 2D: [emb1, emb2, ...]
        embs_array = np.array(all_embs)
        
        flat_embs = np.mean(embs_array, axis=0).tolist()

    # 7) сохраняем в кеш
    if not flat_embs: 
        cacher('empty_embs')
        traceprint()
    else: 
        cacher(flat_embs)
        traceprint()
    
    return flat_embs

import tkinter as tk
from tkinter import simpledialog
use_user = False

def _execute_with_cache_and_error_handling(generation_func):
    """
    Вспомогательная функция для обработки кэширования и ошибок при генерации.
    """
    start = time.time()
    try:
        generated = generation_func()
        cacher(generated) # только при успехе
    except RuntimeError as e:
        if 'ContextOverflowError' in str(e):
            cacher('[ask_model error]')  # кладём только переполнение
            raise RuntimeError("ContextOverflowError")
        else:
            sys.exit(1)  # любая другая RuntimeError — выход
    except Exception:
        sys.exit(1) # любая другая ошибка — выход

    elapsed = time.time() - start
    send_log_to_ui('model:\n' + generated)
    traceprint()
    let_log(generated)
    let_log(f'СГЕНЕРИРОВАНО {len(generated)} токенов за {elapsed:.2f}s')
    let_log(f'Генерация заняла {elapsed:.2f}s, вывод {len(generated)} токенов')
    return generated

def ask_model(prompt_text, 
              system_prompt: str = None,
              all_user: bool = False,
              limit: int = None,
              temperature: float = 0.6,
              **extra_params) -> str:
    
    let_log(prompt_text)
    let_log(f'ВХОД {len(prompt_text)} токенов')
    let_log(f'ВХОД ({len(prompt_text)}):\n{prompt_text}')

    # Проверка кэша
    cached = cacher()
    traceprint()
    if cached is not None:
        let_log("[Используется кэшированный ответ]")
        let_log(cached)
        if cached == '[ask_model error]':
            raise RuntimeError('ContextOverflowError')
        send_log_to_ui('model:\n' + cached)
        return cached

    # Проверка длины контекста
    if len(prompt_text) * text_tokens_coefficient > token_limit - 1000:
        # TODO: можно добавить аварийное сжатие только потом разобраться с ним, оно будет проверять переменную была ли она вызвана каттером
        cacher('[ask_model error]')
        raise RuntimeError("ContextOverflowError")

    # --- Обработка пользовательского ввода ---
    if use_user:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        input_text = simpledialog.askstring("Ввод текста", 
                                        "Пожалуйста, введите текст:",
                                        parent=root)
        root.destroy()
        if input_text is not None:
            if input_text != "":
                cacher(input_text)
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
        for name, val in extra_params.items():
            generation_params[name] = val
        
        return _execute_with_cache_and_error_handling(
            lambda: _process_chat_response(ask_provider_model_chat(generation_params))
        )
    
    # Особый случай 2: all_user
    if all_user:
        let_log("Режим (Особый случай): all_user=True -> chat/completions")
        messages = [{"role": "user", "content": prompt_text}]
        generation_params = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": limit or token_limit
        }
        for name, val in extra_params.items():
            generation_params[name] = val
        
        return _execute_with_cache_and_error_handling(
            lambda: _process_chat_response(ask_provider_model_chat(generation_params))
        )
    
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
        for name, val in extra_params.items():
            generation_params[name] = val
        
        return _execute_with_cache_and_error_handling(
            lambda: ask_provider_model(generation_params) # Вызов completions
        )
    
    # Режим 2: Парсинг чата БЕЗ function call
    elif do_chat_construct and not native_func_call:
        let_log("Режим 2 (do_chat_construct=2): Парсинг (без функций) -> chat/completions")
        
        messages = _parse_roles_to_messages_no_functions(prompt_text) # Используем старый парсер
        generation_params = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": limit or token_limit
        }
        for name, val in extra_params.items():
            generation_params[name] = val
        
        return _execute_with_cache_and_error_handling(
            lambda: _process_chat_response(ask_provider_model_chat(generation_params)) # Вызов chat
        )
    
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
            
        for name, val in extra_params.items():
            generation_params[name] = val
        let_log(generation_params)
        
        # 4. Вызываем модель и получаем полный ответ
        try: 
            openai_response = ask_provider_model_chat(generation_params)
        except RuntimeError as e:
            if 'ContextOverflowError' in str(e):
                cacher('[ask_model error]')  # кладём только переполнение
                raise RuntimeError("ContextOverflowError")
            else: 
                raise
        
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
            # Модель хочет вызвать один или несколько инструментов
            import json # нужен для парсинга JSON-строки с аргументами
            # TODO: тут может быть и не json
            for tool_call in tool_calls:
                function_name = tool_call['function']['name']
                arguments_json_str = tool_call['function']['arguments']
                
                try:
                    args_dict = json.loads(arguments_json_str)
                    # Извлекаем значение по ключу 'arguments'
                    arguments_str = args_dict.get('arguments', '')
                except Exception:
                    # Если парсинг не удался или это не JSON, 
                    # используем "сырую" строку (менее надежно)
                    arguments_str = arguments_json_str
                
                # Собираем маркер, который ожидает tools_selector
                marker = f"\n!!!{function_name}!!!{arguments_str}"
                response_content = marker + response_content
            cacher(response_content)
            return response_content

        elif response_content:
            # Обычный ответ ассистента, без вызова функций
            cacher(response_content)
            return response_content
        
        cacher(response_content)
        # Fallback (если, например, 'role' == 'assistant', но 'content' пустой и 'tool_calls' нет)
        return response_content

def _process_chat_response(api_response):
    """
    Обрабатывает ответ от ask_provider_model_chat (словарь) и извлекает текстовый контент
    Теперь поддерживает как OpenAI-совместимый формат, так и формат Ollama
    """
    from cross_gpt import let_log
    
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

'''
def _process_chat_response(api_response):
    """
    Обрабатывает ответ от ask_provider_model_chat (словарь) и извлекает текстовый контент
    """
    if "choices" not in api_response or not api_response["choices"]:
        let_log("_process_chat_response: Некорректный формат ответа - нет choices")
        raise RuntimeError("Некорректный формат ответа - нет choices")

    choice = api_response["choices"][0]
    if "message" in choice and "content" in choice["message"]:
        result = choice["message"]["content"].strip()
        let_log(f"_process_chat_response: Результат: '{result}'")
        return result
    else:
        let_log(f"_process_chat_response: Некорректный формат ответа - нет message/content в choice: {choice}")
        raise RuntimeError("Некорректный формат ответа - нет message/content в choice")
'''
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
        if system_part.strip():
            messages.append({"role": "system", "content": system_part.strip()})
        
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
            if system_content:
                messages.append({"role": "system", "content": system_content})
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
            if pos == -1:
                break
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
        for role in [operator_role_text, worker_role_text]:
            clean_content = clean_content.replace(role, '').strip()
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
        if i + 1 < len(found_roles):
            content_end = found_roles[i + 1]['pos']
        
        content = remaining_prompt[content_start:content_end].strip()
        
        # Очистка содержимого в зависимости от типа роли
        if role_type == 'function':
            # Для function роли: убираем только начальный \n если есть, но оставляем сам маркер
            clean_content = content
            # Убираем начальный перевод строки если он есть
            if clean_content.startswith('\n'):
                clean_content = clean_content[1:].strip()
            # Добавляем func_role_text в начало содержимого
            clean_content = func_role_text.replace('\n', '') + clean_content
        else:
            # Для operator и worker ролей: полностью удаляем маркеры
            clean_content = content
            for role in [operator_role_text, worker_role_text]:
                clean_content = clean_content.replace(role, '').strip()
        
        if not clean_content:
            continue
            
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
        for role in [operator_role_text, worker_role_text]:
            clean_content = clean_content.replace(role, '').strip()
        messages.append({"role": "user", "content": clean_content})

    # Проверка на четность количества сообщений (включая системное)
    # Если нечетное - удаляем последнее сообщение
    if len(messages) % 2 != 0:
        removed_message = messages.pop()
        let_log(f"Удалено последнее сообщение (нечетное количество): {removed_message['role']} - {removed_message['content'][:100]}...")

    let_log(f"Спарсено сообщений (Режим 2): {len(messages)}")
    for i, msg in enumerate(messages):
        let_log(f"Сообщение {i}: {msg['role']} - {msg['content'][:100]}...")
    
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
    if not base_messages:
        return base_messages

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
            if content.startswith(func_text_for_parse):
                cleaned_content = content[len(func_text_for_parse):].strip()
            elif func_text_for_parse in content:
                # Если не в начале, всё равно вырезаем первое вхождение
                cleaned_content = content.replace(func_text_for_parse, '', 1).strip()
            
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
    try:
        now_commands = global_state.tools_commands_dict.get(sid, {})
    except Exception:
        now_commands = {}

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
                            # удаляем первое вхождение маркера !!!found_key!!!
                            marker_token = "!!!" + found_key + "!!!"
                            if marker_token in prev_txt:
                                new_prev_txt = prev_txt.replace(marker_token, "", 1).strip()
                                if result:
                                    result[-1]["content"] = new_prev_txt
                                else:
                                    base_messages[prev_index]["content"] = new_prev_txt
                            else:
                                # если точного маркера нет, уберём просто первый встреченный маркер
                                first = prev_txt.find("!!!")
                                second = prev_txt.find("!!!", first + 3) if first != -1 else -1
                                if first != -1 and second != -1:
                                    new_prev_txt = (prev_txt[:first] + prev_txt[second+3:]).strip()
                                    if result:
                                        result[-1]["content"] = new_prev_txt
                                    else:
                                        base_messages[prev_index]["content"] = new_prev_txt

                            # создаем function-role сообщение с правильным именем и очищенным контентом
                            # вырезаем func_text_for_parse из начала контента
                            cleaned_content = content
                            if content_l.startswith(func_text_for_parse):
                                cleaned_content = content[len(func_text_for_parse):].strip()
                            
                            function_message = {
                                "role": "function",
                                "name": found_key,
                                "content": cleaned_content
                            }
                            result.append(function_message)
                            i += 1
                            continue
                        else:
                            # найден маркер, но нет команды в сессии -> падаем обратно в assistant
                            msg["role"] = "assistant"
                    except Exception:
                        msg["role"] = "assistant"
                else:
                    # нет маркера в предыдущем сообщении -> рассматриваем как обычный assistant
                    msg["role"] = "assistant"
            else:
                # нет предыдущего -> обычное сообщение
                msg["role"] = "assistant"

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
                
                # удаляем маркер из текста
                cleaned_content = content
                marker_token = "!!!" + found_key + "!!!"
                if marker_token in content:
                    cleaned_content = content.replace(marker_token, "", 1).strip()
                else:
                    # если точного маркера нет, уберём первый встреченный маркер
                    first = content.find("!!!")
                    second = content.find("!!!", first + 3) if first != -1 else -1
                    if first != -1 and second != -1:
                        cleaned_content = (content[:first] + content[second+3:]).strip()
                
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
    if not commands_dict:
        return []
        
    tools_list = []
    for name, details in commands_dict.items():
        if not isinstance(name, str) or not (isinstance(details, (tuple, list)) and len(details) >= 1):
            continue
            
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
        
        if content.startswith(operator_role_text):
            content = content[len(operator_role_text):].strip()
        elif content.startswith(worker_role_text):
            content = content[len(worker_role_text):].strip()
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
            try:
                response = ask_model(text_cutter(prompt), all_user=True)
            except Exception:
                return default_value
        else: raise
    
    first_chars = response[:5]
    
    for char in first_chars:
        if char.isdigit():
            try: return int(char)
            except ValueError: continue
    
    return default_value

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
    
    if not dialog_messages:
        return ""
        
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
    all_results = []  # Собираем все результаты для аннотации
    
    for filename in files_list:
        let_log(filename)
        _, file_extension = os.path.splitext(filename)
        result = cacher(); traceprint()
        
        if result is None:
            global input_info_loaders
            if not input_info_loaders: load_info_loaders(default_handlers_names)
            
            # Определяем обработчик по расширению
            extension = file_extension[1:].lower() if file_extension else ''
            
            # Проверяем, есть ли обработчик для этого расширения
            if extension in input_info_loaders:
                try: 
                    # Пытаемся обработать файл с помощью соответствующего обработчика
                    result = input_info_loaders[extension](filename, input_info_loaders)
                except Exception as e:
                    # В случае ошибки обработки - записываем ошибку и пробуем открыть как текст
                    error_msg = f"{file_open_error_infoloaders}{filename} {e}"
                    let_log(error_msg)
                    result = error_msg
            else:
                # Если расширение не найдено в словаре, пробуем открыть как текст
                try: result = input_info_loaders['txt'](filename, input_info_loaders)
                except Exception as txt_e: result = f"{file_open_error_infoloaders}{filename} {txt_e}"
            cacher(result); traceprint()
        
        # Сохраняем результат для последующей аннотации
        all_results.append({
            'filename': filename,
            'content': result,
            'extension': file_extension[1:]
        })
        
        # Обрабатываем результат в зависимости от типа
        if file_extension[1:].lower() == 'zip' and isinstance(result, list):
            # Обработка ZIP-архива (список файлов)
            for file_data in result:
                if file_data['type'] in ['file', 'unsupported']:
                    content = file_data['content']
                    if isinstance(content, str):
                        let_log('разбиение файла из ZIP')
                        chunks = split_text_with_cutting(content)
                        if not chunks: continue
                        
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
            # Обработка обычного файла (строка)
            if isinstance(result, str):
                let_log('разбиение обычного файла')
                chunks = split_text_with_cutting(result)
                if not chunks: continue
                
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
    
    from info_loaders import cleanup_image_models
    cleanup_image_models()
    
    # АННОТАЦИЯ ВНЕ ЦИКЛА - после обработки всех файлов
    if all_results:
        try:
            # Собираем весь текст для аннотации
            annotation_text = ""
            for result in all_results:
                if isinstance(result['content'], str):
                    annotation_text += f"\n\n--- {result['filename']} ---\n{result['content']}"
                elif isinstance(result['content'], list):
                    for file_data in result['content']:
                        if isinstance(file_data.get('content'), str):
                            annotation_text += f"\n\n--- {result['filename']}/{file_data['filename']} ---\n{file_data['content']}"
            
            global_state.summ_attach = annotation_available_prompt + ask_model(annotation_text, system_prompt=summarize_text_some_phrases)
        except:
            try:
                # Если не удалось сделать аннотацию полного текста, пробуем сокращенный вариант
                global_state.summ_attach = annotation_available_prompt + ask_model(text_cutter(annotation_text), system_prompt=summarize_text_some_phrases)
            except: 
                # Если аннотация совсем не удалась, ставим заглушку
                global_state.summ_attach = annotation_failed_text

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
    traceprint()
    return parse_prompt_response(select_little_best_text + task + solution_text_1 + fr + solution_text_2 + sr, 1)

def is_similar(task, first_result, second_result):
    traceprint()
    return parse_prompt_response(is_similar_text + task + solution_text_1 + fr + solution_text_2 + sr, 0)

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

def _check_module_uses_cross_gpt(file_contents):
    """Проверяет, использует ли модуль функции из cross_gpt, которые требуют кэширования."""
    
    # Список важных функций и модулей
    important_functions = [
        'cacher',
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
    
    # Удаляем комментарии из содержимого файла
    lines = file_contents.split('\n')
    clean_lines = []
    for line in lines:
        # Удаляем комментарии (все, что после #)
        if '#' in line:
            line = line[:line.index('#')]
        clean_lines.append(line)
    
    clean_content = '\n'.join(clean_lines)
    
    # 1. Проверяем импорт всего модуля cross_gpt или chat_manager
    for module in important_modules:
        # Ищем import module
        if f'import {module}' in clean_content:
            return True
        
        # Ищем from module import
        if f'from {module} import' in clean_content:
            return True
    
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
            if ' as ' in imp:
                imp = imp.split(' as ')[0].strip()
            if imp in important_functions:
                return True
    
    # 3. Проверяем однострочные импорты из cross_gpt
    # Ищем паттерн: from cross_gpt import func1, func2, func3
    single_line_pattern = r'from\s+cross_gpt\s+import\s+([^\(\n]+)'
    matches = re.findall(single_line_pattern, clean_content, re.IGNORECASE)
    
    for match in matches:
        # Исключаем импорт с *
        if '*' in match:
            return True
        
        # Разбиваем импортируемые имена по запятым
        imports = [imp.strip().split()[0] for imp in match.split(',') if imp.strip()]
        # Проверяем, есть ли среди них важные функции
        for imp in imports:
            # Убираем возможные as-алиасы
            if ' as ' in imp:
                imp = imp.split(' as ')[0].strip()
            if imp in important_functions:
                return True
    
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
        else:
            # Однострочный импорт
            import_list = import_part
        
        # Разбиваем по запятым
        imports = [imp.strip().split()[0] for imp in import_list.split(',') if imp.strip()]
        
        # Проверяем, есть ли среди них важные функции
        for imp in imports:
            # Убираем возможные as-алиасы
            if ' as ' in imp:
                imp = imp.split(' as ')[0].strip()
            if imp == '*':
                return True
            if imp in important_functions:
                return True
    
    # 5. Проверяем использование cross_gpt.функция
    for func in important_functions:
        if f'cross_gpt.{func}' in clean_content:
            return True
    
    # 6. Проверяем импорт из chat_manager
    # Ищем from chat_manager import что-угодно
    if re.search(r'from\s+chat_manager\s+import', clean_content, re.IGNORECASE):
        return True
    
    # 7. Проверяем использование chat_manager.что-угодно
    if 'chat_manager.' in clean_content:
        return True
    
    return False

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
            else: let_log('НЕ ДОЛЖЕН')
            if _check_module_uses_cross_gpt(file_contents): global_state.system_tools_keys.append(command_name)
            # Добавляем модуль в список
            loaded_modules.append((command_name, description, main_func))

        except Exception as e:
            let_log(f"⚠ Ошибка при обработке {mod_file}: {e}")
            continue

    return loaded_modules

start_dialog_command_name = ''

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
        global start_dialog_address
        target = sys.modules[__name__].__dict__
        for mod, path in zip(modules, files):
            filename = os.path.splitext(os.path.basename(path))[0]
            func = mod[2]
            if filename == 'start_dialog': start_dialog_address = len(target)
            target[filename] = func
            let_log(f"→ Глобализовано: {filename} → {func}")
    start_dialog_address = -1
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
    
    if start_dialog_address != -1:
        global start_dialog_command_name
        start_dialog_command_name = ivan_modules[start_dialog_command_name][0]

    # Глобализуем общие И ivan-модули
    globalize_by_filename(common_modules, common_files)
    globalize_by_filename(ivan_modules, ivan_files)
    globalize_by_filename(milana_modules, milana_files)

    # Формируем словари с ключом — кортеж токенов имени команды
    def to_dict(modules):
        d = {}
        for cmd_t, desc_tokens, func in modules:
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
    
    # --- Парсинг ответа и возврат результата ---
    if marker_decision_revise in decision_response:
        try:
            start_index = decision_response.index(marker_new_task) + len(marker_new_task)
            new_task = decision_response[start_index:].strip()
            
            if new_task:
                print("Критик: Требуется доработка. Сформулирована новая задача.")
                global_state.critic_wants_retry = True
                if global_state.conversations % 2 == 0: global_state.critic_reactions[global_state.conversations - 1] += 1
                else: global_state.critic_reactions[global_state.conversations] += 1
                return new_task
            
        except ValueError:
            return 1 
            
    elif marker_decision_approve in decision_response:
        print("Критик: Задача выполнена успешно.")
        return 1
    
    # <-- Новая, явная проверка на неуверенность
    elif marker_decision_unsure in decision_response:
        print("Критик: Не уверен в результате, требуется проверка человеком.")
        return 1
    
    # Если вердикт не распознан или ответ пустой
    return 1

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
                    split_pos = min(text2.find(slash_n), text2.find(dot_space))
                    if split_pos == -1: # если не нашли ни \n, ни . , ищем просто пробел
                        split_pos = text2.find(' ')
                    if split_pos != -1:
                         text2 = text2[split_pos + 2:] # +2 для \n или ". "

                except Exception as split_e: # Используем более общий Exception для отлова ошибок find
                    traceprint()
                    let_log(f"Ошибка при поиске точки разделения: {split_e}")
                    # В случае ошибки просто продолжаем с половиной текста
                    pass

                text1 = current_chunk[:current_chunk.find(text2)]

                # ВАЖНО: Вместо рекурсивного вызова, добавляем
                # полученные части обратно в начало списка для обработки.
                # text2 добавляем первым, чтобы text1 оказался в самом начале списка
                # и был обработан на следующей итерации, сохраняя порядок.
                if text2: # Добавляем, только если часть не пустая
                    chunks_to_process.insert(0, text2)
                if text1:
                    chunks_to_process.insert(0, text1)

            else:
                # Другая ошибка RuntimeError, останавливаемся
                sys.exit(1)
        except Exception as e:
            # Любая другая непредвиденная ошибка, останавливаемся
            traceprint()
            print(e)
            sys.exit(1)

    # Когда все части обработаны, объединяем их
    return just_space.join(summarized_chunks)

# лучше использовать токинайзер из модели, чтобы не скачивать лишнего
# резулт тру фолс может быть излишеством
# нужен пробел после цифр во избежание проблем
# проблемы с выводом, количество токенов не вызывает ошибку, но ответ просто срезан
# нужно делить сохранение на половину контекста
# также нужно сделать ограничение по контексту ибо слишком большой сбивает модель
# квадратные скобки лучше заменить на круглые чтобы модель их не писала и добавить слово "здесь"
# текст желательно сжиматься не должен, особенно, если он не превышает saved_message_ctx, может добавить сжатие опционально

def save_emb_dialog(text, tag, result=None):
    """
    Парсит диалог, группирует сообщения с помощью LLM (по батчам, если нужно), 
    очищает группы от ролей и нумерации и сохраняет их в RAG-коллекцию.
    Унифицированная версия с интеграцией в global_state и использование функций cross_gpt.py
    """
    let_log(f"\n{'='*60}")
    let_log(f"ВЫЗОВ save_emb_dialog (Длина: {len(text)} символов, Тег: {tag})")
    let_log(f"--- ИСХОДНЫЙ ТЕКСТ ДИАЛОГА (первые 500 символов) ---\n{text[:500]}...\n{'-'*60}")

    # --- 0. Вспомогательные инкапсулированные функции ---
    
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
                if pos == -1:
                    break
                positions.append((pos, role_marker, role_type))
                start_idx = pos + len(role_marker)
        
        if not positions:
            return []
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
            
            if len(content_only) > 300:
                message_preview = content_only[:300] + "..."
            else:
                message_preview = content_only
            
            numbered_text += f"\n{i}. {message_preview}"
        return numbered_text

    def _create_grouping_prompt(numbered_messages_text, total_messages):
        """Создает финальный промпт."""
        # Определяем промпты (можно вынести в глобальные переменные)
        return grouping_prompt_1 + numbered_messages_text + grouping_prompt_2

    def _parse_ranges_from_response(response):
        """Парсит ответ модели в список диапазонов."""
        cleaned = re.sub(r'[^\d,\-]', '', response)
        ranges = []
        for part in cleaned.split(','):
            if not part: 
                continue
            if '-' in part:
                try:
                    start, end = map(int, part.split('-'))
                    if 1 <= start <= end: 
                        ranges.append((start, end))
                except ValueError: 
                    continue
            else:
                try:
                    num = int(part)
                    if 1 <= num: 
                        ranges.append((num, num))
                except ValueError: 
                    continue
        ranges.sort(key=lambda x: x[0])
        return ranges

    def _create_groups_from_ranges(msgs, ranges, offset=0):
        """Создает группы, очищенные от ролей и объединенные."""
        groups = []
        for start, end in ranges:
            if start < 1 or end > len(msgs) or start > end: 
                continue
            
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
            
            if estimated_tokens <= (token_limit - 1000):
                return batch_size
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

    # --- 1. Обработка Result (старая логика) ---
    if result:
        let_log("Сохранение результата отдельно")
        set_common_save_id()
        coll_exec(
            action="add", 
            coll_name="milana_collection", 
            ids=[get_common_save_id()],
            embeddings=[get_embs(result)], 
            metadatas=[{"done": tag, "result": True}], 
            documents=[result]
        )
        return

    # --- 2. Парсинг и группировка ---
    messages = _parse_dialog_to_messages(text)
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

    # --- 3. Сохранение всех групп ---
    let_log(f"\n--- Сохранение {len(all_groups)} групп ---")
    
    # Получаем текущий try и parent_id из global_state
    now_try = get_now_try()
    
    # Универсальная логика для работы с форматами '/' и '1/2/3'
    if now_try == '/' or now_try == '' or now_try is None:
        parent_id = '/'
        try_id = '1'
    else:
        parts = now_try.strip('/').split('/')
        if len(parts) > 1:
            parent_id = '/' + '/'.join(parts[:-1])
            try_id = parts[-1]
        else:
            parent_id = '/'
            try_id = parts[0] if parts[0] else '1'
    
    # Дополнительная проверка на пустые значения
    if not parent_id or parent_id == '':
        parent_id = '/'
    if not try_id or try_id == '':
        try_id = '1'
    
    let_log(f"Метаданные для сохранения: parent_id={parent_id}, try_id={try_id}")

    for i, group in enumerate(all_groups):
        set_common_save_id()
        metadata = {
            "done": tag, 
            "parents_id": parent_id, 
            "try_id": try_id, 
            "result": False
        }
        
        let_log(f"\n### ГРУППА {i+1} (Сохраняемый документ {get_common_save_id()}) ###")
        let_log(f"  Диапазон: {group['global_start']}-{group['global_end']}")
        let_log(f"  СОХРАНЯЕМЫЙ ТЕКСТ (первые 500 символов):\n---START---\n{group['text'][:500]}...\n---END---")

        coll_exec(
            action="add", 
            coll_name="milana_collection", 
            ids=[get_common_save_id()],
            embeddings=[get_embs(group['text'])], 
            metadatas=[metadata], 
            documents=[group['text']]
        )
    
    let_log(f"Всего сохранено {len(all_groups)} групп сообщений")
    let_log(f"{'='*60}")

def get_input_message(command=None, timeout=None, wait=False):
    cached = cacher(); traceprint()
    if cached is not None:
        if cached == "None": return None
        return cached
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
    if answer == None: answer = "None"
    cacher(answer); traceprint()
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

def find_all_commands(text: str, available_commands: list[str], cutoff: float = 0.75) -> list[str]:
    """
    Находит все уникальные команды из списка, которые нечетко соответствуют словам в тексте.

    Эта функция идеально подходит для выбора нескольких инструментов на основе
    описания задачи, сгенерированного моделью.

    Args:
        text (str): Входной текст для поиска (например, ответ модели).
        available_commands (list[str]): Список всех доступных имен команд/инструментов.
        cutoff (float): Порог схожести для difflib (от 0 до 1). Чем выше, тем строже соответствие.

    Returns:
        list[str]: Список уникальных имен команд, которые были найдены в тексте.
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

clean_variables_content = []

def remove_commands_roles(text):  # TODO: почему-то не работает, учти отступы и что-то напоминающее команды
    if not filter_generations: return text
    """
    Ищет второе вхождение конструкции !!!некие_символы!!! где первые !!! 
    находятся на позициях 1-5 символа, удаляет его и всё после него, 
    а также удаляет содержимое переменных role_text и всё после них из текста.
    
    Args:
        text (str): Исходный текст для обработки
        operator_role_text (str): Содержимое переменной operator_role_text
        worker_role_text (str): Содержимое переменной worker_role_text
        func_role_text (str): Содержимое переменной func_role_text
        
    Returns:
        str: Оставшийся текст после удалений
    """
    cleaned_text = text
    
    # Проверяем все переменные по порядку
    
    for var_content in clean_variables_content:
        if var_content:
            # Ищем начало содержимого переменной в тексте
            start_pos = cleaned_text.find(var_content)
            if start_pos != -1:
                # Нашли содержимое - удаляем всё начиная с этой позиции
                cleaned_text = cleaned_text[:start_pos]
                break  # Прерываем после первого найденного
    
    # Теперь ищем второе вхождение конструкции !!!something!!! 
    # где первые !!! находятся на позициях 1-5
    count = 0
    i = 0
    n = len(cleaned_text)
    second_occurrence_pos = -1
    
    while i < n - 3:
        # Проверяем, что первые !!! находятся в позициях 1-5 (индексы 0-4)
        if i < 5 and cleaned_text[i:i+3] == '!!!':
            # Ищем закрывающие !!!
            j = i + 3
            found_end = False
            while j < n - 2 and not found_end:
                if cleaned_text[j:j+3] == '!!!':
                    # Проверяем внутреннее содержимое
                    inner_content = cleaned_text[i+3:j]
                    if (len(inner_content) > 0 and 
                        not any(c.isspace() for c in inner_content)):
                        count += 1
                        if count == 2:
                            second_occurrence_pos = i
                            found_end = True
                            break
                        else:
                            found_end = True
                            i = j + 2
                    else:
                        j += 1
                else:
                    j += 1
            
            if second_occurrence_pos != -1:
                break
        i += 1
    
    # Если найдено второе вхождение, обрезаем текст
    if second_occurrence_pos != -1:
        cleaned_text = cleaned_text[:second_occurrence_pos]
    
    return cleaned_text

def find_and_match_command(text, commands_dict):
    """
    Ищет маркер !!!name!!! в тексте и пытается сопоставить имя с commands_dict.
    Возвращает (found_key, content_str) или None.
    commands_dict ожидается в формате: {'name': ('description', func), ...}
    Поиск: сначала простое in / substring, потом fuzzy через difflib.get_close_matches.
    """
    if not text:
        return None

    # найти первый и второй "!!!" без regex
    first = text.find("!!!")
    if first == -1 or first > 4:
        return None
    second = text.find("!!!", first + 3)
    if second == -1:
        return None

    # извлечь имя команды и контент после второго !!!
    raw_name = text[first + 3:second].strip()
    content = text[second + 3:].strip()

    # защититься, если имя пустое
    if not raw_name:
        return None

    # построить словарь ключей (строки)
    key_map = {}
    try:
        if isinstance(commands_dict, dict):
            for k, v in commands_dict.items():
                key_map[str(k).strip()] = v
        else:
            # если передан список/итерируемый
            for item in commands_dict:
                try:
                    k = str(item[0]).strip()
                    key_map[k] = item[1] if len(item) > 1 else item
                except Exception:
                    continue
    except Exception:
        return None

    # попытка простого поиска
    found_key = None
    for k in key_map:
        if raw_name in k or k in raw_name:
            found_key = k
            break

    # если не нашли — fuzzy match
    if not found_key and key_map:
        try:
            matches = difflib.get_close_matches(raw_name, list(key_map.keys()), n=1, cutoff=0.7)
            if matches:
                found_key = matches[0]
        except Exception:
            found_key = None

    if not found_key:
        return None

    # вернём имя ключа (как он в словаре) и строку аргумента
    return (found_key, content)

def tools_selector(text, sid):
    """
    Вызывает инструменты, используя поиск маркеров и словарь команд в global_state.tools_commands_dict[sid].
    Здесь сохраняется и используется кэш (cacher), как в исходном коде.
    Возвращает результат выполнения команды или None.
    """
    let_log("=== [TOOLS_SELECTOR ЗАПУЩЕН (НОВАЯ ВЕРСИЯ)] ===")
    let_log(f"[TOOLS_SELECTOR] входной текст (начало 200):\n{text[:200]}")

    # 1) получить кэш
    cached = cacher()
    let_log(f"[TOOLS_SELECTOR] кэш: {cached}")
    if cached == [False, False]: 
        return None
    
    # 2) получить словарь команд для сессии
    try:
        now_commands = global_state.tools_commands_dict.get(sid, {})
    except Exception:
        now_commands = {}

    # 3) system keys
    try:
        sys_keys = [str(k) for k in global_state.system_tools_keys]
    except Exception:
        sys_keys = []

    # 4) найти маркер и сопоставить с командами
    match = find_and_match_command(text, now_commands)
    if not match:
        let_log("[TOOLS_SELECTOR] маркер не найден или команда не сопоставилась")
        let_log("=== [TOOLS_SELECTOR ЗАВЕРШЁН] ===")
        cacher([False, False])
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
            if close:
                is_system = True
    except Exception:
        is_system = False
    let_log(f"[TOOLS_SELECTOR] команда системная? {is_system}")

    # 6) Обработка кэша в зависимости от типа команды
    if not is_system:
        # ТОЛЬКО для несистемных команд: проверяем кэш
        if cached is not None and cached != ["SYSTEM", False]:
            let_log("[TOOLS_SELECTOR] Возвращаем не-системный результат из кэша")
            let_log("=== [TOOLS_SELECTOR ЗАВЕРШЁН] ===")
            return cached
    else:
        # Для системных команд: проверяем, не выполняем ли мы её уже (рекурсия)
        if cached is not None and cached != ["SYSTEM", False]:
            raise RuntimeError('СБОЙ КЭШЕРА В ТУЛЗ СЕЛЕКТОРЕ')
        # Помечаем, что начинаем выполнение системной команды
        if cached is None: cacher(["SYSTEM", False])
        traceprint()

    # 7) получить callable из now_commands
    try:
        entry = now_commands.get(found_key)
    except Exception:
        entry = None

    if entry is None:
        let_log("[TOOLS_SELECTOR] команда не найдена в словаре сессии (после сопоставления)")
        let_log("=== [TOOLS_SELECTOR ЗАВЕРШЁН] ===")
        cacher([False, False])
        return None

    func_callable = None
    try:
        if isinstance(entry, tuple) or isinstance(entry, list):
            if len(entry) >= 2 and callable(entry[1]):
                func_callable = entry[1]
            elif len(entry) >= 3 and callable(entry[2]):
                func_callable = entry[2]
            elif callable(entry[0]):
                func_callable = entry[0]
        elif callable(entry):
            func_callable = entry
        else:
            try:
                func_callable = entry.get("func")
            except Exception:
                func_callable = None
    except Exception:
        func_callable = None

    if not func_callable:
        let_log("[TOOLS_SELECTOR] не удалось получить callable для команды")
        let_log("=== [TOOLS_SELECTOR ЗАВЕРШЁН] ===")
        cacher([False, False])
        return None

    # 9) выполнить функцию
    let_log("[TOOLS_SELECTOR] Выполняем функцию...")
    try:
        result = func_callable(content)
    except Exception as e:
        result = "__TOOL_ERROR__: " + str(e)

    let_log(f"[TOOLS_SELECTOR] Результат (первые 500):\n{str(result)[:500]}")

    # 10) кэшировать результат если не системная команда
    try:
        if not is_system:
            let_log("[TOOLS_SELECTOR] Кэшируем результат (не системная команда)")
            cacher(result)
            traceprint()
    except Exception:
        pass

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
        msg_from = operator_role_text

    while not global_state.stop_agent:
        let_log(f"[DEBUG-STD] agent_number={agent_number}, sid={sid}")
        smth = get_chat_context(sid)
        let_log(f'{smth}') # TODO:
        prompt, history = smth

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
        '''
        talk_prompt = remove_commands_roles(talk_prompt)
        '''
        # Обновляем историю. Chat Manager сам разберется, как это сделать.
        # Сначала сообщение от предыдущего, потом ответ от текущего.
        update_history(sid, last_talk_prompt, msg_from) # TODO:
        update_history(sid, talk_prompt, you)

        answer = tools_selector(talk_prompt, sid)
        if answer:
            talk_prompt = answer
            msg_from = func_role_text
            # Ответ от функции тоже сохраняем в историю
            #update_history(sid, talk_prompt, func_role_text) # TODO:
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
        msg_from = operator_role_text

    while not global_state.stop_agent:
        let_log(f"[DEBUG-RAG] agent_number={agent_number}, sid={sid}")

        # 1. Сохраняем входящее сообщение от предыдущего агента в RAG-историю
        update_history(sid, talk_prompt, msg_from)

        # 2. Вызываем RAG-конструктор. Он сам найдет системный промпт и всю историю.
        final_prompt_for_model, _ = get_chat_context(sid, talk_prompt)

        # 3. Вызываем модель, добавив роль текущего агента для корректной генерации
        try:
            talk_prompt = ask_model(final_prompt_for_model + you)
        except Exception as e:
            let_log(f"Ошибка в _rag_agent_func: {e}")
            # Здесь RAG уже должен был обработать длинный контекст
        '''
        talk_prompt = remove_commands_roles(talk_prompt)
        '''
        # 4. Сохраняем ответ самой модели в RAG-историю
        update_history(sid, talk_prompt, you)

        # 5. Логика выбора инструментов
        answer = tools_selector(talk_prompt, sid)
        if answer:
            talk_prompt = answer
            msg_from = func_role_text
            # Ответ от функции тоже сохраняем
            #update_history(sid, talk_prompt, func_role_text) # TODO:
        else:
            break
    global_state.stop_agent = False
    return talk_prompt

def get_token_limit(): return token_limit
def get_text_tokens_coefficient(): return text_tokens_coefficient

agent_func = None

def worker(really_main_task):
    while True:
        global_state.retries = []
        global_state.conversations = 0
        global_state.tools_commands_dict = {}
        global_state.dialog_state = True
        global_state.critic_wants_retry = False # TODO: НЕ ТОЛЬКО ЗДЕСЬ???
        global_state.main_now_task = really_main_task
        global_state.gigo_web_search_allowed = False
        talk_prompt = start_dialog(global_state.main_now_task)
        global_state.gigo_web_search_allowed = True
        need_work = True
        if not global_state.dialog_state:
            need_work = False
            print('воркер диалог завершился не начавшись')
            if global_state.critic_wants_retry:
                if user_review_text1 in really_main_task or user_review_text2 in really_main_task or user_review_text3 in really_main_task or user_review_text4 in really_main_task:
                    really_main_task = really_main_task + user_review_text2 + global_state.dialog_result + user_review_text4 + global_state.critic_comment
                else:
                    really_main_task = user_review_text1 + really_main_task + user_review_text2 + global_state.dialog_result + user_review_text4 + global_state.critic_comment
            else:
                # TODO: это можно вынести в функцию
                while True:
                    if get_input_message() is None: break
                send_output_message(text=global_state.dialog_result, command='end')
                user_message = get_input_message(wait=True)
                if user_review_text1 in really_main_task or user_review_text2 in really_main_task or user_review_text3 in really_main_task or user_review_text4 in really_main_task:
                    really_main_task = really_main_task + user_review_text2 + global_state.dialog_result + user_review_text3 + user_message['text']
                else:
                    really_main_task = user_review_text1 + really_main_task + user_review_text2 + global_state.dialog_result + user_review_text3 + user_message['text']
                if not really_main_task: raise # TODO: доделай тут или общение с пользователем или просто сообщение что текста нет
                if user_message['attachments']:
                    send_output_message(text=start_load_attachments_text)
                    if not input_info_loaders: load_info_loaders(default_handlers_names)
                    upload_user_data(user_message['attachments'])
                    send_output_message(text=end_load_attachments_text)
        while need_work:
            talk_prompt = agent_func(talk_prompt, 0) # ivan
            talk_prompt = agent_func(talk_prompt, 1) # milana
            if not global_state.dialog_state:
                # тут поработай с инициализацией и попытками
                print(global_state.tools_commands_dict)
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
                            global_state.retrying = True # TODO: ЗДЕСЬ РАЗБЕРИСЬ
                            start_dialog('')
                            talk_prompt = make_exec_first
                            continue
                if global_state.conversations <= 0:
                    print('критик 1740')
                    if global_state.critic_wants_retry: # TODO: в критике тоже добавь пояснения И ОН НЕ ВЕЗДЕ ДОЛЖЕН ИСПОЛЬЗОВАТЬ РИЛИ МАЙН ТАКС А ЕЩЁ НАДО УНИФИЦИРОВАТЬ ЕГО С КЛИЕНТСКИМИ ТЕКСТАМИ
                        if user_review_text1 in really_main_task or user_review_text2 in really_main_task or user_review_text3 in really_main_task or user_review_text4 in really_main_task:
                            really_main_task = really_main_task + user_review_text2 + global_state.dialog_result + user_review_text4 + global_state.critic_comment
                        else:
                            really_main_task = user_review_text1 + really_main_task + user_review_text2 + global_state.dialog_result + user_review_text4 + global_state.critic_comment
                
                    else:
                        print('1830 просим пользователя подсказать')
                        while True:
                            if get_input_message() is None: break
                        send_output_message(text=global_state.dialog_result, command='end')
                        user_message = get_input_message(wait=True)
                        if user_review_text1 in really_main_task or user_review_text2 in really_main_task or user_review_text3 in really_main_task or user_review_text4 in really_main_task:
                            really_main_task = really_main_task + user_review_text2 + global_state.dialog_result + user_review_text3 + user_message['text']
                        else:
                            really_main_task = user_review_text1 + really_main_task + user_review_text2 + global_state.dialog_result + user_review_text3 + user_message['text']
                
                        if not really_main_task: raise # TODO: доделай тут или общение с пользователем или просто сообщение что текста нет
                        if user_message['attachments']:
                            send_output_message(text=start_load_attachments_text)
                            if not input_info_loaders: load_info_loaders(default_handlers_names)
                            upload_user_data(user_message['attachments'])
                            send_output_message(text=end_load_attachments_text)
                        let_log('тут')
                    break
                else:
                    print('критик 1768')
                    global_state.dialog_state = True # это куда
                    if global_state.critic_wants_retry:
                        global_state.critic_wants_retry = False
                        if user_review_text1 in global_state.main_now_task or user_review_text2 in global_state.main_now_task or user_review_text3 in global_state.main_now_task or user_review_text4 in global_state.main_now_task:
                            global_state.main_now_task = global_state.main_now_task + user_review_text2 + global_state.dialog_result + user_review_text4 + global_state.critic_comment
                        else:
                            global_state.main_now_task = user_review_text1 + global_state.main_now_task + user_review_text2 + global_state.dialog_result + user_review_text4 + global_state.critic_comment
                        talk_prompt = start_dialog(global_state.main_now_task)
                    else: talk_prompt = global_state.dialog_result + udp_exec_if_needed # тут ошибка, иван не создан
                    continue

# можно в случае ошибки ролбэк сделать тогда замуск инициалайз с трэйсбэком и траем

cache_path = ''

import pickle
import gzip
def cacher(info=None, rollback=False):
    """
    Получает значение из кэша или сохраняет новое значение, используя SQLite BLOB/Gzip.
    Вся логика инкапсулирована внутри.
    База данных инициализируется только при первом обращении или если не существует.
    
    :param info: Данные для записи (если None, то режим чтения).
    :param rollback: Если True, откатывает последнюю запись.
    :return: Десериализованное значение при чтении или True/None при ошибке.
    """
    
    # Константы и локальные переменные
    COMPRESS_LEVEL = 9
    
    global cache_counter
    cache_conn = None 

    # --- ВНУТРЕННИЕ ФУНКЦИИ ---

    def _serialize_data(value):
        """Pickle -> Gzip (уровень 9) -> Байты (BLOB)"""
        try:
            pickled_bytes = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            return gzip.compress(pickled_bytes, compresslevel=COMPRESS_LEVEL)
        except Exception as e:
            let_log(f"Ошибка сериализации данных: {e}")
            return None

    def _deserialize_data(blob):
        """Распаковка Gzip -> Десериализация Pickle."""
        try:
            decompressed_bytes = gzip.decompress(blob)
            return pickle.loads(decompressed_bytes)
        except Exception as e:
            let_log(f"Ошибка десериализации данных: {e}")
            return None
    
    def _initialize_cache_db(cache_conn):
        """Инициализация БД - создание таблицы BLOB."""
        try:
            cache_cursor = cache_conn.cursor()
            
            # Установка лимита BLOB
            cache_cursor.execute("PRAGMA max_page_count = 2147483647;") 
            
            cache_cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    id INTEGER PRIMARY KEY,
                    value BLOB
                )
            ''')
            cache_conn.commit()
            let_log("База данных кэша успешно инициализирована.")
            return True
        except Exception as e:
            let_log(f"Ошибка инициализации базы данных: {e}")
            return False

    def _rollback_cache(cache_conn, num_records):
        """Удаляет последние num_records записей из кэша."""
        try:
            cache_cursor = cache_conn.cursor()
            cache_cursor.execute('DELETE FROM cache WHERE id IN (SELECT id FROM cache ORDER BY id DESC LIMIT ?)', (num_records,))
            cache_conn.commit()
            let_log(f"Выполнен откат последних {num_records} записей.")
        except Exception as e:
            let_log(f"Ошибка отката кэша: {e}")

    # --- ОСНОВНАЯ ЛОГИКА ---
    
    try:
        # Подключаемся к БД (создаст файл, если не существует)
        cache_conn = connect(cache_path)
        cache_cursor = cache_conn.cursor()
        
        if rollback:
            # Откат последней записи 
            _rollback_cache(cache_conn, 1)
            # Если был только rollback, выходим
            if info is None:
                return None
        
        if info is None:
            # --- ЧТЕНИЕ ИЗ КЭША ---
            
            let_log(f"Попытка чтения кэша с id: {cache_counter}")

            # Получаем BLOB по ID
            cache_cursor.execute('SELECT value FROM cache WHERE id = ?', (cache_counter,))
            result = cache_cursor.fetchone()
            
            if result is None:
                let_log(f"Запись кэша с id {cache_counter} не найдена.")
                return None

            # Десериализуем BLOB
            deserialized_value = _deserialize_data(result[0])
            
            cache_counter += 1
            let_log(f"Чтение кэша успешно, следующее id: {cache_counter}")
            return deserialized_value
            
        else:
            # --- ЗАПИСЬ В КЭШ ---
            
            # Сериализуем данные
            compressed_data = _serialize_data(info)
            if compressed_data is None:
                let_log("Не удалось записать: ошибка сериализации.")
                return None

            # Вставляем BLOB. Используем cache_counter как ID.
            cache_cursor.execute('INSERT INTO cache (id, value) VALUES (?, ?)', (cache_counter, compressed_data))
            cache_conn.commit()
            
            let_log(f"Запись кэша с id {cache_counter} успешно завершена.")
            cache_counter += 1
            return True
            
    except OperationalError as e:
        # Обрабатываем ошибку "таблица не существует"
        error_msg = str(e).lower()
        if "no such table" in error_msg or "cache" in error_msg:
            let_log(f"Таблица кэша не найдена, выполняется инициализация: {e}")
            if cache_conn:
                if _initialize_cache_db(cache_conn):
                    # После инициализации повторяем операцию
                    try:
                        if info is None:
                            # Для чтения - таблица пустая, возвращаем None
                            let_log("Таблица кэша только что создана, записей нет.")
                            return None
                        else:
                            # Для записи - повторяем вставку
                            cache_cursor = cache_conn.cursor()
                            compressed_data = _serialize_data(info)
                            if compressed_data:
                                cache_cursor.execute('INSERT INTO cache (id, value) VALUES (?, ?)', (cache_counter, compressed_data))
                                cache_conn.commit()
                                let_log(f"Запись кэша с id {cache_counter} успешно завершена после инициализации БД.")
                                cache_counter += 1
                                return True
                    except Exception as retry_error:
                        let_log(f"Ошибка при повторной попытке после инициализации: {retry_error}")
                else:
                    let_log("Не удалось инициализировать таблицу кэша.")
        else:
            let_log(f"Критическая ошибка SQLite (возможно, блокировка): {e}")
        return None
    
    except Exception as e:
        let_log(f"Неожиданная ошибка в функции cacher: {e}")
        return None
    
    finally:
        if cache_conn:
            cache_conn.close()

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
do_chat_construct = False
native_func_call = False
use_rag = None

def initialize_work(base_dir, chat_id, input_queue, output_queue, stop_event, pause_event, log_queue):
    global memory_sql
    global actual_handlers_names, another_tools_files_addresses
    global token_limit, emb_token_limit, most_often, max_tasks
    global client
    global milana_collection
    global user_collection
    global rag_collection
    global ui_conn
    global cache_path
    global ask_provider_model, get_provider_embs # Для модели
    global language
    global chat_path
    global do_chat_construct
    global native_func_call
    global use_rag
    global agent_func
    global clean_variables_content
    global filter_generations
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
    
    # Пытаемся удалить старые данные (опционально)
    # try: os.remove(db_path)
    # except: pass
    # try: os.remove(cache_path)
    # except: pass
    # try: os.remove(os.path.join(chat_path, "chroma_db"))
    # except: pass

    memory_sql = connect(db_path)

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
    
    # === Формируем параметры в старом формате ===
    params = {
        'chat_id': chat_id,
        'chat_path': chat_path,
        'initial_text': initial_text,
        'settings': settings,
        'tools': settings.get("another_tools", []),
        'files_list': attachments
    }
    
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
    use_rag = int(settings.get("use_rag", 1))
    
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
    
    default_tools_dir = os.path.join(base_dir, "default_tools")  # ИСПРАВЛЕНО: используем base_dir
    for rel_path in tool_paths:
        # Если в списке уже могут быть абсолютные пути, можно проверить:
        if os.path.isabs(rel_path):
            full_path = rel_path
        else:
            full_path = os.path.join(default_tools_dir, rel_path)
        full_path = os.path.normpath(full_path)
        another_tools_files_addresses.append(full_path)

    # === Инициализация модели ===
    model_type = settings.get("model_type", "ollama")
    language = settings.get("language", "ru")
    
    try:
        # Обновляем путь для импорта model_providers
        model_providers_path = os.path.join(base_dir, "model_providers")
        if model_providers_path not in sys.path:
            sys.path.append(model_providers_path)
        
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
    
    if filter_generations == 1:
        cvtl = []
        bos_tag = unified_tags.get('bos')
        user_start_tag = unified_tags.get('user_start')
        
        if bos_tag is not None: 
            cvtl.append(bos_tag)
        if user_start_tag is not None: 
            cvtl.append(user_start_tag)
        
        clean_variables_content = [operator_role_text, worker_role_text, func_role_text]
        clean_variables_content.extend(cvtl)
        filter_generations = True
    else: filter_generations = False
    
    global_state.ivan_module_tools, global_state.milana_module_tools = system_tools_loader()
    
    # === Загружаем пользовательские модули ===
    let_log(f"\n=== ЗАГРУЗКА ПОЛЬЗОВАТЕЛЬСКИХ МОДУЛЕЙ ===")
    let_log(f"Всего файлов для загрузки: {len(another_tools_files_addresses)}")
    
    # 1. Сначала ищем веб-поиск по имени файла
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
                    def web_search_wrapper(arg, func=module.main):
                        return func(arg)
                    web_search_module = (module, web_search_wrapper, file_path)
                    let_log(f"  Модуль веб-поиска загружен: {module_name}")
                else:
                    let_log(f"  ⚠ Файл веб-поиска не содержит функцию main")
            except Exception as e:
                let_log(f"  ⚠ Ошибка загрузки веб-поиска: {e}")
        
        # Ищем ask_user.py
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
                else:
                    let_log(f"  ⚠ Файл ask_user не содержит функцию main")
            except Exception as e:
                let_log(f"  ⚠ Ошибка загрузки ask_user: {e}")
        
        else:
            other_modules_files.append(file_path)
    
    # 2. Загружаем остальные модули через mod_loader
    loaded_tools = []
    if other_modules_files:
        let_log(f"\nЗагрузка остальных модулей ({len(other_modules_files)} файлов)")
        loaded_tools = mod_loader(other_modules_files)
    
    # 3. Добавляем веб-поиск и ask_user в список инструментов (если найдены)
    #    и глобализуем их
    if web_search_module:
        module, web_search_wrapper, file_path = web_search_module
        # Извлекаем command_name и description из файла
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                match = re.match(r'^\s*[\'"]{3}\s*\n\s*([^\n]+)\n\s*([^\n]+)', content)
                if match:
                    command_name = match.group(1).strip()
                    description = match.group(2).strip()
                    # Добавляем в loaded_tools
                    loaded_tools.append((command_name, description, module.main))
                    let_log(f"  Веб-поиск добавлен в список инструментов: {command_name}")
                    
                    # Глобализуем веб-поиск
                    globals()['web_search'] = web_search_wrapper
                    let_log(f"  Функция web_search глобализована")
                else:
                    let_log(f"  ⚠ Не удалось извлечь command_name из файла веб-поиска")
        except Exception as e:
            let_log(f"  ⚠ Ошибка обработки файла веб-поиска: {e}")
    
    if ask_user_module:
        module, ask_user_func, file_path = ask_user_module
        # Извлекаем command_name и description из файла
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                match = re.match(r'^\s*[\'"]{3}\s*\n\s*([^\n]+)\n\s*([^\n]+)', content)
                if match:
                    command_name = match.group(1).strip()
                    description = match.group(2).strip()
                    # Добавляем в loaded_tools
                    loaded_tools.append((command_name, description, ask_user_func))
                    let_log(f"  Ask_user добавлен в список инструментов: {command_name}")
                    
                    # Глобализуем ask_user
                    globals()['ask_user'] = ask_user_func
                    let_log(f"  Функция ask_user глобализована")
                else:
                    let_log(f"  ⚠ Не удалось извлечь command_name из файла ask_user")
        except Exception as e:
            let_log(f"  ⚠ Ошибка обработки файла ask_user: {e}")
    
    # 4. Проверяем, что веб-поиск глобализован
    if 'web_search' in globals() and callable(globals()['web_search']):
        let_log(f"✅ Веб-поиск доступен как cross_gpt.web_search")
    else:
        let_log(f"❌ Веб-поиск НЕ доступен в globals()")
        # Попробуем найти в loaded_tools по имени
        for cmd_name, desc, main_func in loaded_tools:
            if 'web' in cmd_name.lower() or 'search' in cmd_name.lower():
                let_log(f"  Найдена возможная функция веб-поиска: {cmd_name}")
                globals()['web_search'] = main_func
                let_log(f"  Веб-поиск глобализован из команды: {cmd_name}")
                break
    
    # 5. Настраиваем global_state
    global_state.another_tools = loaded_tools
    global_state.tools_str = make_tools_str(global_state.another_tools)
    global_state.an_t_str = {}
    for tt, _, _ in global_state.another_tools: 
        global_state.an_t_str[tt] = tt
    global_state.module_tools_keys = [tool[0] for tool in global_state.another_tools]
    
    let_log(f"\n=== ИТОГИ ЗАГРУЗКИ ===")
    let_log(f"Всего загружено инструментов: {len(loaded_tools)}")
    let_log(f"Веб-поиск доступен: {'web_search' in globals()}")
    let_log(f"Ask_user доступен: {'ask_user' in globals()}")
    let_log(f"Список инструментов: {[t[0] for t in loaded_tools]}")
    
    # === Загрузка пользовательских данных ===
    fl = params.get("files_list")
    
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
    
    # Отключаем модель при завершении
    try:
        model_disconnect()
    except:
        pass
