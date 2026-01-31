import tkinter as tk
from PIL import Image, ImageTk
from pathlib import Path
import multiprocessing
import sys
import customtkinter
import inspect
import importlib.util
import ast
import os

initialize_work = None

# ====== БАЗОВЫЕ ПУТИ ======
def get_base_dir():
    """
    Возвращает абсолютный путь к каталогу, где находится исполняемый файл (ui.exe)
    или файл скрипта (ui.py). Это наш новый BASE_DIR.
    """
    if getattr(sys, 'frozen', False):
        # В режиме PyInstaller возвращаем каталог, где лежит .exe
        return os.path.dirname(os.path.abspath(sys.executable))
    else:
        # В режиме скрипта возвращаем каталог ui.py
        return os.path.dirname(os.path.abspath(__file__))

def resource_path(relative_path):
    """
    Строит абсолютный путь к ресурсу (иконка, БД), используя BASE_DIR.
    """
    return os.path.join(get_base_dir(), relative_path)

BASE_DIR = get_base_dir()

if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)

PROVIDER_DIR = resource_path("providers")
if os.path.exists(PROVIDER_DIR) and PROVIDER_DIR not in sys.path: sys.path.append(PROVIDER_DIR)

# Импорты, необходимые для основной логики приложения,
# будут лениво загружены внутри функции run_main_app
# чтобы сплэш-скрин стартовал мгновенно.

def get_ast_value(node):
    """Безопасное получение значения из AST узла"""
    if isinstance(node, ast.Constant): return node.value
    elif isinstance(node, ast.Str): return node.s
    elif isinstance(node, ast.Num): return node.n
    elif isinstance(node, ast.NameConstant): return node.value
    else: return None

# ====== КЭШ ПРИЛОЖЕНИЯ (НОВЫЙ) ======
class AppCache:
    _instance = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super(AppCache, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.chats = None
            self.global_settings = None
            self.chats_loaded = False
            self.settings_loaded = False
            self.initialized = True
    
    def clear_chats_cache(self):
        self.chats = None
        self.chats_loaded = False
    
    def clear_settings_cache(self):
        self.global_settings = None
        self.settings_loaded = False
    
    def get_chats(self, backend):
        if not self.chats_loaded or self.chats is None:
            self.chats = backend._load_chats_from_db()
            self.chats_loaded = True
        return self.chats
    
    def update_chats(self, chats):
        self.chats = chats
        self.chats_loaded = True
    
    def get_global_settings(self, backend):
        if not self.settings_loaded or self.global_settings is None:
            self.global_settings = backend._load_global_settings_from_db()
            self.settings_loaded = True
        return self.global_settings
    
    def update_global_settings(self, settings):
        self.global_settings = settings
        self.settings_loaded = True

# ====== МЕНЕДЖЕР ЛОКАЛИЗАЦИИ (без изменений) ======
class LanguageManager:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance: cls._instance = super(LanguageManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.texts = {}
            self.available_languages = {}
            self.current_language = None
            self.scan_languages()
            self.initialized = True

    def scan_languages(self):
        lang_dir = Path("lang")
        if not lang_dir.is_dir(): return
        for lang_code_dir in lang_dir.iterdir():
            if lang_code_dir.is_dir():
                ui_file = lang_code_dir / "ui_text.py"
                if ui_file.is_file(): self.available_languages[lang_code_dir.name] = str(ui_file)

    def load_language(self, lang_code="en"):
        codes_to_try = [lang_code, "en"] + list(self.available_languages.keys())
        loaded = False
        for code in codes_to_try:
            if code in self.available_languages:
                try:
                    spec = importlib.util.spec_from_file_location("ui_text", self.available_languages[code])
                    lang_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(lang_module)
                    self.texts = lang_module.TEXTS
                    self.current_language = code
                    loaded = True
                    break
                except Exception as e:
                    print(f"Failed to load language {code}: {e}")
                    continue
        if not loaded: self.texts = {"lang_load_error_title": "Language Error", "lang_load_error_message": "Could not load any language files. Please ensure 'lang/en' directory exists."}
        return loaded

    def get(self, key, **kwargs):
        defaults = {
            "ok": "OK", "cancel": "Cancel", "yes": "Yes", "no": "No",
            "cut": "Cut", "copy": "Copy", "paste": "Paste", "select_all": "Select All",
            "undo": "Undo", "redo": "Redo"
        }
        if key in defaults and key not in self.texts: return defaults[key]
        template = self.texts.get(key, f"[{key.upper()}]")
        return template.format(**kwargs)

# ====== ВАЛИДАТОР МОДУЛЕЙ (без изменений) ======
class ModuleValidator:
    @staticmethod
    def validate_module(module_path):
        import os
        try:
            if not os.path.isfile(module_path): return False, Lang.get("module_err_not_found", path=module_path)
            with open(module_path, 'r', encoding='utf-8') as f: source = f.read()
            tree = ast.parse(source)
            docstring = ast.get_docstring(tree)
            if not docstring: return False, Lang.get("module_err_no_docstring")
            doc_lines = docstring.strip().split('\n')
            if len(doc_lines) < 4: return False, Lang.get("module_err_docstring_len")
            main_found = any(
                isinstance(node, ast.FunctionDef) and node.name == 'main' and len(node.args.args) == 1
                for node in tree.body
            )
            if not main_found: return False, Lang.get("module_err_main_not_found")
            return True, Lang.get("module_validated")
        except SyntaxError as e:
            return False, Lang.get("module_err_syntax", e=e)
        except Exception as e:
            return False, Lang.get("module_err_generic", e=e)

# ====== МЕНЕДЖЕР ПРОВАЙДЕРОВ (С ИЗМЕНЕНИЯМИ: сканирование только при инициализации) ======
class ProviderManager:
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance: cls._instance = super(ProviderManager, cls).__new__(cls)
        return cls._instance
        
    def __init__(self):
        if not self._initialized:
            self.providers = {}
            self.scan_providers()
            self._initialized = True

    def scan_providers(self):
        self.providers.clear()
        provider_dir = Path("model_providers")
        if not provider_dir.is_dir():
            print("Warning: 'model_providers' directory not found.")
            return
        for py_file in provider_dir.glob("*.py"):
            if py_file.name.startswith("_") or not py_file.is_file(): continue
            module_name = py_file.stem
            display_name = module_name.replace("_", " ").title()
            is_valid, funcs, has_token_limit, has_params_in_connect = self._validate_provider(py_file)
            if is_valid and has_token_limit and has_params_in_connect:
                params = self._parse_params_from_connect_as_params_list(py_file)
                self.providers[module_name] = {
                    "path": py_file,
                    "name": display_name,
                    "params": params,
                }
            else:
                print(f"Warning: Skipping invalid provider file {py_file.name}")
                if not has_token_limit: print(f"  - Missing required variable: token_limit")
                if not has_params_in_connect: print(f"  - Missing params variable in connect function")
        print(f"Discovered providers: {list(self.providers.keys())}")

    def _validate_provider(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f: source = f.read()
            tree = ast.parse(source)
            required_funcs = {"connect", "disconnect", "ask_model", "ask_model_chat", "create_embeddings"}
            found_funcs = {
                node.name for node in tree.body
                if isinstance(node, ast.FunctionDef)
            }
            has_token_limit = False
            has_params_in_connect = False
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if target.id == 'token_limit':
                                has_token_limit = True
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == "connect":
                    for stmt in node.body:
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if isinstance(target, ast.Name) and target.id == 'params':
                                    has_params_in_connect = True
                                    break
            return required_funcs.issubset(found_funcs), found_funcs, has_token_limit, has_params_in_connect
        except Exception as e:
            print(f"Error validating provider {path}: {e}")
            return False, set(), False, False

    def _parse_params_from_connect_as_params_list(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f: source = f.read()
            tree = ast.parse(source)
            params_info = []
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == "connect":
                    for stmt in node.body:
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if isinstance(target, ast.Name) and target.id == 'params':
                                    if isinstance(stmt.value, ast.Dict):
                                        for key, value in zip(stmt.value.keys, stmt.value.values):
                                            key_name = get_ast_value(key)
                                            if key_name is None:
                                                continue
                                            
                                            default_value = get_ast_value(value)
                                            
                                            param_data = {
                                                'name': key_name,
                                                'default': default_value,
                                                'is_file': 'path' in key_name.lower() or 
                                                        'file' in key_name.lower() or 
                                                        'dir' in key_name.lower()
                                            }
                                            params_info.append(param_data)
            return params_info
        except Exception as e:
            print(f"Could not parse params from connect function for {path.name}: {e}")
            return []

    def get_providers(self): return self.providers

# ====== МЕНЕДЖЕР МОДУЛЕЙ (С ИСПРАВЛЕНИЕМ: предотвращение двойной загрузки) ======
class ModuleManager:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance: cls._instance = super(ModuleManager, cls).__new__(cls)
        return cls._instance
        
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.default_modules = []
            self.custom_modules = []
            self.initialized = True
            self.loaded = False  # Флаг, что модули уже загружены
    
    def load_modules(self, backend):
        """Загружает модули один раз при запуске и сохраняет в кэш"""
        if self.loaded:
            print("Modules already loaded, skipping")
            return
            
        try:
            # Загружаем модули по умолчанию
            self.default_modules = backend.get_default_mods()
            # Загружаем кастомные модули
            self.custom_modules = backend.get_custom_mods()
            self.loaded = True
            print(f"Loaded {len(self.default_modules)} default modules and {len(self.custom_modules)} custom modules to cache")
        except Exception as e:
            print(f"Error loading modules: {e}")
            self.default_modules = []
            self.custom_modules = []
    
    def get_default_modules(self): return self.default_modules
    
    def get_custom_modules(self): return self.custom_modules
    
    def update_custom_modules(self, backend):
        """Обновляет только кастомные модули (при добавлении/удалении)"""
        self.custom_modules = backend.get_custom_mods()

# ====== БЭКЕНД (С ИСПРАВЛЕНИЯМИ ДЛЯ РАБОТЫ С КЭШЕМ) ======
class Backend:
    def __init__(self):
        # !!! ИСПРАВЛЕНИЕ: Используем resource_path для пути к БД !!!
        self.db_path = resource_path(os.path.join("data", "settings.db"))
        self.cache = AppCache()  # Инициализируем кэш
        self.init_settings_db()
        
    def sql_exec(self, db_path, query, params=(), fetchone=False, fetchall=False, commit=True):
        import sqlite3
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(query, params)
            if commit: conn.commit()
            result = None
            if fetchone: result = cursor.fetchone()
            elif fetchall: result = cursor.fetchall()
            return result
        except Exception as e:
            print(f"[SQL Error] {query} | {params} -> {e}")
            return None
        finally:
            if 'conn' in locals(): conn.close()

    def _get_localized_doc(self, mod_path: Path):
        localized_name, localized_desc = None, None
        lang_file = mod_path.with_name(f"{mod_path.stem}_lang.py")
        if lang_file.exists() and Lang.current_language and Lang.current_language != "en":
            try:
                spec = importlib.util.spec_from_file_location("lang_module", str(lang_file))
                lang_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(lang_module)
                if hasattr(lang_module, 'locales') and Lang.current_language in lang_module.locales:
                    locale_data = lang_module.locales[Lang.current_language]
                    if 'module_doc' in locale_data and len(locale_data['module_doc']) >= 4:
                        localized_name = locale_data['module_doc'][2]
                        localized_desc = locale_data['module_doc'][3]
            except Exception as e: print(f"Ошибка загрузки локализации для {mod_path.name}: {e}")
        with open(mod_path, 'r', encoding='utf-8') as f:  source = f.read()
        tree = ast.parse(source)
        docstring = ast.get_docstring(tree) or ""
        doc_lines = docstring.strip().split('\n')
        name = localized_name or (doc_lines[2].strip() if len(doc_lines) > 2 else mod_path.stem)
        description = localized_desc or (doc_lines[3].strip() if len(doc_lines) > 3 else Lang.get("module_desc_missing"))
        return name, description
    
    def rescan_and_localize_modules(self):
        db_path = self.db_path
        import platform
        system_os = platform.system().lower()
        current_language_row = self.sql_exec(db_path, "SELECT value FROM settings WHERE key = 'language'", fetchone=True)
        current_language = current_language_row[0] if current_language_row else 'en'
        self._check_existing_modules(db_path, current_language)
        self._scan_default_tools(db_path, system_os, current_language)
        # Обновляем кэш модулей после рескана
        ModuleManager().load_modules(self)
        return True

    def _check_existing_modules(self, db_path, current_language):
        current_defaults = self.sql_exec(db_path, "SELECT id, adress, lang FROM default_mods", fetchall=True) or []
        for mod_id, mod_adress, mod_lang in current_defaults:
            mod_path = Path(resource_path(os.path.join("default_tools", mod_adress)))
            if not mod_path.exists():
                self.sql_exec(db_path, "DELETE FROM default_mods WHERE id = ?", (mod_id,))
                continue
            if mod_lang != current_language:
                new_name, new_desc = self._get_localized_doc(mod_path)
                self.sql_exec(db_path, "UPDATE default_mods SET name = ?, description = ?, lang = ? WHERE id = ?", 
                            (new_name, new_desc, current_language, mod_id))
        current_customs = self.sql_exec(db_path, "SELECT id, adress, lang FROM custom_mods", fetchall=True) or []
        for mod_id, mod_adress, mod_lang in current_customs:
            mod_path = Path(mod_adress)
            if not mod_path.exists():
                self.sql_exec(db_path, "DELETE FROM custom_mods WHERE id = ?", (mod_id,))
                continue
            if mod_lang != current_language:
                new_name, new_desc = self._get_localized_doc(mod_path)
                self.sql_exec(db_path, "UPDATE custom_mods SET name = ?, description = ?, lang = ? WHERE id = ?", 
                            (new_name, new_desc, current_language, mod_id))

    def _scan_default_tools(self, db_path, system_os, current_language):
        default_mods_dir = Path(resource_path("default_tools"))
        if not default_mods_dir.exists(): return

        def process_mod_file(mod_file, relative_path_str):
            if mod_file.name.endswith("_lang.py"): return
            mod_name_stem = mod_file.stem.lower()
            if mod_name_stem in ['windows_cmd', 'linux_cmd', 'macos_cmd']:
                if not ((system_os == 'windows' and mod_name_stem == 'windows_cmd') or
                        (system_os == 'linux' and mod_name_stem == 'linux_cmd') or
                        (system_os == 'darwin' and mod_name_stem == 'macos_cmd')):
                    return
            existing = self.sql_exec(db_path, "SELECT id FROM default_mods WHERE adress = ?", (relative_path_str,), fetchone=True)
            if existing: return
            valid, msg = ModuleValidator.validate_module(str(mod_file.resolve()))
            if not valid:
                print(f"Ошибка в модуле по умолчанию {mod_file.name}: {msg}")
                return
            name, description = self._get_localized_doc(mod_file)
            self.sql_exec(db_path, "INSERT OR IGNORE INTO default_mods (name, description, adress, enabled, lang) VALUES (?, ?, ?, ?, ?)",
                        (name, description, relative_path_str, 0, current_language))
        for item in default_mods_dir.rglob("*.py"):
            if item.is_file():
                relative_path = item.relative_to(default_mods_dir)
                process_mod_file(item, str(relative_path))

    def init_settings_db(self):
        Path("data").mkdir(exist_ok=True)
        db_path = self.db_path
        self.sql_exec(db_path, "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
        self.sql_exec(db_path, """CREATE TABLE IF NOT EXISTS default_mods 
            (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, description TEXT, 
            adress TEXT UNIQUE, enabled INTEGER DEFAULT 0, lang TEXT DEFAULT 'en')""")
        self.sql_exec(db_path, """CREATE TABLE IF NOT EXISTS custom_mods 
            (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, description TEXT, 
            adress TEXT UNIQUE, enabled INTEGER DEFAULT 1, lang TEXT DEFAULT 'en')""")
        self.sql_exec(db_path, "INSERT OR IGNORE INTO settings (key, value) VALUES ('language', 'en')")
        providers = ProviderManager().get_providers()
        default_provider = list(providers.keys())[0] if providers else ""
        defaults = {
            "token_limit": "8192", "max_tasks": "4", "model_provider_params": "",
            "model_type": default_provider, "use_rag": "1", 
            "filter_generations": "0", "hierarchy_limit": "0",
            "write_log": "1"  # Добавлена настройка write_log по умолчанию включена
        }
        for key, value in defaults.items(): self.sql_exec(db_path, "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (key, value))
    
    def generate_id(self, length=12):
        import random, string
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    
    def _load_chats_from_db(self):
        """Внутренний метод для загрузки чатов из БД (используется кэшем)"""
        import os
        result = []
        chats_dir = resource_path(os.path.join("data", "chats"))
        if not os.path.exists(chats_dir): return []
        chats_dir = Path(chats_dir)
        for folder in sorted(chats_dir.iterdir(), key=os.path.getmtime, reverse=True):
            if folder.is_dir():
                settings_db = folder / "chatsettings.db"
                name = folder.name
                if settings_db.exists():
                    name_in_db_row = self.sql_exec(str(settings_db), "SELECT value FROM settings WHERE key = 'chat_name'", fetchone=True)
                    if name_in_db_row: name = name_in_db_row[0]
                result.append({"id": folder.name, "name": name})
        return result
    
    def get_chats(self):
        """Получает чаты из кэша"""
        return self.cache.get_chats(self)
    
    def _load_global_settings_from_db(self):
        """Внутренний метод для загрузки настроек из БД (используется кэшем)"""
        rows = self.sql_exec(self.db_path, "SELECT key, value FROM settings", fetchall=True) or []
        return {k: v for k, v in rows}
    
    def get_global_settings(self):
        """Получает глобальные настройки из кэша"""
        return self.cache.get_global_settings(self)
    
    def update_global_settings(self, settings):
        """Обновляет глобальные настройки в БД и кэше"""
        for key, value in settings.items(): self.sql_exec(self.db_path, "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
        # Обновляем кэш
        self.cache.update_global_settings({**self.cache.get_global_settings(self), **settings})
        return True
    
    def create_chat(self, chat_name, settings_data):
        # Проверяем имя чата по кэшу
        existing_chats = self.cache.get_chats(self)
        if any(chat['name'].lower() == chat_name.lower() for chat in existing_chats):  return None
        chat_id = self.generate_id()
        chat_path = Path(resource_path(os.path.join("data", "chats"))) / chat_id
        chat_path.mkdir(parents=True, exist_ok=True)
        # Добавляем чат в кэш сразу (изменение 1)
        new_chat = {"id": chat_id, "name": chat_name}
        updated_chats = [new_chat] + existing_chats
        self.cache.update_chats(updated_chats)
        # Создаем папки в зависимости от включенных модулей (изменение 3)
        default_mods = ModuleManager().get_default_modules()
        # Проверяем, включен ли модуль create_file.py
        create_file_mod = next((mod for mod in default_mods if mod['adress'] == 'create_file.py'), None)
        if create_file_mod:
            mod_enabled = settings_data.get('default_mods_config', {}).get(create_file_mod['id'], False)
            if mod_enabled: (chat_path / "files").mkdir(exist_ok=True)
        # Проверяем, есть ли модуль, оканчивающийся на cmd.py
        cmd_mod = next((mod for mod in default_mods if mod['adress'].endswith('cmd.py')), None)
        if cmd_mod:
            mod_enabled = settings_data.get('default_mods_config', {}).get(cmd_mod['id'], False)
            if mod_enabled: (chat_path / "console_folders").mkdir(exist_ok=True)
        # Проверяем, включен ли модуль create_report.py
        create_report_mod = next((mod for mod in default_mods if mod['adress'] == 'create_report.py'), None)
        if create_report_mod:
            mod_enabled = settings_data.get('default_mods_config', {}).get(create_report_mod['id'], False)
            if mod_enabled: (chat_path / "reports").mkdir(exist_ok=True)
        settings_db = str(chat_path / "chatsettings.db")
        self.sql_exec(settings_db, "PRAGMA max_page_count = 2147483647")
        self.sql_exec(settings_db, "CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT)")
        self.sql_exec(settings_db, "CREATE TABLE default_mods (id INTEGER PRIMARY KEY, name TEXT, adress TEXT, enabled INTEGER)")
        self.sql_exec(settings_db, "CREATE TABLE custom_mods (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, description TEXT, adress TEXT UNIQUE)")
        all_settings = {**settings_data.get('model_config', {}), **settings_data.get('chat_config', {})}
        all_settings['chat_name'] = chat_name
        for key, value in all_settings.items(): self.sql_exec(settings_db, "INSERT INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
        # Используем кэшированные модули
        default_mods = ModuleManager().get_default_modules()
        enabled_defaults = settings_data.get('default_mods_config', {})
        for mod in default_mods:
            self.sql_exec(settings_db, "INSERT INTO default_mods (id, name, adress, enabled) VALUES (?, ?, ?, ?)", (mod['id'], mod['name'], mod['adress'], 1 if enabled_defaults.get(mod['id'], False) else 0))
        for mod in settings_data.get('custom_mods_list', []):
            self.sql_exec(settings_db, "INSERT INTO custom_mods (name, description, adress) VALUES (?, ?, ?)", (mod['name'], mod['description'], mod['adress']))
        dialog_db = str(chat_path / "chatsettings.db")
        self.sql_exec(dialog_db, "CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, is_my INTEGER, attachments TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
        return new_chat
    
    def delete_chat(self, chat_id):
        import shutil
        chat_path = Path(resource_path(os.path.join("data", "chats"))) / chat_id
        if not chat_path.exists(): return False
        shutil.rmtree(chat_path)
        # Обновляем кэш
        existing_chats = self.cache.get_chats(self)
        updated_chats = [chat for chat in existing_chats if chat['id'] != chat_id]
        self.cache.update_chats(updated_chats)
        return True
    
    def get_messages(self, chat_id):
        import json
        db = Path(resource_path(os.path.join("data", "chats", chat_id, "chatsettings.db")))
        if not db.exists(): return []
        rows = self.sql_exec(str(db), "SELECT text, is_my, attachments FROM messages ORDER BY timestamp", fetchall=True) or []
        return [{"text": r[0], "isMy": bool(r[1]), "attachments": json.loads(r[2]) if r[2] else []} for r in rows]
    
    def add_message(self, chat_id, text, is_my, attachments=None):
        import json
        db = Path(resource_path(os.path.join("data", "chats", chat_id, "chatsettings.db")))
        if not db.exists(): return False
        attachments_str = json.dumps([str(a) for a in attachments]) if attachments else None
        self.sql_exec(str(db), "INSERT INTO messages (text, is_my, attachments) VALUES (?, ?, ?)", (text, int(is_my), attachments_str))
        return True
    
    def get_chat_settings(self, chat_id):
        db = Path(resource_path(os.path.join("data", "chats", chat_id, "chatsettings.db")))
        if not db.exists(): return {}
        rows = self.sql_exec(str(db), "SELECT key, value FROM settings", fetchall=True) or []
        return {k: v for k, v in rows}
    
    def get_default_mods(self):
        rows = self.sql_exec(self.db_path, "SELECT id, name, description, adress, enabled FROM default_mods", fetchall=True) or []
        return [{"id": r[0], "name": r[1], "description": r[2], "adress": r[3], "enabled": bool(r[4])} for r in rows]
    
    def get_custom_mods(self):
        rows = self.sql_exec(self.db_path, "SELECT id, name, description, adress, enabled FROM custom_mods", fetchall=True) or []
        return [{"id": r[0], "name": r[1], "description": r[2], "adress": r[3], "enabled": bool(r[4])} for r in rows]
    
    def update_default_mod_enabled(self, mod_id, enabled):
        self.sql_exec(self.db_path, "UPDATE default_mods SET enabled = ? WHERE id = ?", (1 if enabled else 0, mod_id))
        return True
    
    def remove_custom_mod(self, mod_id):
        self.sql_exec(self.db_path, "DELETE FROM custom_mods WHERE id = ?", (mod_id,))
        return True
    
    def add_custom_mod(self, file_path):
        valid, error_msg = ModuleValidator.validate_module(file_path)
        if not valid: raise ValueError(Lang.get("module_validation_error", error_msg=error_msg))
        name, description = self._get_localized_doc(Path(file_path))
        self.sql_exec(self.db_path, "INSERT INTO custom_mods (name, description, adress, enabled) VALUES (?, ?, ?, ?)", (name, description, file_path, 1))
        return True
    
    def is_main_config_complete(self):
        settings = self.get_global_settings()
        if not settings: return False
        return bool(settings.get("model_type")) and bool(settings.get("model_provider_params"))

    def validate_model_settings(self, model_type, connection_string):
        max_tokens = 8192
        try:
            if not model_type: return False, Lang.get("model_err_no_provider"), max_tokens
            provider_manager = ProviderManager()
            provider_data = provider_manager.providers.get(model_type)
            if not provider_data: return False, Lang.get("model_err_provider_missing", provider=model_type), max_tokens
            provider_module = importlib.import_module(f"model_providers.{model_type}")
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f): valid, tokens, _ = provider_module.connect(connection_string)
            if hasattr(provider_module, 'disconnect'): provider_module.disconnect()
            if valid: return True, Lang.get("model_validated_success", tokens=tokens), tokens
            else: return False, Lang.get("custom_api_fail"), max_tokens
        except ImportError as e:
            print(e)
            return False, Lang.get("model_err_provider_missing", provider=model_type), max_tokens
        except Exception as e: return False, Lang.get("model_err_validation_generic", e=str(e)), max_tokens

# ====== ТЕМНАЯ ПАНЕЛЬ ЗАГОЛОВКА (без изменений) ======
def set_windows_dark_titlebar(window):
    if sys.platform != "win32": return
    try:
        import ctypes
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        value = ctypes.c_int(2)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(value), ctypes.sizeof(value))
        DWMWA_CAPTION_COLOR = 35
        color = ctypes.c_int(0x00000000)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_CAPTION_COLOR, ctypes.byref(color), ctypes.sizeof(color))
        ctypes.windll.user32.SetWindowPos(hwnd, None, 0, 0, 0, 0, 0x0027)
    except Exception as e: print(f"Failed to set dark title bar: {e}")

# ====== МЕНЕДЖЕР КОНТЕКСТНОГО МЕНЮ (без изменений) ======
def enhance_text_widget(widget):
    is_textbox = isinstance(widget, customtkinter.CTkTextbox)
    is_entry = isinstance(widget, customtkinter.CTkEntry)

    def select_all(event=None):
        if is_textbox: widget.tag_add("sel", "1.0", "end")
        elif is_entry: widget.select_range(0, 'end')
        return "break"

    def copy_action(event=None):
        try:
            if is_textbox and widget.tag_ranges("sel"):
                selected_text = widget.get(tk.SEL_FIRST, tk.SEL_LAST)
                widget.clipboard_clear()
                widget.clipboard_append(selected_text)
            elif is_entry and widget.select_present():
                selected_text = widget.selection_get()
                widget.clipboard_clear()
                widget.clipboard_append(selected_text)
        except (tk.TclError, AttributeError): pass
        return "break"

    def cut_action(event=None):
        is_disabled = hasattr(widget, '_state') and widget._state == 'disabled'
        if is_disabled: return "break"
        try:
            copy_action(event)
            if is_textbox and widget.tag_ranges("sel"): widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
            elif is_entry and widget.select_present(): widget.delete(widget.index(tk.SEL_FIRST), widget.index(tk.SEL_LAST))
        except (tk.TclError, AttributeError): pass
        return "break"

    def paste_action(event=None):
        is_disabled = hasattr(widget, '_state') and widget._state == 'disabled'
        if is_disabled: return "break"
        try:
            clipboard_content = widget.clipboard_get()
            if is_textbox:
                if widget.tag_ranges("sel"): widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
                widget.insert(tk.INSERT, clipboard_content)
            elif is_entry:
                if widget.select_present(): widget.delete(widget.index(tk.SEL_FIRST), widget.index(tk.SEL_LAST))
                widget.insert(tk.INSERT, clipboard_content)
        except tk.TclError: pass
        return "break"
    widget.bind("<Control-c>", copy_action)
    widget.bind("<Control-x>", cut_action)
    widget.bind("<Control-v>", paste_action)
    widget.bind("<Control-a>", select_all)
    if sys.platform == "darwin":
        widget.bind("<Command-c>", copy_action)
        widget.bind("<Command-x>", cut_action)
        widget.bind("<Command-v>", paste_action)
        widget.bind("<Command-a>", select_all)
    menu = tk.Menu(widget, tearoff=0, bg=DARK_SECONDARY, fg=WHITE, relief="flat", borderwidth=0)

    def show_menu(event):
        has_selection = False
        try:
            if is_textbox and widget.tag_ranges("sel"): has_selection = True
            elif is_entry and widget.select_present(): has_selection = True
        except tk.TclError: pass
        has_clipboard = False
        try:
            if widget.clipboard_get(): has_clipboard = True
        except tk.TclError: pass
        is_disabled = hasattr(widget, '_state') and widget._state == 'disabled'
        menu.entryconfigure(Lang.get("cut"), state="normal" if has_selection and not is_disabled else "disabled")
        menu.entryconfigure(Lang.get("copy"), state="normal" if has_selection else "disabled")
        menu.entryconfigure(Lang.get("paste"), state="normal" if has_clipboard and not is_disabled else "disabled")
        menu.entryconfigure(Lang.get("select_all"), state="normal")
        menu.tk_popup(event.x_root, event.y_root)

    menu.add_command(label=Lang.get("cut"), command=lambda: cut_action(None))
    menu.add_command(label=Lang.get("copy"), command=lambda: copy_action(None))
    menu.add_command(label=Lang.get("paste"), command=lambda: paste_action(None))
    menu.add_separator()
    menu.add_command(label=Lang.get("select_all"), command=lambda: select_all(None))
    widget.bind("<Button-3>", show_menu)
    if sys.platform == "darwin": widget.bind("<Button-2>", show_menu)

# ====== ОСНОВНОЙ КЛАСС UI (С ИСПРАВЛЕНИЯМИ) ======
class ChatApp(customtkinter.CTk):
    def __init__(self, backend):
        super().__init__(fg_color=DARK_BG)
        self.backend = backend
        self.waiting_for_answer = {}
        self.current_chat_id = None
        self.attachments = []
        self.active_chats = set()
        self.chat_processes = {}
        self.input_queues = {}
        self.output_queues = {}
        self.stop_events = {}
        self.pause_events = {}
        self.log_queues = {}
        self.log_windows = {}
        self.chat_blink_states = {}
        self.blink_timer_id = None
        self.settings_window = None
        self.create_chat_window = None
        self.attachment_overlay_frame = None
        self.title(Lang.get("app_title"))
        self.geometry("700x400")
        self.minsize(700, 400)
        self.after(0, lambda: set_windows_dark_titlebar(self))
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        if not self.backend.is_main_config_complete(): self.after(0, self.show_initial_settings)
        else:
            self.backend.rescan_and_localize_modules()
            # Инициализируем менеджер модулей после загрузки
            ModuleManager().load_modules(self.backend)
            self.setup_main_ui()
        
        self.bind("<Configure>", self._update_message_wraplengths)
        global initialize_work
        from cross_gpt import initialize_work

    @staticmethod
    def add_label_context_menu(master, widget):
        menu = tk.Menu(master, tearoff=0, bg=DARK_SECONDARY, fg=WHITE, relief="flat", borderwidth=0)
        
        def copy_action():
            try:
                copy_text = widget.cget("text")
                if copy_text:
                    master.clipboard_clear()
                    master.clipboard_append(copy_text)
            except tk.TclError: print("Clipboard error.")

        menu.add_command(label=Lang.get("copy"), command=copy_action)
        widget.bind("<Button-3>", lambda e: menu.tk_popup(e.x_root, e.y_root))
        widget.bind("<Button-2>", lambda e: menu.tk_popup(e.x_root, e.y_root))

    def bring_to_front(self):
        self.lift()
        self.attributes('-topmost', True)
        self.after(100, lambda: self.attributes('-topmost', False))
        self.focus_force()

    def flash_window(self):
        if sys.platform == "win32":
            try:
                import ctypes
                hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
                class FLASHWINFO(ctypes.Structure):
                    _fields_ = [("cbSize", ctypes.c_uint), ("hwnd", ctypes.c_void_p), ("dwFlags", ctypes.c_uint), ("uCount", ctypes.c_uint), ("dwTimeout", ctypes.c_uint)]
                flash_info = FLASHWINFO(cbSize=ctypes.sizeof(FLASHWINFO), hwnd=hwnd, dwFlags=2 | 12, uCount=3, dwTimeout=0)
                ctypes.windll.user32.FlashWindowEx(ctypes.byref(flash_info))
            except Exception as e: print(f"Ошибка мигания окна (Windows): {e}")
        elif sys.platform == "darwin":
            try:
                from AppKit import NSApp
                NSApp.requestUserAttention_(0)
            except ImportError: print("AppKit не доступен для мигания на macOS")
        else:
            try: self.bell()
            except Exception as e: print(f"Ошибка мигания окна (Linux): {e}")

    def on_close(self):
        active_chats = list(self.chat_processes.keys())
        if active_chats:
            try:
                # Создаем кастомное диалоговое окно с черным фоном
                dialog = tk.Toplevel(self)
                dialog.title(Lang.get("active_chats_on_close_title"))
                dialog.configure(bg=DARK_BG)
                dialog.geometry("400x200")  # Малый размер
                dialog.resizable(False, False)
                dialog.attributes('-topmost', True)
                # Центрируем окно
                dialog.update_idletasks()
                width = dialog.winfo_width()
                height = dialog.winfo_height()
                x = (self.winfo_screenwidth() // 2) - (width // 2)
                y = (self.winfo_screenheight() // 2) - (height // 2)
                dialog.geometry(f'{width}x{height}+{x}+{y}')
                # Устанавливаем темную тему
                dialog.after(10, lambda: set_windows_dark_titlebar(dialog))
                # Блокируем взаимодействие с главным окном
                dialog.transient(self)
                dialog.grab_set()
                # Создаем контент
                main_frame = customtkinter.CTkFrame(dialog, fg_color="transparent")
                main_frame.pack(fill="both", expand=True, padx=20, pady=20)
                # Сообщение
                customtkinter.CTkLabel(
                    main_frame, 
                    text=Lang.get("active_chats_on_close_message", count=len(active_chats)), 
                    wraplength=350,
                    font=("Arial", 12)
                ).pack(pady=(10, 5))
                customtkinter.CTkLabel(
                    main_frame, 
                    text=Lang.get("active_chats_on_close_detail"), 
                    wraplength=350, 
                    text_color=DARK_TEXT_SECONDARY,
                    font=("Arial", 11)
                ).pack(pady=(0, 20))
                # Кнопки
                btn_frame = customtkinter.CTkFrame(main_frame, fg_color="transparent")
                btn_frame.pack(fill="x", pady=(10, 0))
                
                def close_action(action):
                    dialog.grab_release()
                    dialog.destroy()
                    if action == "safe":
                        for chat_id in active_chats: self.stop_chat_process(chat_id)
                    elif action == "terminate":
                        for chat_id in active_chats: self.terminate_chat_process(chat_id)
                    if action != "cancel": self.after(100, self.destroy)
                customtkinter.CTkButton(
                    btn_frame, 
                    text=Lang.get("stop_safely"), 
                    **BUTTON_THEME, 
                    command=lambda: close_action("safe")
                ).pack(side="left", padx=5, fill="x", expand=True)
                customtkinter.CTkButton(
                    btn_frame, 
                    text=Lang.get("terminate"), 
                    **BUTTON_THEME, 
                    command=lambda: close_action("terminate")
                ).pack(side="left", padx=5, fill="x", expand=True)
                customtkinter.CTkButton(
                    btn_frame, 
                    text=Lang.get("cancel"), 
                    **BUTTON_THEME, 
                    command=lambda: close_action("cancel")
                ).pack(side="left", padx=5, fill="x", expand=True)
                # Обработка закрытия окна
                def on_dialog_close(): close_action("cancel")

                dialog.protocol("WM_DELETE_WINDOW", on_dialog_close)
                # Ждем закрытия диалога
                self.wait_window(dialog)
            except Exception as e:
                print(f"Error creating on_close popup: {e}")
                self.destroy()
        else: self.destroy()

    def stop_chat_process(self, chat_id):
        if chat_id in self.stop_events: self.stop_events[chat_id].set()
        if chat_id in self.chat_processes:
            process = self.chat_processes[chat_id]
            if process.is_alive(): process.join(timeout=2.0)
            if process.is_alive(): self.terminate_chat_process(chat_id)
            else: self._cleanup_chat_process_data(chat_id)
    
    def terminate_chat_process(self, chat_id):
        if chat_id in self.chat_processes:
            process = self.chat_processes[chat_id]
            if process.is_alive():
                process.terminate()
                process.join()
            self._cleanup_chat_process_data(chat_id)

    def _cleanup_chat_process_data(self, chat_id):
        if chat_id in self.log_windows:
            if self.log_windows[chat_id].winfo_exists(): self.log_windows[chat_id].destroy()
            del self.log_windows[chat_id]
        for d in [self.chat_processes, self.input_queues, self.output_queues, self.stop_events, self.pause_events, self.chat_blink_states, self.log_queues]:
            if chat_id in d: del d[chat_id]
        if chat_id in self.active_chats: self.active_chats.remove(chat_id)
        self.update_chat_list_colors()

    def start_chat_blinking(self):
        if self.blink_timer_id: self.after_cancel(self.blink_timer_id)
        for chat_id in list(self.chat_blink_states.keys()): self.chat_blink_states[chat_id] = not self.chat_blink_states[chat_id]
        self.update_chat_list_colors()
        self.blink_timer_id = self.after(1000, self.start_chat_blinking)
    
    def update_chat_list_colors(self):
        if not hasattr(self, 'chats_list_frame') or not self.chats_list_frame.winfo_exists(): return
        for row_frame in self.chats_list_frame.winfo_children():
            if not isinstance(row_frame, customtkinter.CTkFrame) or not hasattr(row_frame, 'winfo_children') or not row_frame.winfo_children(): continue
            chat_button = row_frame.winfo_children()[0]
            chat_id = getattr(chat_button, "chat_id", None)
            if not chat_id: continue
            is_blinking = self.chat_blink_states.get(chat_id, False)
            if self.current_chat_id == chat_id: color = PURPLE_ACCENT
            elif is_blinking: color = ACTIVE_CHAT_COLOR
            else: color = "transparent"
            chat_button.configure(fg_color=color)

    def show_initial_settings(self):
        self.withdraw()
        InitialSettingsWindow(self, self.backend)

    def browse_file(self, entry_widget, entry_var, full_path_var, filetypes=None):
        from tkinter import filedialog
        if filetypes is None: filetypes = [("All files", "*.*")]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            full_path_var.set(path)
            entry_var.set(Path(path).name)
            entry_widget.delete(0, "end")
            entry_widget.insert(0, Path(path).name)

    def setup_main_ui(self):
        for widget in self.winfo_children(): widget.destroy()
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        left_panel_container = customtkinter.CTkFrame(self, fg_color="transparent", width=200)
        left_panel_container.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        left_panel_container.grid_rowconfigure(0, weight=1)
        chats_bordered_frame = customtkinter.CTkFrame(left_panel_container, fg_color=DARK_BG, border_color=WHITE, border_width=1, corner_radius=CORNER_RADIUS)
        chats_bordered_frame.grid(row=0, column=0, sticky="nsew")
        self.chats_list_frame = customtkinter.CTkScrollableFrame(chats_bordered_frame, label_text="", fg_color="transparent", corner_radius=0, scrollbar_button_color=PURPLE_ACCENT, scrollbar_button_hover_color=WHITE)
        self.chats_list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        if hasattr(self.chats_list_frame, '_scrollbar'): self.chats_list_frame._scrollbar.configure(width=10)
        bottom_buttons_frame = customtkinter.CTkFrame(left_panel_container, fg_color="transparent")
        bottom_buttons_frame.grid(row=1, column=0, sticky="ew", pady=(5,0))
        bottom_buttons_frame.grid_columnconfigure((0,1), weight=1)
        self.new_chat_btn = customtkinter.CTkButton(bottom_buttons_frame, text=Lang.get("new_chat"), **BUTTON_THEME, command=self.create_chat_window_show)
        self.new_chat_btn.grid(row=0, column=0, padx=(0, 2), sticky="ew")
        self.settings_btn = customtkinter.CTkButton(bottom_buttons_frame, text=Lang.get("settings"), **BUTTON_THEME, command=self.open_settings)
        self.settings_btn.grid(row=0, column=1, padx=(2, 0), sticky="ew")
        right_panel_container = customtkinter.CTkFrame(self, fg_color="transparent")
        right_panel_container.grid(row=0, column=1, sticky="nsew", padx=(0, 5), pady=5)
        right_panel_container.grid_columnconfigure(0, weight=1)
        right_panel_container.grid_rowconfigure(0, weight=1)
        right_panel_container.grid_rowconfigure(1, weight=0)
        self.messages_bordered_frame = customtkinter.CTkFrame(right_panel_container, fg_color=DARK_BG, border_color=WHITE, border_width=1, corner_radius=CORNER_RADIUS)
        self.messages_bordered_frame.grid(row=0, column=0, sticky="nsew", pady=(0,5))
        self.messages_bordered_frame.grid_rowconfigure(0, weight=1)
        self.messages_bordered_frame.grid_columnconfigure(0, weight=1)
        self.messages_frame = customtkinter.CTkScrollableFrame(self.messages_bordered_frame, bg_color="transparent", fg_color="transparent", label_text="", scrollbar_button_color=PURPLE_ACCENT, scrollbar_button_hover_color=WHITE, border_width=0, corner_radius=CORNER_RADIUS,)
        self.messages_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        if hasattr(self.messages_frame, '_scrollbar'): self.messages_frame._scrollbar.configure(width=10)
        self.input_outer_frame = customtkinter.CTkFrame(right_panel_container, fg_color="transparent")
        self.input_outer_frame.grid(row=1, column=0, sticky="ew")
        self.input_outer_frame.grid_columnconfigure(1, weight=1)
        self.left_controls_frame = customtkinter.CTkFrame(self.input_outer_frame, fg_color="transparent")
        self.left_controls_frame.grid(row=0, column=0, padx=(0, 5), sticky="s")
        self.stop_btn = customtkinter.CTkButton(self.left_controls_frame, text="☐", width=30, height=30, **BUTTON_THEME, command=self.stop_chat)
        self.pause_btn = customtkinter.CTkButton(self.left_controls_frame, text="||", width=30, height=30, **BUTTON_THEME, command=self.pause_chat)
        self.play_btn = customtkinter.CTkButton(self.left_controls_frame, text="▷", width=30, height=30, **BUTTON_THEME, command=self.resume_chat)
        self.log_btn = customtkinter.CTkButton(self.left_controls_frame, text="Log", width=30, height=30, **BUTTON_THEME, command=self.open_log_window)
        self.input_text = customtkinter.CTkTextbox(self.input_outer_frame, corner_radius=CORNER_RADIUS, border_color=PURPLE_ACCENT, border_width=1, fg_color=DARK_SECONDARY, font=("Arial", 12), wrap="word")
        self.input_text.grid(row=0, column=1, sticky="nsew")
        self.input_text.bind("<Return>", self.on_enter_pressed)
        self.input_text.bind("<KeyRelease>", self.adjust_input_height, add=True)
        enhance_text_widget(self.input_text)
        self.adjust_input_height()
        self.right_controls_frame = customtkinter.CTkFrame(self.input_outer_frame, fg_color="transparent")
        self.right_controls_frame.grid(row=0, column=2, padx=(5,0), sticky="s")
        self.send_btn = customtkinter.CTkButton(self.right_controls_frame, text="↑", width=30, height=30, **BUTTON_THEME, command=self.send_message)
        self.send_btn.pack(side=tk.TOP, anchor="ne")
        self.attach_btn = customtkinter.CTkButton(self.right_controls_frame, text=" + ", width=30, height=30, **BUTTON_THEME, command=self.add_attachment)
        self.attach_btn.pack(side=tk.TOP, anchor="ne", pady=(5,0))
        self.load_chats()
        self.start_chat_blinking()
        self.update_chat_controls()
            
    def update_chat_controls(self):
        #for btn in [self.stop_btn, self.pause_btn, self.play_btn, self.log_btn]:
        for btn in [self.stop_btn, self.play_btn, self.log_btn]:
            btn.pack_forget()
        if not self.current_chat_id:
            self.play_btn.pack(side=tk.TOP)
            self.play_btn.configure(state="disabled")
            return
        has_messages = len(self.backend.get_messages(self.current_chat_id)) > 0
        is_active = self.current_chat_id in self.chat_processes
        is_paused = is_active and self.pause_events.get(self.current_chat_id, multiprocessing.Event()).is_set()
        if is_active:
            self.stop_btn.pack(side=tk.TOP)
            if is_paused:
                self.play_btn.pack(side=tk.TOP, pady=(5, 0))
                self.play_btn.configure(state="normal")
            else: pass
                #self.pause_btn.pack(side=tk.TOP, pady=(5, 0))
            self.log_btn.pack(side=tk.TOP, pady=(5, 0))
        else:
            self.play_btn.pack(side=tk.TOP)
            self.play_btn.configure(state="normal" if has_messages else "disabled")

    def stop_chat(self):
        if self.current_chat_id and self.current_chat_id in self.chat_processes:
            self.stop_chat_process(self.current_chat_id)
            self.update_chat_controls()
    
    def pause_chat(self):
        if self.current_chat_id and self.current_chat_id in self.pause_events:
            self.pause_events[self.current_chat_id].set()
            self.update_chat_controls()
    
    def resume_chat(self):
        if self.current_chat_id:
            if self.current_chat_id in self.pause_events and self.pause_events[self.current_chat_id].is_set():
                self.pause_events[self.current_chat_id].clear()
            elif self.current_chat_id not in self.chat_processes:
                self.start_chat_process(self.current_chat_id)
            self.update_chat_controls()
        
    def adjust_input_height(self, event=None):
        lines = self.input_text.get("1.0", "end-1c").count('\n') + 1
        height = min(max(lines, 1), 6) * 20 + 10
        self.input_text.configure(height=height)
        
    def on_enter_pressed(self, event):
        if not event.state & 0x1:
            self.send_message()
            return "break"
        return None
    
    def add_attachment(self):
        from tkinter import filedialog
        files = filedialog.askopenfilenames()
        if files:
            self.attachments.extend(Path(file) for file in files)
            self.show_attachments()
    
    def show_attachments(self):
        if hasattr(self, 'attachment_overlay_frame') and self.attachment_overlay_frame and self.attachment_overlay_frame.winfo_exists():
            self.attachment_overlay_frame.destroy()
        self.attachment_overlay_frame = None
        if not self.attachments: return
        self.attachment_overlay_frame = customtkinter.CTkFrame(self.messages_bordered_frame, fg_color=DARK_SECONDARY, border_color=PURPLE_ACCENT, border_width=1, corner_radius=CORNER_RADIUS)
        self.attachment_overlay_frame.place(relx=0.5, y=10, anchor='n', relwidth=0.75)
        header_text = f"{Lang.get('attachments')}"
        scrollable_container = customtkinter.CTkScrollableFrame(self.attachment_overlay_frame, fg_color="transparent", label_text=header_text, label_text_color=WHITE,
                                                                scrollbar_button_color=PURPLE_ACCENT, scrollbar_button_hover_color=WHITE)
        if hasattr(scrollable_container, '_scrollbar'): scrollable_container._scrollbar.configure(width=10)
        scrollable_container.pack(fill="both", expand=True, padx=5, pady=5)
        for i, attachment in enumerate(self.attachments):
            att_frame = customtkinter.CTkFrame(scrollable_container, fg_color="transparent")
            att_frame.pack(fill="x", pady=2, padx=2)
            customtkinter.CTkLabel(att_frame, text=attachment.name, font=("Arial", 11), anchor="w").pack(side=tk.LEFT, padx=5, expand=True, fill="x")
            customtkinter.CTkButton(att_frame, text="X", width=22, height=22, fg_color=DARK_SECONDARY, hover_color=PURPLE_ACCENT, text_color=WHITE,
                                    command=lambda idx=i: self.remove_attachment(idx)).pack(side=tk.RIGHT)

        def _update_height():
            try:
                if not self.attachment_overlay_frame or not self.attachment_overlay_frame.winfo_exists(): return
                self.attachment_overlay_frame.update_idletasks()
                required_height = 35 + len(self.attachments) * 30 
                parent_height = self.messages_bordered_frame.winfo_height()
                max_height = parent_height * 0.8
                final_height = min(required_height, max_height)
                self.attachment_overlay_frame.configure(height=final_height)
            except Exception: pass
        self.after(50, _update_height)
    
    def remove_attachment(self, index):
        self.attachments.pop(index)
        self.show_attachments()
        
    def load_chats(self):
        current_selection = self.current_chat_id
        for widget in self.chats_list_frame.winfo_children(): widget.destroy()
        chats = self.backend.get_chats()  # Теперь из кэша
        first_chat_id = chats[0]["id"] if chats else None
        for chat in chats:
            row_frame = customtkinter.CTkFrame(self.chats_list_frame, fg_color="transparent")
            row_frame.pack(fill="x", pady=1)
            row_frame.grid_columnconfigure(0, weight=1)
            chat_button = customtkinter.CTkButton(row_frame, text=chat['name'], anchor="w", fg_color="transparent", command=lambda c_id=chat["id"]: self.on_chat_select(c_id))
            chat_button.grid(row=0, column=0, sticky="ew")
            setattr(chat_button, "chat_id", chat["id"])
            # Проверяем существование папок files, console_folders, reports (изменение 3)
            files_path = Path(resource_path(os.path.join("data", "chats", chat["id"], "files")))
            console_folders_path = Path(resource_path(os.path.join("data", "chats", chat["id"], "console_folders")))
            reports_path = Path(resource_path(os.path.join("data", "chats", chat["id"], "reports")))
            has_files = files_path.exists() and files_path.is_dir()
            has_console_folders = console_folders_path.exists() and console_folders_path.is_dir()
            has_reports = reports_path.exists() and reports_path.is_dir()
            column_offset = 1
            if has_files:
                files_button = customtkinter.CTkButton(row_frame, text="💾", width=30, fg_color="transparent", hover_color=PURPLE_ACCENT, 
                                                       command=lambda c_id=chat["id"]: self.open_folder(c_id, "files"))
                files_button.grid(row=0, column=column_offset, padx=(2, 0))
                column_offset += 1
            if has_console_folders:
                console_button = customtkinter.CTkButton(row_frame, text=">_", width=30, fg_color="transparent", hover_color=PURPLE_ACCENT, 
                                                         command=lambda c_id=chat["id"]: self.open_folder(c_id, "console_folders"))
                console_button.grid(row=0, column=column_offset, padx=(2, 0))
                column_offset += 1
            if has_reports:
                reports_button = customtkinter.CTkButton(row_frame, text="📄", width=30, fg_color="transparent", hover_color=PURPLE_ACCENT, 
                                                         command=lambda c_id=chat["id"]: self.open_folder(c_id, "reports"))
                reports_button.grid(row=0, column=column_offset, padx=(2, 0))
                column_offset += 1
            delete_button = customtkinter.CTkButton(row_frame, text="X", width=30, fg_color="transparent", hover_color=PURPLE_ACCENT, command=lambda c_id=chat["id"]: self.delete_selected_chat(c_id))
            delete_button.grid(row=0, column=column_offset, padx=(2,0))
        self.chats_list_frame.update_idletasks()
        chat_ids = [c["id"] for c in chats]
        if current_selection not in chat_ids:
            self.on_chat_select(first_chat_id) if first_chat_id else self.clear_chat_view()
        self.update_chat_list_colors()

    def open_folder(self, chat_id, folder_name):
        """Открывает папку для указанного чата (изменение 3)"""
        folder_path = Path(resource_path(os.path.join("data", "chats", chat_id, folder_name)))
        if not folder_path.exists():
            showinfo(self, Lang.get("info"), Lang.get("folder_not_found", folder_name=folder_name))
            return
        import subprocess, platform
        try:
            if platform.system() == "Windows": os.startfile(str(folder_path))
            elif platform.system() == "Darwin": subprocess.run(["open", str(folder_path)])
            else: subprocess.run(["xdg-open", str(folder_path)])
        except Exception as e: showerror(self, Lang.get("error"), Lang.get("open_folder_error", e=str(e)))

    def clear_chat_view(self):
        self.current_chat_id = None
        self.clear_messages()
        self.update_chat_controls()
    
    def delete_selected_chat(self, chat_id):
        chat_name = ""
        for chat in self.backend.get_chats():  # Из кэша
            if chat['id'] == chat_id:
                chat_name = chat['name']
                break
        if chat_id in self.chat_processes:
            showwarning(self, Lang.get("active_chat_delete_title"), Lang.get("active_chat_delete_message", chat_name=chat_name))
            return
        if askyesno(self, Lang.get("delete_chat_confirm_title"), Lang.get("delete_chat_confirm_message", chat_name=chat_name)):
            if self.backend.delete_chat(chat_id): self.load_chats()
    
    def on_chat_select(self, chat_id):
        if chat_id == self.current_chat_id: return
        if chat_id in self.waiting_for_answer: del self.waiting_for_answer[chat_id]
        if chat_id in self.chat_blink_states: del self.chat_blink_states[chat_id]
        self.current_chat_id = chat_id
        self.load_chat_messages()
        self.update_chat_controls()
        self.update_chat_list_colors()
    
    def clear_messages(self):
        for widget in self.messages_frame.winfo_children(): widget.destroy()
    
    def load_chat_messages(self):
        self.clear_messages()
        if not self.current_chat_id: return
        self.messages_frame.update_idletasks()
        messages = self.backend.get_messages(self.current_chat_id)
        for msg in messages: self.add_message_to_ui(msg["text"], msg["isMy"], attachments=msg.get("attachments", []))
        self.after(100, lambda: self.messages_frame._parent_canvas.yview_moveto(1))
    
    def copy_text_to_clipboard(self, text):
        self.clipboard_clear()
        self.clipboard_append(text)

    def add_message_to_ui(self, text, is_my, is_question=False, attachments=None):
        pack_side = "right" if is_my else "left"
        anchor_side = "e" if is_my else "w"
        justify_side = tk.RIGHT if is_my else tk.LEFT
        border_color = PURPLE_ACCENT if is_my else WHITE
        row_frame = customtkinter.CTkFrame(self.messages_frame, fg_color="transparent")
        row_frame.pack(fill=tk.X, pady=2, padx=5)
        bubble = customtkinter.CTkFrame(row_frame, border_width=1, border_color=border_color, corner_radius=CORNER_RADIUS, fg_color=DARK_BG)
        bubble.pack(side=pack_side, expand=False, anchor=anchor_side)
        content_frame = customtkinter.CTkFrame(bubble, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=5, pady=5)
        if is_question and not is_my: customtkinter.CTkLabel(content_frame, text="❓", font=("Arial", 16)).pack(side=tk.LEFT, padx=(5,5))
        # Создаем CTkLabel с динамическим wraplength
        msg_text_widget = customtkinter.CTkLabel(
            content_frame, 
            text=text, 
            wraplength=400,  # Начальное значение
            justify=justify_side, 
            font=("Arial", 12), 
            anchor=anchor_side, 
            fg_color="transparent"
        )
        msg_text_widget.pack(fill=tk.X, expand=True)
        # Функция для обновления wraplength при изменении размера
        def update_wraplength(event=None):
            try:
                if msg_text_widget.winfo_exists():
                    # Вычисляем доступную ширину для текста
                    available_width = self.messages_frame.winfo_width() - 100
                    if available_width > 200: msg_text_widget.configure(wraplength=available_width)
            except Exception as e: print(f"Ошибка при обновлении wraplength: {e}")
        # Привязываем обновление к изменению размера messages_frame
        self.messages_frame.bind("<Configure>", update_wraplength)
        # Вызываем сразу для установки начального значения
        self.after(100, update_wraplength)
        # Контекстное меню для копирования
        def create_context_menu(event):
            menu = tk.Menu(self, tearoff=0, bg=DARK_SECONDARY, fg=WHITE)
            menu.add_command(label=Lang.get("copy"), command=lambda: self.copy_text_to_clipboard(text))
            menu.tk_popup(event.x_root, event.y_root)
        msg_text_widget.bind("<Button-3>", create_context_menu)
        if attachments:
            att_container = customtkinter.CTkFrame(bubble, fg_color="transparent")
            att_container.pack(fill="x", pady=(8, 5), padx=5)
            for att in attachments:
                att_frame = customtkinter.CTkFrame(att_container, fg_color="transparent")
                att_frame.pack(fill=tk.X, pady=1, anchor='w')
                customtkinter.CTkLabel(att_frame, text=Path(att).name).pack(side=tk.LEFT)
                customtkinter.CTkButton(att_frame, text="📂", font=("Arial", 12), width=25, height=25,
                    fg_color="transparent", hover_color=PURPLE_ACCENT, command=lambda a=att: self.open_attachment(a)).pack(side=tk.RIGHT)
        self.after(100, lambda: self.messages_frame._parent_canvas.yview_moveto(1.0))

    def open_attachment(self, file_path):
        import os, subprocess
        try:
            if sys.platform == "win32": os.startfile(file_path)
            elif sys.platform == "darwin": subprocess.run(["open", file_path])
            else: subprocess.run(["xdg-open", file_path])
        except Exception as e: showerror(self, Lang.get("error"), Lang.get("attachment_open_error", e=str(e)))
    
    def send_message(self):
        text = self.input_text.get("1.0", "end-1c").strip()
        if not text and not self.attachments: return
        if not self.current_chat_id:
            self.create_chat_window_show()
            return
        attachments_paths = [str(a.resolve()) for a in self.attachments] if self.attachments else []
        message_data = {'text': text, 'attachments': attachments_paths or None, 'command': 'answer_user' if self.waiting_for_answer.get(self.current_chat_id) else None}
        if self.waiting_for_answer.get(self.current_chat_id): self.waiting_for_answer[self.current_chat_id] = False
        if self.backend.add_message(self.current_chat_id, text, True, attachments_paths):
            self.add_message_to_ui(text, True, attachments=attachments_paths)
            self.input_text.delete("1.0", "end")
            self.attachments.clear()
            self.show_attachments()
            self.adjust_input_height()
            if self.current_chat_id in self.input_queues: self.input_queues[self.current_chat_id].put(message_data)
            if self.current_chat_id not in self.chat_processes: self.resume_chat()
            else: self.update_chat_controls()
    
    def start_chat_process(self, chat_id):
        input_queue, output_queue = multiprocessing.Queue(), multiprocessing.Queue()
        log_queue = multiprocessing.Queue()
        stop_event, pause_event = multiprocessing.Event(), multiprocessing.Event()
        p = multiprocessing.Process(target=initialize_work, args=(BASE_DIR, chat_id, input_queue, output_queue, stop_event, pause_event, log_queue))
        p.start()
        self.chat_processes[chat_id], self.input_queues[chat_id], self.output_queues[chat_id] = p, input_queue, output_queue
        self.log_queues[chat_id], self.stop_events[chat_id], self.pause_events[chat_id] = log_queue, stop_event, pause_event
        self.active_chats.add(chat_id)
        self.check_chat_responses(chat_id)
        self.update_chat_controls()
    
    def check_chat_responses(self, chat_id):
        import queue
        if chat_id not in self.output_queues: return
        try:
            while True:
                response = self.output_queues[chat_id].get_nowait()
                is_question = isinstance(response, dict) and response.get('command') == 'ask_user'
                message_text = response.get('text', '') if isinstance(response, dict) else str(response)
                if not message_text: continue
                if is_question: self.waiting_for_answer[chat_id] = True
                self.backend.add_message(chat_id, message_text, False)
                if chat_id == self.current_chat_id: self.add_message_to_ui(message_text, False, is_question=is_question)
                else: self.chat_blink_states[chat_id] = True
                if not self.focus_get(): self.flash_window()
        except queue.Empty: pass
        
        if chat_id in self.chat_processes and self.chat_processes[chat_id].is_alive(): self.after(500, lambda c=chat_id: self.check_chat_responses(c))
        else:
             if chat_id in self.active_chats:
                self._cleanup_chat_process_data(chat_id)
                if chat_id == self.current_chat_id: self.update_chat_controls()

    def create_chat_window_show(self):
        if self.create_chat_window and self.create_chat_window.winfo_exists():
            self.create_chat_window.lift()
            return
        self.create_chat_window = CreateChatWindow(self, self.backend)

    def open_settings(self):
        if self.settings_window and self.settings_window.winfo_exists():
            self.settings_window.lift()
            return
        self.settings_window = SettingsWindow(self, self.backend)

    def open_log_window(self):
        if not self.current_chat_id or self.current_chat_id not in self.chat_processes: return
        if self.current_chat_id in self.log_windows and self.log_windows[self.current_chat_id].winfo_exists():
            self.log_windows[self.current_chat_id].lift()
            return
        log_queue = self.log_queues.get(self.current_chat_id)
        if log_queue:
            log_win = LogWindow(self, self.current_chat_id, log_queue)
            self.log_windows[self.current_chat_id] = log_win

    def _update_message_wraplengths(self, event=None):
        if not hasattr(self, 'messages_frame') or not self.messages_frame.winfo_exists(): return
        for row_frame in self.messages_frame.winfo_children():
            try:
                bubble = next((w for w in row_frame.winfo_children() if isinstance(w, customtkinter.CTkFrame)), None)
                if not bubble: continue
                content_frame = next((w for w in bubble.winfo_children() if isinstance(w, customtkinter.CTkFrame)), None)
                if not content_frame: continue
                for widget in content_frame.winfo_children():
                    if isinstance(widget, customtkinter.CTkTextbox): self._adjust_textbox_height(widget)
            except (IndexError, tk.TclError, StopIteration): continue

class BaseTopLevel(customtkinter.CTkToplevel):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.geometry("600x600")
        self.minsize(600, 600)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(10, lambda: set_windows_dark_titlebar(self))
        self.after_idle(self.setup_and_center)

    def setup_and_center(self):
        setup_icon(self)
        self.center_window()
        self.lift()
        self.grab_set()

    def center_window(self):
        self.update_idletasks()
        try:
            width, height = self.winfo_width(), self.winfo_height()
            x = (self.winfo_screenwidth() // 2) - (width // 2)
            y = (self.winfo_screenheight() // 2) - (height // 2)
            self.geometry(f'{width}x{height}+{x}+{y}')
        except tk.TclError: pass

    def on_close(self):
        self.grab_release()
        self.destroy()

    def browse_file(self, entry_widget, entry_var, full_path_var, filetypes=None):
        ChatApp.browse_file(self, entry_widget, entry_var, full_path_var, filetypes)

# ====== СТИЛИЗОВАННЫЕ ДИАЛОГИ (без изменений) ======
class CustomMessageBox(BaseTopLevel):
    def __init__(self, parent, title, message, buttons):
        lines = message.count('\n') + 1
        width = 400
        height = 120 + lines * 15
        height = min(max(height, 180), 500)
        super().__init__(parent)
        self.title(title)
        self.geometry(f"{width}x{height}")
        self.minsize(width, 180)
        self.maxsize(width, 500)
        self.configure(fg_color=DARK_BG)
        self.result = None
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        main_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        message_label = customtkinter.CTkLabel(main_frame, text=message, wraplength=width - 60, justify="left", font=("Arial", 13))
        message_label.grid(row=0, column=0, sticky="nsew")
        btn_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=1, column=0, sticky="se", padx=20, pady=(0, 20))
        for text_key, value in buttons:
            btn_text = Lang.get(text_key.lower())
            btn = customtkinter.CTkButton(btn_frame, text=btn_text, **BUTTON_THEME, command=lambda v=value: self.set_result(v))
            btn.pack(side="left", padx=(10,0))
            if text_key.lower() in ["ok", "yes"]: self.bind("<Return>", lambda e, v=value: self.set_result(v))
        self.bind("<Escape>", lambda e: self.on_close())
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(10, self.setup_and_center)

    def set_result(self, value):
        self.result = value
        self.on_close()

    def on_close(self):
        if self.result is None:
            is_yesno = any(b[0].lower() == 'no' for b in [])
            self.result = False if is_yesno else None
        super().on_close()

def showinfo(parent, title, message):
    dialog = CustomMessageBox(parent, title, message, [("OK", True)])
    parent.wait_window(dialog)

def showerror(parent, title, message):
    dialog = CustomMessageBox(parent, title, message, [("OK", True)])
    parent.wait_window(dialog)

def showwarning(parent, title, message):
    dialog = CustomMessageBox(parent, title, message, [("OK", True)])
    parent.wait_window(dialog)
    
def askyesno(parent, title, message):
    dialog = CustomMessageBox(parent, title, message, [("Yes", True), ("No", False)])
    parent.wait_window(dialog)
    return dialog.result

# ====== UI ДЛЯ НАСТРОЕК МОДЕЛЕЙ (С ИЗМЕНЕНИЯМИ: не сканирует заново провайдеры) ======
class DynamicModelUI:
    def __init__(self):
        self.provider_param_full_paths = {}
        self.model_frames = {}
        # Используем уже отсканированные провайдеры
        self.provider_manager = ProviderManager()
        self.providers = self.provider_manager.get_providers()

    def _create_model_ui(self, parent):
        # Убрано повторное сканирование провайдеров
        # self.provider_manager.scan_providers() - УДАЛЕНО
        # self.providers = self.provider_manager.get_providers() - УДАЛЕНО
        model_type_frame = customtkinter.CTkFrame(parent, fg_color="transparent")
        model_type_frame.pack(fill="x", pady=10)
        customtkinter.CTkLabel(model_type_frame, text=Lang.get("model_type")).pack(side="top")
        radio_frame = customtkinter.CTkFrame(parent, fg_color="transparent")
        radio_frame.pack(fill="x")
        provider_items = list(self.providers.items())
        if not provider_items:
            customtkinter.CTkLabel(radio_frame, text=Lang.get("no_providers_found")).pack()
            return
        for i, (module_name, data) in enumerate(provider_items):
            display_name = data['name']
            customtkinter.CTkRadioButton(
                radio_frame, text=display_name, variable=self.settings_vars['model_type'],
                value=module_name,
                command=self.toggle_model_frames, fg_color=PURPLE_ACCENT
            ).grid(row=i // 4, column=i % 4, sticky="w", padx=5, pady=2)
        self.frames_container = customtkinter.CTkFrame(parent, fg_color="transparent")
        self.frames_container.pack(fill="x", expand=True, pady=10)
        self.model_frames = {}
        self._create_specific_model_frames(self.frames_container)
        token_frame = customtkinter.CTkFrame(parent, fg_color="transparent")
        token_frame.pack(fill="x", pady=5)
        self.token_label = customtkinter.CTkLabel(token_frame, text=Lang.get("token_limit"))
        self.token_label.pack(side="left", padx=(0,10))
        self.token_entry = customtkinter.CTkEntry(token_frame, textvariable=self.settings_vars['token_limit'], **ENTRY_THEME)
        self.token_entry.pack(side="left", fill="x", expand=True)
        enhance_text_widget(self.token_entry)
        if provider_items and not self.settings_vars['model_type'].get(): self.settings_vars['model_type'].set(provider_items[0][0])
        # Теперь toggle_model_frames будет обращаться к уже созданной save_btn
        self.toggle_model_frames()

    def _create_specific_model_frames(self, container):
        try:
            for module_name, p_data in self.providers.items():
                main_frame = customtkinter.CTkFrame(container, fg_color="transparent")
                self.model_frames[module_name] = main_frame
                params = p_data.get('params', [])
                has_scroll = len(params) > 6
                content_parent = main_frame
                if has_scroll:
                    scroll_frame = customtkinter.CTkScrollableFrame(main_frame, fg_color=DARK_BG, 
                                                                    scrollbar_button_color=PURPLE_ACCENT,
                                                                    scrollbar_button_hover_color=WHITE)
                    if hasattr(scroll_frame, '_scrollbar'): scroll_frame._scrollbar.configure(width=10)
                    scroll_frame.pack(fill="both", expand=True)
                    content_parent = scroll_frame
                self.provider_param_full_paths.setdefault(module_name, {})
                self.settings_vars.setdefault(module_name, {})
                for param in params:
                    param_name = param['name']
                    default_val = param.get('default')
                    self.settings_vars[module_name][param_name] = tk.StringVar(value='')
                    param_frame = customtkinter.CTkFrame(content_parent, fg_color="transparent")
                    param_frame.pack(fill="x", pady=2)
                    param_frame.grid_columnconfigure(1, weight=1)
                    label_text = param_name
                    if default_val is not None and str(default_val).strip() != '': label_text += f" {default_val}"
                    customtkinter.CTkLabel(param_frame, text=label_text).grid(row=0, column=0, sticky="w", padx=(0, 10))
                    input_frame = customtkinter.CTkFrame(param_frame, fg_color="transparent")
                    input_frame.grid(row=0, column=1, sticky="ew")
                    input_frame.grid_columnconfigure(0, weight=1)
                    if param['is_file']:
                        self.provider_param_full_paths[module_name][param_name] = tk.StringVar()
                        display_var = tk.StringVar()
                        entry = customtkinter.CTkEntry(input_frame, textvariable=display_var, **ENTRY_THEME)
                        entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
                        browse_cmd = lambda e=entry, dv=display_var, fpv=self.provider_param_full_paths[module_name][param_name]: \
                            self.browse_file(e, dv, fpv, [(Lang.get("all_files"), "*.*")])
                        customtkinter.CTkButton(input_frame, text=Lang.get("browse"), width=80, **BUTTON_THEME, command=browse_cmd).grid(row=0, column=1)
                    else:
                        entry = customtkinter.CTkEntry(input_frame, textvariable=self.settings_vars[module_name][param_name], **ENTRY_THEME)
                        entry.grid(row=0, column=0, sticky="ew")
                    enhance_text_widget(entry)
        except Exception as e: print(f"Error creating specific model frames: {e}")

    def _load_provider_params_from_string(self):
        params_str = self.settings_vars['model_provider_params'].get()
        current_provider_module = self.settings_vars['model_type'].get()
        if not params_str or not current_provider_module: return
        try:
            params_map = dict(part.split('=', 1) for part in params_str.split(';') if '=' in part)
            provider_ui_vars = self.settings_vars.get(current_provider_module, {})
            provider_path_vars = self.provider_param_full_paths.get(current_provider_module, {})
            provider_info = self.providers.get(current_provider_module, {})
            if not provider_info: return
            for param_info in provider_info.get('params', []):
                param_name = param_info['name']
                value = params_map.get(param_name, '')
                if param_info['is_file']:
                    if param_name in provider_path_vars: provider_path_vars[param_name].set(value)
                else:
                    if param_name in provider_ui_vars: provider_ui_vars[param_name].set(value)
        except (ValueError, KeyError) as e: print(f"Warning: Could not parse provider params string: {params_str}. Error: {e}")

    def _build_connection_string(self) -> str:
        provider_module_name = self.settings_vars['model_type'].get()
        if not provider_module_name: return ""
        provider_data = self.providers.get(provider_module_name)
        if not provider_data: return ""
        parts = []
        provider_params_info = provider_data.get('params', [])
        provider_ui_vars = self.settings_vars.get(provider_module_name, {})
        provider_path_vars = self.provider_param_full_paths.get(provider_module_name, {})
        for param in provider_params_info:
            param_name = param['name']
            value = ""
            if param['is_file']:
                if param_name in provider_path_vars: value = provider_path_vars[param_name].get().strip()
            else:
                if param_name in provider_ui_vars: value = provider_ui_vars[param_name].get().strip()
            if not value:
                default_val = param.get('default')
                if default_val is not None: value = str(default_val)
            if value: parts.append(f"{param_name}={value}")
        return ";".join(parts)
        
    def toggle_model_frames(self):
        self.validated = False
        # Безопасная проверка существования save_btn
        try:
            if hasattr(self, 'save_btn') and self.save_btn.winfo_exists(): self.save_btn.configure(state="disabled")
        except (tk.TclError, AttributeError): pass
        selected_type = self.settings_vars['model_type'].get()
        # Безопасно скрываем/показываем фреймы
        for name, frame in self.model_frames.items():
            try:
                if name == selected_type and frame.winfo_exists(): frame.pack(fill="x", expand=True)
                elif frame.winfo_exists(): frame.pack_forget()
            except (tk.TclError, AttributeError): continue
        self._load_provider_params_from_string()


class InitialSettingsWindow(BaseTopLevel, DynamicModelUI):
    def __init__(self, master, backend):
        super().__init__(master)
        DynamicModelUI.__init__(self)
        self.master = master
        self.backend = backend
        self.title("Setup") 
        self.geometry("600x500")
        self.minsize(600, 500)
        self.configure(fg_color=DARK_BG)
        self.max_tokens = 8192
        self.validated = False
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.show_step1_language()

    def show_step1_language(self):
        for widget in self.winfo_children(): widget.destroy()
        self.lang_var = tk.StringVar(value=Lang.current_language or "en")
        container = customtkinter.CTkFrame(self, fg_color="transparent")
        container.pack(expand=True, fill='both')
        lang_frame = customtkinter.CTkFrame(container, fg_color="transparent")
        lang_frame.pack(pady=10)
        customtkinter.CTkLabel(lang_frame, text=f"{Lang.get('language')}:").pack(side='left', padx=(0, 10))
        lang_combo = customtkinter.CTkOptionMenu(
            lang_frame, variable=self.lang_var, 
            values=list(Lang.available_languages.keys()), **OPTIONMENU_THEME)
        lang_combo.pack(side='left')
        customtkinter.CTkButton(container, text="→", **BUTTON_THEME, command=self.show_step2_model).pack(pady=20)

    def show_step2_model(self):
        Lang.load_language(self.lang_var.get())
        self.title(Lang.get("initial_settings_title"))
        self.backend.rescan_and_localize_modules()
        # Инициализируем менеджер модулей после загрузки языка
        ModuleManager().load_modules(self.backend)
        for widget in self.winfo_children(): widget.destroy()
        self.settings_vars = self._get_default_settings()
        # Создаем кнопки ДО вызова _create_model_ui
        btn_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(side="bottom", fill="x", pady=(0, 20), padx=20)
        self.validate_btn = customtkinter.CTkButton(btn_frame, text=Lang.get("validate_model"), **BUTTON_THEME, command=self.validate_model)
        self.validate_btn.pack(side="left")
        back_btn = customtkinter.CTkButton(btn_frame, text="←", **BUTTON_THEME, command=self.show_step1_language)
        back_btn.pack(side="left", padx=(10, 10))
        self.save_btn = customtkinter.CTkButton(btn_frame, text=Lang.get("save_and_continue"), **BUTTON_THEME, command=self.save_settings, state="disabled")
        self.save_btn.pack(side="right")
        main_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        # Теперь создаем UI модели
        self._create_model_ui(main_frame)
        self._load_provider_params_from_string()

    def _get_default_settings(self):
        settings = self.backend.get_global_settings()  # Из кэша
        settings['language'] = self.lang_var.get()
        return {key: tk.StringVar(value=val) for key, val in settings.items()}

    def validate_model(self):
        model_type = self.settings_vars['model_type'].get()
        connection_string = self._build_connection_string()
        valid, msg, max_tokens = self.backend.validate_model_settings(model_type, connection_string)
        if valid:
            self.max_tokens = max_tokens
            self.validated = True
            showinfo(self, Lang.get("success"), msg)
            self.token_label.configure(text=Lang.get("token_limit_info", max_tokens=self.max_tokens))
            self.settings_vars['token_limit'].set(str(self.max_tokens))
            # Безопасное обновление состояния кнопки
            try:
                if hasattr(self, 'save_btn') and self.save_btn.winfo_exists(): self.save_btn.configure(state="normal")
            except (tk.TclError, AttributeError): pass
        else:
            self.validated = False
            showerror(self, Lang.get("validation_error"), msg)
            # Безопасное обновление состояния кнопки
            try:
                if hasattr(self, 'save_btn') and self.save_btn.winfo_exists(): self.save_btn.configure(state="disabled")
            except (tk.TclError, AttributeError): pass

    def save_settings(self):
        if not self.validated:
            showerror(self, Lang.get("error"), Lang.get("model_not_validated"))
            return
        try:
            token_limit = int(self.settings_vars['token_limit'].get())
            if not (1 <= token_limit <= self.max_tokens): raise ValueError
        except (ValueError, TypeError):
            showerror(self, Lang.get("error"), Lang.get("token_limit_info", max_tokens=self.max_tokens))
            return
        settings_to_save = {
            'language': self.lang_var.get(),
            'model_type': self.settings_vars['model_type'].get(),
            'token_limit': self.settings_vars['token_limit'].get(),
            'max_tasks': self.settings_vars['max_tasks'].get(),
            'model_provider_params': self._build_connection_string()
        }
        self.backend.update_global_settings(settings_to_save)
        self.on_close()

    def on_close(self):
        super().on_close()
        if self.master.winfo_exists():
            if self.backend.is_main_config_complete():
                self.master.deiconify()
                self.master.setup_main_ui()
            else: self.master.destroy()


class SettingsWindow(BaseTopLevel, DynamicModelUI):
    def __init__(self, master, backend):
        super().__init__(master)
        DynamicModelUI.__init__(self)
        self.master = master
        self.backend = backend
        self.title(Lang.get("settings_title"))
        self.configure(fg_color=DARK_BG)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        tabview = customtkinter.CTkTabview(self, **TAB_VIEW_THEME)
        tabview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_tab = tabview.add(Lang.get("tab_main"))
        chat_settings_tab = tabview.add(Lang.get("tab_chat_settings"))
        mods_tab = tabview.add(Lang.get("tab_modules"))
        self.setup_main_tab(main_tab)
        self.setup_chat_settings_tab(chat_settings_tab)
        self.setup_mods_tab(mods_tab)

    def setup_main_tab(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        self.max_tokens = int(self.backend.get_global_settings().get("token_limit", 8192))  # Из кэша
        self.validated = True
        self.settings_vars = self._get_default_settings()
        self.original_language = self.settings_vars['language'].get()
        self.original_model_type = self.settings_vars['model_type'].get()
        self.original_connection_string = self.settings_vars['model_provider_params'].get()
        main_frame = customtkinter.CTkFrame(parent, fg_color="transparent")
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_frame.grid_columnconfigure(0, weight=1)
        lang_frame = customtkinter.CTkFrame(main_frame, fg_color="transparent")
        lang_frame.pack(fill='x', pady=5)
        lang_frame.grid_columnconfigure(0, weight=1)
        lang_frame.grid_columnconfigure(2, weight=1)
        lang_content = customtkinter.CTkFrame(lang_frame, fg_color="transparent")
        lang_content.grid(row=0, column=1, sticky="ew")
        customtkinter.CTkLabel(lang_content, text=f"{Lang.get('language')}:").grid(row=0, column=0, padx=(0, 10), sticky="w")
        self.lang_combo = customtkinter.CTkOptionMenu(
            lang_content, variable=self.settings_vars['language'], 
            values=list(Lang.available_languages.keys()), **OPTIONMENU_THEME)
        self.lang_combo.grid(row=0, column=1, sticky="ew")
        lang_content.grid_columnconfigure(1, weight=1)
        self._create_model_ui(main_frame)
        btn_frame = customtkinter.CTkFrame(parent, fg_color="transparent")
        btn_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        customtkinter.CTkButton(btn_frame, text=Lang.get("validate_model"), **BUTTON_THEME, command=self.validate_model).pack(side="left", padx=5)
        self.save_btn_settings = customtkinter.CTkButton(btn_frame, text=Lang.get("save"), **BUTTON_THEME, command=self.save_settings)
        self.save_btn_settings.pack(side="left", padx=5)
        customtkinter.CTkButton(btn_frame, text=Lang.get("reset_settings_button"), **BUTTON_THEME, command=self.reset_settings).pack(side="right", padx=5)
        self._load_provider_params_from_string()
    
    def _get_default_settings(self):
        settings = self.backend.get_global_settings()  # Из кэша
        return {key: tk.StringVar(value=val) for key, val in settings.items()}
    
    def validate_model(self):
        model_type = self.settings_vars['model_type'].get()
        connection_string = self._build_connection_string()
        valid, msg, max_tokens = self.backend.validate_model_settings(model_type, connection_string)
        if valid:
            self.max_tokens = max_tokens
            self.validated = True
            self.original_model_type = model_type
            self.original_connection_string = connection_string
            showinfo(self, Lang.get("success"), msg)
            self.token_label.configure(text=Lang.get("token_limit_info", max_tokens=self.max_tokens))
            self.settings_vars['token_limit'].set(str(self.max_tokens))
        else:
            self.validated = False
            showerror(self, Lang.get("validation_error"), msg)

    def save_settings(self):
        # Сначала закрываем окно (изменение 2)
        self.on_close()
        # Блокируем кнопки в главном окне
        if self.master.winfo_exists():
            if hasattr(self.master, 'new_chat_btn'): self.master.new_chat_btn.configure(state="disabled")
            if hasattr(self.master, 'settings_btn'): self.master.settings_btn.configure(state="disabled")
            if hasattr(self.master, 'send_btn'): self.master.send_btn.configure(state="disabled")
        current_model_type = self.settings_vars['model_type'].get()
        current_connection_string = self._build_connection_string()
        settings_changed = (current_model_type != self.original_model_type or current_connection_string != self.original_connection_string)
        if not self.validated and settings_changed:
            if not askyesno(self.master, Lang.get("warning"), Lang.get("model_not_validated_continue")):
                # Разблокируем кнопки при отмене
                if self.master.winfo_exists():
                    if hasattr(self.master, 'new_chat_btn'): self.master.new_chat_btn.configure(state="normal")
                    if hasattr(self.master, 'settings_btn'): self.master.settings_btn.configure(state="normal")
                    if hasattr(self.master, 'send_btn'): self.master.send_btn.configure(state="normal")
                return
        try:
            token_limit = int(self.settings_vars['token_limit'].get())
            if not (1 <= token_limit <= self.max_tokens): raise ValueError
        except (ValueError, TypeError):
            # Разблокируем кнопки при ошибке
            if self.master.winfo_exists():
                if hasattr(self.master, 'new_chat_btn'):
                    self.master.new_chat_btn.configure(state="normal")
                if hasattr(self.master, 'settings_btn'):
                    self.master.settings_btn.configure(state="normal")
                if hasattr(self.master, 'send_btn'):
                    self.master.send_btn.configure(state="normal")
            showerror(self.master, Lang.get("error"), Lang.get("token_limit_info", max_tokens=self.max_tokens))
            return
        settings_to_save = {
            'language': self.settings_vars['language'].get(),
            'model_type': self.settings_vars['model_type'].get(),
            'token_limit': self.settings_vars['token_limit'].get(),
            'max_tasks': self.settings_vars['max_tasks'].get(),
            'model_provider_params': self._build_connection_string(),
            'use_rag': self.settings_vars['use_rag'].get(),
            'filter_generations': self.settings_vars['filter_generations'].get(),
            'hierarchy_limit': self.settings_vars['hierarchy_limit'].get(),
            'write_log': self.settings_vars['write_log'].get()
        }
        self.backend.update_global_settings(settings_to_save)
        new_language = self.settings_vars['language'].get()
        if new_language != self.original_language: Lang.load_language(new_language)
        # Разблокируем кнопки
        if self.master.winfo_exists():
            if hasattr(self.master, 'new_chat_btn'): self.master.new_chat_btn.configure(state="normal")
            if hasattr(self.master, 'settings_btn'): self.master.settings_btn.configure(state="normal")
            if hasattr(self.master, 'send_btn'): self.master.send_btn.configure(state="normal")

    def reset_settings(self):
        if askyesno(self, Lang.get("reset_settings_confirm_title"), Lang.get("reset_settings_confirm_message")):
            db_path = Path(self.backend.db_path)
            if db_path.exists(): db_path.unlink()
            showinfo(self, Lang.get("info"), Lang.get("restart_required"))
            self.master.destroy()

    def setup_chat_settings_tab(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        frame = customtkinter.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        frame.grid_columnconfigure(1, weight=1)
        # Использовать RAG (по умолчанию вкл)
        self.settings_vars['use_rag'] = tk.StringVar(value=self.backend.get_global_settings().get('use_rag', '1'))
        customtkinter.CTkLabel(frame, text=Lang.get("use_rag")).grid(row=0, column=0, sticky='w', pady=5)
        switch_rag = customtkinter.CTkSwitch(frame, text="", variable=self.settings_vars['use_rag'], 
                               onvalue="1", offvalue="0", 
                               switch_width=50, switch_height=25,
                               progress_color=PURPLE_ACCENT)
        switch_rag.grid(row=0, column=1, sticky='w', padx=10, pady=5)
        # Фильтровать генерации (по умолчанию выкл)
        self.settings_vars['filter_generations'] = tk.StringVar(value=self.backend.get_global_settings().get('filter_generations', '0'))
        customtkinter.CTkLabel(frame, text=Lang.get("filter_generations")).grid(row=1, column=0, sticky='w', pady=5)
        switch_filter = customtkinter.CTkSwitch(frame, text="", variable=self.settings_vars['filter_generations'], 
                               onvalue="1", offvalue="0",
                               switch_width=50, switch_height=25,
                               progress_color=PURPLE_ACCENT)
        switch_filter.grid(row=1, column=1, sticky='w', padx=10, pady=5)
        # Записывать лог (по умолчанию вкл)
        self.settings_vars['write_log'] = tk.StringVar(value=self.backend.get_global_settings().get('write_log', '1'))
        customtkinter.CTkLabel(frame, text=Lang.get("write_log")).grid(row=2, column=0, sticky='w', pady=5)
        switch_write_log = customtkinter.CTkSwitch(frame, text="", variable=self.settings_vars['write_log'], 
                               onvalue="1", offvalue="0",
                               switch_width=50, switch_height=25,
                               progress_color=PURPLE_ACCENT)
        switch_write_log.grid(row=2, column=1, sticky='w', padx=10, pady=5)
        # Максимальное количество задач
        customtkinter.CTkLabel(frame, text=Lang.get("max_tasks")).grid(row=3, column=0, sticky='w', pady=5)
        e_max_tasks = customtkinter.CTkEntry(frame, textvariable=self.settings_vars['max_tasks'], **ENTRY_THEME)
        e_max_tasks.grid(row=3, column=1, sticky='ew', padx=10, pady=5)
        enhance_text_widget(e_max_tasks)
        # Ограничение ступеней иерархии (по умолчанию 0)
        self.settings_vars['hierarchy_limit'] = tk.StringVar(value=self.backend.get_global_settings().get('hierarchy_limit', '0'))
        customtkinter.CTkLabel(frame, text=Lang.get("hierarchy_limit")).grid(row=4, column=0, sticky='w', pady=5)
        e_hierarchy = customtkinter.CTkEntry(frame, textvariable=self.settings_vars['hierarchy_limit'], **ENTRY_THEME)
        e_hierarchy.grid(row=4, column=1, sticky='ew', padx=10, pady=5)
        enhance_text_widget(e_hierarchy)

    def setup_mods_tab(self, parent):
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        self.scrollable_frame = customtkinter.CTkScrollableFrame(parent, fg_color="transparent", scrollbar_button_color=PURPLE_ACCENT, scrollbar_button_hover_color=WHITE, label_text="")
        if hasattr(self.scrollable_frame, '_scrollbar'): self.scrollable_frame._scrollbar.configure(width=10)
        self.scrollable_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.rebuild_mods_list()
        customtkinter.CTkButton(parent, text=Lang.get("add_module"), **BUTTON_THEME, command=self.add_custom_mod).grid(row=1, column=0, pady=10, padx=5)

    def rebuild_mods_list(self):
        for widget in self.scrollable_frame.winfo_children(): widget.destroy()
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        # Используем кэшированные модули вместо повторной загрузки из базы
        module_manager = ModuleManager()
        default_mods = module_manager.get_default_modules()
        custom_mods = module_manager.get_custom_modules()
        if default_mods:
            customtkinter.CTkLabel(self.scrollable_frame, text=Lang.get("system_modules"), font=("Arial", 14, "bold")).pack(anchor="w", padx=5, pady=(10,5))
            for mod in default_mods: self.create_mod_ui(self.scrollable_frame, mod, is_default=True)
        customtkinter.CTkLabel(self.scrollable_frame, text=Lang.get("global_custom_modules"), font=("Arial", 14, "bold")).pack(anchor="w", padx=5, pady=(15,5))
        for mod in custom_mods: self.create_mod_ui(self.scrollable_frame, mod, is_default=False)

    def create_mod_ui(self, parent, mod_data, is_default):
        frame = customtkinter.CTkFrame(parent, border_width=1, border_color=DARK_BORDER, corner_radius=CORNER_RADIUS)
        frame.pack(fill="x", padx=5, pady=3, ipady=5)
        frame.grid_columnconfigure(1, weight=1)
        enabled_var = tk.BooleanVar(value=mod_data["enabled"])
        cb = customtkinter.CTkCheckBox(frame, variable=enabled_var, text="", border_color=WHITE, checkmark_color=PURPLE_ACCENT,
            command=lambda mid=mod_data["id"], var=enabled_var: self.toggle_default_mod(mid, var.get()))
        cb.grid(row=0, column=0, rowspan=2, padx=10)
        if not is_default: cb.configure(state="disabled") 
        info_frame = customtkinter.CTkFrame(frame, fg_color="transparent")
        info_frame.grid(row=0, column=1, rowspan=2, sticky="ew")
        customtkinter.CTkLabel(info_frame, text=mod_data["name"], font=("Arial", 12, "bold")).pack(anchor="w")
        desc_label = customtkinter.CTkLabel(info_frame, text=mod_data["description"], text_color=DARK_TEXT_SECONDARY, wraplength=400, justify="left")
        desc_label.pack(anchor="w", fill="x")
        ChatApp.add_label_context_menu(self, desc_label)
        if not is_default:
            remove_btn = customtkinter.CTkButton(frame, text="X", width=25, height=25, fg_color="transparent", hover_color=PURPLE_ACCENT, text_color=WHITE,
                                                 command=lambda mid=mod_data["id"]: self.remove_custom_mod(mid))
            remove_btn.grid(row=0, column=2, rowspan=2, padx=10)

    def toggle_default_mod(self, mod_id, enabled): self.backend.update_default_mod_enabled(mod_id, enabled)

    def remove_custom_mod(self, mod_id):
        if askyesno(self, Lang.get("warning"), Lang.get("remove_module_confirm")):
            self.backend.remove_custom_mod(mod_id)
            # Обновляем кэш после удаления модуля
            ModuleManager().update_custom_modules(self.backend)
            self.rebuild_mods_list()

    def add_custom_mod(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=[(Lang.get("python_files"), "*.py")])
        if not path: return
        try:
            self.backend.add_custom_mod(str(Path(path).resolve()))
            # Обновляем кэш после добавления модуля
            ModuleManager().update_custom_modules(self.backend)
            self.rebuild_mods_list()
        except ValueError as e: showerror(self, Lang.get("error"), str(e))

class LogWindow(BaseTopLevel):
    def __init__(self, master, chat_id, log_queue):
        super().__init__(master, fg_color=DARK_BG)
        self.geometry("400x200")
        self.minsize(400, 200)
        self.master, self.chat_id, self.log_queue = master, chat_id, log_queue
        chat_name = next((chat['name'] for chat in self.master.backend.get_chats() if chat['id'] == chat_id), chat_id)
        self.title(f"Log - {chat_name}")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.messages_frame = customtkinter.CTkScrollableFrame(self, scrollbar_button_color=PURPLE_ACCENT, scrollbar_button_hover_color=WHITE, fg_color="transparent", label_text="")
        if hasattr(self.messages_frame, '_scrollbar'): self.messages_frame._scrollbar.configure(width=10)
        self.messages_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.messages_frame.grid_columnconfigure(0, weight=1)
        self.log_message_widgets = []
        #self.bind("<Configure>", self._update_message_wraplengths)
        self.check_log_queue()

    def add_log_message_to_ui(self, text):
        bubble = customtkinter.CTkFrame(self.messages_frame, border_width=1, border_color=PURPLE_ACCENT, corner_radius=CORNER_RADIUS, fg_color=DARK_BG)
        bubble.pack(fill=tk.X, padx=5, pady=3, ipady=5)
        msg_widget = customtkinter.CTkLabel(bubble, text=text, wraplength=self.messages_frame.winfo_width() - 50, justify="left", fg_color="transparent")
        msg_widget.pack(fill=tk.X, padx=10, pady=5)
        ChatApp.add_label_context_menu(self.master, msg_widget)
        self.log_message_widgets.append(bubble)
        if len(self.log_message_widgets) > 200: self.log_message_widgets.pop(0).destroy()
        self.after(50, lambda: self.messages_frame._parent_canvas.yview_moveto(1.0))

    def check_log_queue(self):
        import queue
        try:
            while True: self.add_log_message_to_ui(self.log_queue.get_nowait())
        except queue.Empty: pass
        if self.winfo_exists():
            self.after(250, self.check_log_queue)

class CreateChatWindow(BaseTopLevel, DynamicModelUI):
    def __init__(self, master, backend):
        super().__init__(master)
        DynamicModelUI.__init__(self)
        self.master, self.backend = master, backend
        self.title(Lang.get("create_chat_title"))
        self.configure(fg_color=DARK_BG)
        module_manager = ModuleManager()
        self.custom_mods_for_chat = module_manager.get_custom_modules()
        self.newly_added_mods = []
        self.max_tokens = int(self.backend.get_global_settings().get("token_limit", 8192))
        self.validated = True
        self.settings_vars = self._get_default_settings()
        self.original_model_type = self.settings_vars['model_type'].get()
        self.original_connection_string = self.settings_vars['model_provider_params'].get()
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        top_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        customtkinter.CTkLabel(top_frame, text=Lang.get("chat_name")).pack(side='left', padx=(0,10))
        e = customtkinter.CTkEntry(top_frame, textvariable=self.settings_vars['chat_name'], **ENTRY_THEME)
        e.pack(fill='x', expand=True)
        enhance_text_widget(e)
        tabview = customtkinter.CTkTabview(self, **TAB_VIEW_THEME)
        tabview.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        model_tab = tabview.add(Lang.get("tab_model"))
        chat_tab = tabview.add(Lang.get("tab_chat_settings"))
        mods_tab = tabview.add(Lang.get("tab_modules"))
        self.setup_model_tab(model_tab)
        self.setup_chat_tab(chat_tab)
        self.setup_mods_tab(mods_tab)
        bottom_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        bottom_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(10, 10))
        self.create_btn = customtkinter.CTkButton(bottom_frame, text=Lang.get("create"), **BUTTON_THEME, command=self.create_chat_finalize)
        self.create_btn.pack(side='left')
        customtkinter.CTkButton(bottom_frame, text=Lang.get("validate_model"), **BUTTON_THEME, command=self.validate_model).pack(side='left', padx=5)
        customtkinter.CTkButton(bottom_frame, text=Lang.get("cancel"), **BUTTON_THEME, command=self.destroy).pack(side='left', padx=5)
        self._load_provider_params_from_string()

    def _get_default_settings(self):
        settings = self.backend.get_global_settings()  # Из кэша
        s_vars = {key: tk.StringVar(value=val) for key, val in settings.items()}
        s_vars['chat_name'] = tk.StringVar(value=f"{Lang.get('chat_prefix')} {self.backend.generate_id(4)}")
        # Используем кэшированные модули по умолчанию
        module_manager = ModuleManager()
        default_mods = module_manager.get_default_modules()
        s_vars['default_mods'] = { mod['id']: tk.BooleanVar(value=mod['enabled']) for mod in default_mods }
        return s_vars

    def setup_model_tab(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        main_frame = customtkinter.CTkFrame(parent, fg_color="transparent")
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_frame.grid_columnconfigure(0, weight=1)
        self._create_model_ui(main_frame)
    
    def validate_model(self):
        model_type = self.settings_vars['model_type'].get()
        connection_string = self._build_connection_string()
        valid, msg, max_tokens = self.backend.validate_model_settings(model_type, connection_string)
        if valid:
            self.max_tokens = max_tokens
            self.validated = True
            self.original_model_type = model_type
            self.original_connection_string = connection_string
            showinfo(self, Lang.get("success"), msg)
            self.token_label.configure(text=Lang.get("token_limit_info", max_tokens=self.max_tokens))
            self.settings_vars['token_limit'].set(str(self.max_tokens))
        else:
            self.validated = False
            showerror(self, Lang.get("validation_error"), msg)

    def setup_chat_tab(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        frame = customtkinter.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        frame.grid_columnconfigure(1, weight=1)
        # Использовать RAG (по умолчанию вкл)
        self.settings_vars['use_rag'] = tk.StringVar(value=self.backend.get_global_settings().get('use_rag', '1'))
        customtkinter.CTkLabel(frame, text=Lang.get("use_rag")).grid(row=0, column=0, sticky='w', pady=5)
        switch_rag = customtkinter.CTkSwitch(frame, text="", variable=self.settings_vars['use_rag'], 
                               onvalue="1", offvalue="0", 
                               switch_width=50, switch_height=25,
                               progress_color=PURPLE_ACCENT)
        switch_rag.grid(row=0, column=1, sticky='w', padx=10, pady=5)
        # Фильтровать генерации (по умолчанию выкл)
        self.settings_vars['filter_generations'] = tk.StringVar(value=self.backend.get_global_settings().get('filter_generations', '0'))
        customtkinter.CTkLabel(frame, text=Lang.get("filter_generations")).grid(row=1, column=0, sticky='w', pady=5)
        switch_filter = customtkinter.CTkSwitch(frame, text="", variable=self.settings_vars['filter_generations'], 
                               onvalue="1", offvalue="0",
                               switch_width=50, switch_height=25,
                               progress_color=PURPLE_ACCENT)
        switch_filter.grid(row=1, column=1, sticky='w', padx=10, pady=5)
        # Записывать лог (по умолчанию вкл)
        self.settings_vars['write_log'] = tk.StringVar(value=self.backend.get_global_settings().get('write_log', '1'))
        customtkinter.CTkLabel(frame, text=Lang.get("write_log")).grid(row=2, column=0, sticky='w', pady=5)
        switch_write_log = customtkinter.CTkSwitch(frame, text="", variable=self.settings_vars['write_log'], 
                               onvalue="1", offvalue="0",
                               switch_width=50, switch_height=25,
                               progress_color=PURPLE_ACCENT)
        switch_write_log.grid(row=2, column=1, sticky='w', padx=10, pady=5)
        # Максимальное количество задач
        customtkinter.CTkLabel(frame, text=Lang.get("max_tasks")).grid(row=3, column=0, sticky='w', pady=5)
        e_max_tasks = customtkinter.CTkEntry(frame, textvariable=self.settings_vars['max_tasks'], **ENTRY_THEME)
        e_max_tasks.grid(row=3, column=1, sticky='ew', padx=10, pady=5)
        enhance_text_widget(e_max_tasks)
        # Ограничение ступеней иерархии (по умолчанию 0)
        self.settings_vars['hierarchy_limit'] = tk.StringVar(value=self.backend.get_global_settings().get('hierarchy_limit', '0'))
        customtkinter.CTkLabel(frame, text=Lang.get("hierarchy_limit")).grid(row=4, column=0, sticky='w', pady=5)
        e_hierarchy = customtkinter.CTkEntry(frame, textvariable=self.settings_vars['hierarchy_limit'], **ENTRY_THEME)
        e_hierarchy.grid(row=4, column=1, sticky='ew', padx=10, pady=5)
        enhance_text_widget(e_hierarchy)

    def setup_mods_tab(self, parent):
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        self.mods_scrollable_frame = customtkinter.CTkScrollableFrame(parent, fg_color="transparent", scrollbar_button_color=PURPLE_ACCENT, scrollbar_button_hover_color=WHITE)
        if hasattr(self.mods_scrollable_frame, '_scrollbar'): self.mods_scrollable_frame._scrollbar.configure(width=10)
        self.mods_scrollable_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.rebuild_mods_list()
    
    def rebuild_mods_list(self):
        for widget in self.mods_scrollable_frame.winfo_children(): widget.destroy()
        self.mods_scrollable_frame.grid_columnconfigure(0, weight=1)
        # Используем кэшированные модули
        module_manager = ModuleManager()
        default_mods = module_manager.get_default_modules()
        if default_mods:
            customtkinter.CTkLabel(self.mods_scrollable_frame, text=Lang.get("system_modules"), font=("Arial", 14, "bold")).pack(anchor="w", padx=5, pady=(10,5))
            for mod in default_mods: self.create_mod_ui(self.mods_scrollable_frame, mod, "default")
        if self.custom_mods_for_chat:
            customtkinter.CTkLabel(self.mods_scrollable_frame, text=Lang.get("global_custom_modules"), font=("Arial", 14, "bold")).pack(anchor="w", padx=5, pady=(15,5))
            for mod in self.custom_mods_for_chat: self.create_mod_ui(self.mods_scrollable_frame, mod, "global_custom")
        header_frame = customtkinter.CTkFrame(self.mods_scrollable_frame, fg_color="transparent")
        header_frame.pack(fill='x', pady=(15,5))
        customtkinter.CTkLabel(header_frame, text=Lang.get("chat_specific_modules"), font=("Arial", 14, "bold")).pack(side='left', anchor="w", padx=5)
        customtkinter.CTkButton(header_frame, text="+", width=30, **BUTTON_THEME, command=self.add_new_local_mod).pack(side='left', padx=5)
        if self.newly_added_mods:
            for mod in self.newly_added_mods: self.create_mod_ui(self.mods_scrollable_frame, mod, "new_custom")

    def create_mod_ui(self, parent, mod_data, mod_type):
        frame = customtkinter.CTkFrame(parent, border_width=1, border_color=DARK_BORDER, corner_radius=CORNER_RADIUS)
        frame.pack(fill="x", padx=5, pady=3, ipady=5)
        frame.grid_columnconfigure(1, weight=1)
        if mod_type == "default": customtkinter.CTkCheckBox(frame, variable=self.settings_vars['default_mods'][mod_data["id"]], text="", border_color=WHITE, checkmark_color=PURPLE_ACCENT).grid(row=0, column=0, rowspan=2, padx=10)
        info_frame = customtkinter.CTkFrame(frame, fg_color="transparent")
        info_frame.grid(row=0, column=1, rowspan=2, sticky="ew", padx=10)
        customtkinter.CTkLabel(info_frame, text=mod_data["name"], font=("Arial", 12, "bold")).pack(anchor="w")
        desc_label = customtkinter.CTkLabel(info_frame, text=mod_data["description"], text_color=DARK_TEXT_SECONDARY, wraplength=400, justify="left")
        desc_label.pack(anchor="w", fill="x")
        ChatApp.add_label_context_menu(self, desc_label)
        if mod_type != "default":
            btn = customtkinter.CTkButton(frame, text="X", width=25, height=25, fg_color="transparent", hover_color=PURPLE_ACCENT, text_color=WHITE)
            if mod_type == "global_custom": btn.configure(command=lambda mid=mod_data["id"]: self.remove_mod_from_chat_list(mid, self.custom_mods_for_chat))
            elif mod_type == "new_custom": btn.configure(command=lambda mid=mod_data["id"]: self.remove_mod_from_chat_list(mid, self.newly_added_mods))
            btn.grid(row=0, column=2, rowspan=2, padx=10)

    def remove_mod_from_chat_list(self, mod_id_to_remove, mod_list):
        mod_list[:] = [m for m in mod_list if m.get("id") != mod_id_to_remove]
        self.rebuild_mods_list()

    def add_new_local_mod(self):
        from tkinter import filedialog
        path_str = filedialog.askopenfilename(filetypes=[(Lang.get("python_files"), "*.py")])
        if not path_str: return
        path = str(Path(path_str).resolve())
        valid, msg = ModuleValidator.validate_module(path)
        if not valid:
            showerror(self, Lang.get("error"), msg)
            return
        name, description = self.backend._get_localized_doc(Path(path))
        new_mod = {"id": self.backend.generate_id(6), "name": name, "description": description, "adress": path}
        self.newly_added_mods.append(new_mod)
        self.rebuild_mods_list()

    def create_chat_finalize(self):
        chat_name = self.settings_vars['chat_name'].get().strip()
        if not chat_name:
            showerror(self, Lang.get("error"), Lang.get("enter_chat_name"))
            return
        # Блокируем кнопку создания
        self.create_btn.configure(state="disabled")
        current_model_type = self.settings_vars['model_type'].get()
        current_connection_string = self._build_connection_string()
        settings_changed = (current_model_type != self.original_model_type or current_connection_string != self.original_connection_string)
        if not self.validated and settings_changed:
            if not askyesno(self, Lang.get("warning"), Lang.get("model_not_validated_continue")):
                self.create_btn.configure(state="normal")
                return
        model_config = {
            'model_type': self.settings_vars['model_type'].get(),
            'model_provider_params': self._build_connection_string(),
            'token_limit': self.settings_vars['token_limit'].get()
        }
        chat_config = {
            "max_tasks": self.settings_vars['max_tasks'].get(), 
            "language": Lang.current_language,
            "use_rag": self.settings_vars['use_rag'].get(),
            "filter_generations": self.settings_vars['filter_generations'].get(),
            "hierarchy_limit": self.settings_vars['hierarchy_limit'].get(),
            "write_log": self.settings_vars['write_log'].get()
        }
        # Используем кэшированные модули для создания чата
        module_manager = ModuleManager()
        default_mods = module_manager.get_default_modules()
        default_mods_config = {mid: var.get() for mid, var in self.settings_vars['default_mods'].items()}
        final_custom_mods = self.custom_mods_for_chat + self.newly_added_mods
        settings_bundle = {
            "model_config": model_config, "chat_config": chat_config, 
            "default_mods_config": default_mods_config, "custom_mods_list": final_custom_mods
        }
        # Сначала закрываем окно
        self.on_close()
        # Создаем чат (это обновит кэш)
        chat_data = self.backend.create_chat(chat_name, settings_bundle)
        if not chat_data:
            showerror(self.master, Lang.get("error"), Lang.get("chat_name_exists"))
            # Разблокируем кнопку создания чата в главном окне
            if self.master.winfo_exists() and hasattr(self.master, 'new_chat_btn'): self.master.new_chat_btn.configure(state="normal")
            return
        self.master.load_chats()
        self.master.on_chat_select(chat_data["id"])
        if self.master.winfo_exists() and hasattr(self.master, 'send_btn'): self.master.send_btn.configure(state="disabled")

def setup_icon(root: tk.Tk):
    base = Path(__file__).resolve().parent
    png_path = str(base / "data" / "icons" / "icon.png")
    ico_path = str(base / "data" / "icons" / "icon.ico")
    if not Path(png_path).exists(): return
    try:
        img = tk.PhotoImage(file=png_path)
        root.iconphoto(True, img)
    except Exception as e: print(f"setup_icon: iconphoto(png) failed:", e)
    if sys.platform.startswith("win") and Path(ico_path).exists():
        try: root.iconbitmap(default=ico_path)
        except Exception as e: print("setup_icon: iconbitmap(ico) failed:", e)
    elif sys.platform == "darwin":
        try:
            from AppKit import NSApp, NSImage
            img = NSImage.alloc().initWithContentsOfFile_(png_path)
            if img: NSApp.setApplicationIconImage_(img)
        except ImportError: pass

def send_log_to_ui(log_queue, message: str):
    if log_queue:
        try: log_queue.put(message)
        except Exception as e: print(f"Failed to send log to UI: {e}")

def show_splash(app_ready_event: multiprocessing.Event):
    if not sys.platform.startswith("win32"): return
    icon_path = resource_path(os.path.join("data", "icons", "icon.png"))
    if not os.path.exists(icon_path):
        print(f"Иконка для сплэша не найдена: {icon_path}")
        return
    root = tk.Tk()
    root.withdraw()
    splash = tk.Toplevel(root)
    splash.overrideredirect(True)
    splash.configure(bg='black')
    sw, sh = 800, 600
    try:
        img = Image.open(icon_path)
        # Получаем реальные размеры экрана
        sw = splash.winfo_screenwidth()
        sh = splash.winfo_screenheight()
        max_ratio = 0.3
        max_size = int(min(sw, sh) * max_ratio)
        ratio = min(max_size / img.width, max_size / img.height)
        if ratio < 1: img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        w, h = img_tk.width(), img_tk.height()
        label = tk.Label(splash, image=img_tk, bg='black')
        label.image = img_tk
    except Exception as e:
        print(f"Splash image load failed: {e}")
        w, h = 400, 300
        label = tk.Label(splash, text="Loading...", font=("Arial", 24), bg='black', fg='white')
    x, y = (sw - w) // 2, (sh - h) // 2
    splash.geometry(f"{w}x{h}+{x}+{y}")
    label.pack()
    splash.attributes('-topmost', True)
    # Устанавливаем прозрачный цвет только на Windows
    if sys.platform == "win32": splash.attributes('-transparentcolor', 'black')
    def poll():
        if app_ready_event.is_set(): splash.after(500, lambda: (splash.destroy(), root.quit()))
        else: splash.after(50, poll)
    splash.after(50, poll)
    root.mainloop()

def run_main_app(app_ready_event: multiprocessing.Event):
    from tkinter import filedialog
    import sqlite3, os, random, string, queue, subprocess, platform, shutil, importlib
    from cross_gpt import initialize_work
    if sys.platform == "win32": import ctypes
    elif sys.platform == "darwin":
        try: from AppKit import NSApp
        except ImportError:
            try:
                print("PyObjC not found, attempting to install...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyobjc"])
                from AppKit import NSApp
                print("PyObjC installed successfully.")
            except Exception as e:
                root = tk.Tk()
                root.withdraw()
                showerror(root, "Ошибка установки", f"Не удалось установить или импортировать pyobjc:\n{e}\n\nУстановите его вручную:\npip install pyobjc")
                root.destroy()
                app_ready_event.set()
                sys.exit(1)
    global Lang, DARK_BG, DARK_ENTRY_BG, DARK_SECONDARY, DARK_BORDER, PURPLE_ACCENT, ACTIVE_CHAT_COLOR, WHITE, DARK_TEXT_SECONDARY, CORNER_RADIUS, BUTTON_THEME, ENTRY_THEME, TAB_VIEW_THEME, OPTIONMENU_THEME
    Lang = LanguageManager()
    DARK_BG, DARK_ENTRY_BG, DARK_SECONDARY, DARK_BORDER, PURPLE_ACCENT = "#000000", "#1e1e1e", "#2a2a2a", "#333333", "#5200ff"
    ACTIVE_CHAT_COLOR, WHITE, DARK_TEXT_SECONDARY, CORNER_RADIUS = "#ff9900", "#ffffff", "#b0b0b0", 12
    BUTTON_THEME = {"fg_color": DARK_SECONDARY, "border_color": WHITE, "border_width": 1, "hover_color": PURPLE_ACCENT, "corner_radius": CORNER_RADIUS}
    ENTRY_THEME = {"fg_color": DARK_ENTRY_BG, "border_color": PURPLE_ACCENT, "border_width": 1, "corner_radius": CORNER_RADIUS}
    TAB_VIEW_THEME = {"segmented_button_selected_color": PURPLE_ACCENT, "segmented_button_unselected_color": DARK_SECONDARY, "segmented_button_selected_hover_color": PURPLE_ACCENT, "fg_color": DARK_BG}
    OPTIONMENU_THEME = {"fg_color": DARK_SECONDARY, "button_color": DARK_SECONDARY, "button_hover_color": PURPLE_ACCENT, "dropdown_fg_color": DARK_SECONDARY, "dropdown_hover_color": PURPLE_ACCENT, "corner_radius": CORNER_RADIUS}
    customtkinter.set_appearance_mode("dark")
    selected_language = "en"
    # !!! ИСПРАВЛЕНИЕ: Используем resource_path для пути к БД !!!
    db_path = resource_path(os.path.join("data", "settings.db"))
    is_initial_run = not os.path.exists(db_path)
    if not is_initial_run:
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM settings WHERE key = 'language'")
            result = cursor.fetchone()
            if result: selected_language = result[0]
            conn.close()
        except sqlite3.OperationalError:
            pass
        except Exception as e:
            print(f"Could not read language from DB: {e}")
    if not Lang.load_language(selected_language):
        root = tk.Tk()
        root.withdraw()
        showerror(root, Lang.get("lang_load_error_title"), Lang.get("lang_load_error_message"))
        root.destroy()
        app_ready_event.set()
        sys.exit(1)
    try:
        backend = Backend()
        app = ChatApp(backend)
        setup_icon(app)
        app.after(0, app.bring_to_front)
        app_ready_event.set()
        app.mainloop()
    except Exception as e:
        import traceback
        root_err = tk.Tk()
        root_err.withdraw()
        showerror(root_err, "Критическая ошибка", f"Произошла непредвиденная ошибка:\n\n{e}\n\n{traceback.format_exc()}")
        root_err.destroy()
        app_ready_event.set()
        if 'app' in locals() and 'destroy' in dir(app) and app.winfo_exists(): app.destroy()
        sys.exit(1)

# Помогаем multiprocessing найти функции в динамически загруженном файле
if getattr(sys, 'frozen', False):
    # Находим, как называется этот файл
    import __main__
    # Если мы запущены через лаунчер, подменяем __main__ на текущий модуль
    # чтобы pickle мог найти функцию run_main_app
    for attr in ['run_main_app', 'show_splash_screen']: # перечислите функции, которые запускаете в Process
        if hasattr(sys.modules[__name__], attr): setattr(__main__, attr, getattr(sys.modules[__name__], attr))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app_is_ready_event = multiprocessing.Event()
    import ui
    main_app_process = multiprocessing.Process(target=ui.run_main_app, args=(app_is_ready_event,))
    main_app_process.start()
    show_splash(app_is_ready_event)
    sys.exit(0)