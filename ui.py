import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from pathlib import Path
import multiprocessing
import sys
import customtkinter
import inspect
import importlib.util
import ast
import os
import sqlite3
import json
import platform
import shutil
import subprocess
import queue
import random
import string
from contextlib import redirect_stdout
import io
from customtkinter import CTkButton, CTkEntry, CTkFrame, CTkLabel, CTkScrollableFrame, CTkTabview, CTkRadioButton, CTkSwitch, CTkOptionMenu, CTkTextbox, CTkCheckBox, CTkToplevel, CTk

initialize_work = None

# ====== БАЗОВЫЕ ПУТИ ======
def get_base_dir():
    """Возвращает абсолютный путь к каталогу, где находится исполняемый файл (ui.exe) или файл скрипта (ui.py)."""
    if getattr(sys, 'frozen', False): return os.path.dirname(os.path.abspath(sys.executable))
    else: return os.path.dirname(os.path.abspath(__file__))
def resource_path(relative_path): return os.path.join(get_base_dir(), relative_path)
BASE_DIR = get_base_dir()
if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)
PROVIDER_DIR = resource_path("providers")
if os.path.exists(PROVIDER_DIR) and PROVIDER_DIR not in sys.path: sys.path.append(PROVIDER_DIR)
def get_ast_value(node):
    """Безопасное получение значения из AST узла"""
    if isinstance(node, ast.Constant): return node.value
    elif isinstance(node, ast.Str): return node.s
    elif isinstance(node, ast.Num): return node.n
    elif isinstance(node, ast.NameConstant): return node.value
    else: return None

# ====== КОНСТАНТЫ ======
DARK_BG = "#000000"
DARK_ENTRY_BG = "#1e1e1e"
DARK_SECONDARY = "#2a2a2a"
DARK_BORDER = "#333333"
PURPLE_ACCENT = "#5200ff"
ACTIVE_CHAT_COLOR = "#ff9900"
WHITE = "#ffffff"
DARK_TEXT_SECONDARY = "#b0b0b0"
CORNER_RADIUS = 12
FONT_FAMILY = "Georgia"
FONT_BOLD = (FONT_FAMILY, 12, "bold")
FONT_REGULAR = (FONT_FAMILY, 12)
BUTTON_THEME = {"fg_color": DARK_SECONDARY, "border_color": WHITE, "border_width": 1, "hover_color": PURPLE_ACCENT, "corner_radius": CORNER_RADIUS, "font": FONT_BOLD}
ENTRY_THEME = {"fg_color": DARK_ENTRY_BG, "border_color": PURPLE_ACCENT, "border_width": 1, "corner_radius": CORNER_RADIUS, "font": FONT_REGULAR}
TAB_VIEW_THEME = {"segmented_button_selected_color": PURPLE_ACCENT, "segmented_button_unselected_color": DARK_SECONDARY, "segmented_button_selected_hover_color": PURPLE_ACCENT, "fg_color": DARK_BG}
OPTIONMENU_THEME = {"fg_color": DARK_SECONDARY, "button_color": DARK_SECONDARY, "button_hover_color": PURPLE_ACCENT, "dropdown_fg_color": DARK_SECONDARY, "dropdown_hover_color": PURPLE_ACCENT, "corner_radius": CORNER_RADIUS, "font": FONT_REGULAR}

# ====== ВЫНЕСЕННЫЕ ФУНКЦИИ ======
def create_styled_button(parent, text, command=None, width=None, height=None, **kwargs):
    default_kwargs = BUTTON_THEME.copy()
    if width: default_kwargs["width"] = width
    if height: default_kwargs["height"] = height
    default_kwargs.update(kwargs)
    return CTkButton(parent, text=text, command=command, **default_kwargs)
def create_styled_entry(parent, textvariable=None, **kwargs):
    default_kwargs = ENTRY_THEME.copy()
    if textvariable: default_kwargs["textvariable"] = textvariable
    default_kwargs.update(kwargs)
    entry = CTkEntry(parent, **default_kwargs)
    enhance_text_widget(entry)
    return entry
def create_styled_frame(parent, fg_color="transparent", **kwargs): return CTkFrame(parent, fg_color=fg_color, **kwargs)
def create_styled_label(parent, text, **kwargs):
    default_kwargs = {"font": FONT_BOLD}
    default_kwargs.update(kwargs)
    # По умолчанию делаем прозрачный фон
    if "fg_color" not in default_kwargs: default_kwargs["fg_color"] = "transparent"
    # Убираем минимальную высоту, если не указана явно
    if "height" not in default_kwargs: default_kwargs["height"] = 0
    return CTkLabel(parent, text=text, **default_kwargs)
def create_param_widget(parent, param_info, settings_vars_dict, path_vars_dict):
    param_name = param_info['name']
    default_val = param_info.get('default')
    is_file = param_info['is_file']
    param_frame = create_styled_frame(parent)
    param_frame.pack(fill="x", pady=2, padx=5)  # ДОБАВЛЕН padx=5
    param_frame.grid_columnconfigure(1, weight=1)
    label_text = param_name
    if default_val is not None and str(default_val).strip() != '': label_text += f" {default_val}"
    create_styled_label(param_frame, label_text).grid(row=0, column=0, sticky="w", padx=(0, 10))
    input_frame = create_styled_frame(param_frame)
    input_frame.grid(row=0, column=1, sticky="ew")
    input_frame.grid_columnconfigure(0, weight=1)
    if is_file:
        path_vars_dict[param_name] = tk.StringVar()
        display_var = tk.StringVar()
        entry = create_styled_entry(input_frame, textvariable=display_var)
        entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        browse_cmd = lambda e=entry, dv=display_var, fpv=path_vars_dict[param_name]: \
            browse_file_dialog(e, dv, fpv, [(Lang.get("all_files"), "*.*")])
        create_styled_button(input_frame, text=Lang.get("browse"), width=80, command=browse_cmd).grid(row=0, column=1)
    else:
        settings_vars_dict[param_name] = tk.StringVar(value='')
        entry = create_styled_entry(input_frame, textvariable=settings_vars_dict[param_name])
        entry.grid(row=0, column=0, sticky="ew")
    return param_frame
def create_module_ui_item(parent, module_data, module_type, enabled_var=None, on_toggle=None, on_remove=None, show_checkbox=True): # Создает элемент UI для модуля системного или кастомного
    frame = create_styled_frame(parent, border_width=1, border_color=DARK_BORDER, corner_radius=CORNER_RADIUS)
    frame.pack(fill="x", padx=5, pady=3, ipady=5)
    frame.grid_columnconfigure(1, weight=1)
    if show_checkbox and enabled_var is not None:
        cb = CTkCheckBox(frame, variable=enabled_var, text="", border_color=WHITE, checkmark_color=WHITE, fg_color=(DARK_SECONDARY, PURPLE_ACCENT), command=lambda: on_toggle() if on_toggle else None)
        cb.grid(row=0, column=0, rowspan=2, padx=10)
    info_frame = create_styled_frame(frame)
    info_frame.grid(row=0, column=1, rowspan=2, sticky="ew", padx=10)
    create_styled_label(info_frame, module_data["name"], font=FONT_BOLD).pack(anchor="w")
    desc_label = create_styled_label(info_frame, text=module_data["description"], text_color=DARK_TEXT_SECONDARY, wraplength=400, justify="left")
    desc_label.pack(anchor="w", fill="x")
    if hasattr(parent, 'master') and hasattr(parent.master, 'add_label_context_menu'): parent.master.add_label_context_menu(parent.master, desc_label)
    elif hasattr(parent.winfo_toplevel(), 'add_label_context_menu'): parent.winfo_toplevel().add_label_context_menu(parent.winfo_toplevel(), desc_label)
    if on_remove:
        remove_btn = CTkButton(frame, text="X", width=25, height=25, fg_color="transparent", hover_color=PURPLE_ACCENT, text_color=WHITE, command=on_remove)
        remove_btn.grid(row=0, column=2, rowspan=2, padx=10)
    return frame
def create_chat_message_bubble(parent, text, is_my, attachments=None, is_question=False):
    """Создает пузырь сообщения чата"""
    row_frame = create_styled_frame(parent)
    row_frame.pack(fill=tk.X, pady=2, padx=10, anchor="center")
    # Основной пузырь с рамкой
    bubble = create_styled_frame(row_frame, border_width=2, border_color=PURPLE_ACCENT if is_my else WHITE, corner_radius=CORNER_RADIUS, fg_color=DARK_BG)
    bubble.pack(expand=False, anchor="center")
    # Текст сообщения (wraplength будет установлен динамически)
    msg_text_widget = CTkLabel(
        bubble, 
        text=text, 
        justify="left",
        anchor="w",
        fg_color="transparent",
        text_color=WHITE,
        font=FONT_REGULAR,
        height=0)
    # Настраиваем отступы
    if is_question and not is_my: msg_text_widget.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 8), pady=6)
    else: msg_text_widget.pack(fill=tk.X, expand=True, padx=8, pady=6)
    def update_message_width():
        try:
            if not msg_text_widget.winfo_exists(): return
            max_parent_width = parent.winfo_width() - 40
            bubble_width = min(
                max_parent_width * 0.95,
                msg_text_widget.winfo_reqwidth() + 0)
            # 3. Устанавливаем wraplength как ширину пузыря минус отступы
            wraplength_value = max(20, bubble_width - 20)  # Минимум 20px
            msg_text_widget.configure(wraplength=wraplength_value)
        except Exception: pass
    # Обновляем при создании и при изменении размера
    bubble.after(100, update_message_width)
    parent.bind("<Configure>", lambda e: bubble.after(50, update_message_width))
    return bubble, msg_text_widget
def setup_message_wraplength(widget, messages_frame):
    """Настраивает автоматическое обновление wraplength для сообщений"""
    def update_wraplength(event=None):
        try:
            if widget.winfo_exists():
                available_width = messages_frame.winfo_width() * 0.905
                if available_width > 50: widget.configure(wraplength=available_width)
        except Exception: pass
    messages_frame.bind("<Configure>", update_wraplength)
    widget.after(100, update_wraplength)
    return update_wraplength
def setup_window_geometry(window, width=600, height=500): # Настраивает геометрию окна и центрирует его
    window.geometry(f"{width}x{height}")
    window.minsize(width, height)
    window.after(10, lambda: set_windows_dark_titlebar(window))
    window.after_idle(lambda: center_window(window))
    return window
def center_window(window): # Центрирует окно на экране
    window.update_idletasks()
    try:
        width, height = window.winfo_width(), window.winfo_height()
        x = (window.winfo_screenwidth() // 2) - (width // 2)
        y = (window.winfo_screenheight() // 2) - (height // 2)
        window.geometry(f'{width}x{height}+{x}+{y}')
    except tk.TclError: pass
def create_tabbed_interface(parent, tabs_config):
    """Создает интерфейс с вкладками"""
    tabview = CTkTabview(parent, **TAB_VIEW_THEME)
    tabview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    tabs = {}
    for tab_name, setup_func in tabs_config.items():
        tab = tabview.add(Lang.get(tab_name))
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        setup_func(tab)
        tabs[tab_name] = tab
    return tabview, tabs
def load_settings_from_backend(backend, additional_settings=None): # Загружает настройки из бэкенда
    settings = backend.get_global_settings()
    if additional_settings: settings.update(additional_settings)
    settings_vars = {}
    for key, val in settings.items(): settings_vars[key] = tk.StringVar(value=val)
    return settings_vars
def save_settings_to_backend(backend, settings_vars, keys_to_save=None): #Сохраняет настройки в бэкенд
    if keys_to_save is None: keys_to_save = settings_vars.keys()
    settings_to_save = {}
    for key in keys_to_save:
        if key in settings_vars: settings_to_save[key] = settings_vars[key].get()
    backend.update_global_settings(settings_to_save)
    return True

# ====== ОБЩИЕ ФУНКЦИИ ======
def create_scrollable_frame(parent, **kwargs):
    """Создает скроллируемый фрейм с единым стилем скроллбара"""
    scroll_frame = CTkScrollableFrame(parent, 
        scrollbar_button_color=PURPLE_ACCENT,
        scrollbar_button_hover_color=WHITE,
        **kwargs)
    if hasattr(scroll_frame, '_scrollbar'): 
        scroll_frame._scrollbar.configure(width=10)
    return scroll_frame
def browse_file_dialog(entry_widget, entry_var, full_path_var, filetypes=None):
    """Общая функция для выбора файлов"""
    if filetypes is None: 
        filetypes = [("All files", "*.*")]
    path = filedialog.askopenfilename(filetypes=filetypes)
    if path:
        full_path_var.set(path)
        entry_var.set(Path(path).name)
        entry_widget.delete(0, "end")
        entry_widget.insert(0, Path(path).name)
def show_message_dialog(parent, title, message, buttons):
    """Общая функция для показа диалоговых окон"""
    dialog = CustomMessageBox(parent, title, message, buttons)
    parent.wait_window(dialog)
    return dialog.result
def showinfo(parent, title, message):
    """Показать информационное сообщение"""
    return show_message_dialog(parent, title, message, [("OK", True)])
def showerror(parent, title, message):
    """Показать сообщение об ошибке"""
    return show_message_dialog(parent, title, message, [("OK", True)])
def showwarning(parent, title, message):
    """Показать предупреждение"""
    return show_message_dialog(parent, title, message, [("OK", True)])
def askyesno(parent, title, message):
    """Задать вопрос с выбором Да/Нет"""
    return show_message_dialog(parent, title, message, [("Yes", True), ("No", False)])
def enhance_text_widget(widget):
    """Добавляет контекстное меню и горячие клавиши к текстовым виджетам"""
    is_textbox = isinstance(widget, CTkTextbox)
    is_entry = isinstance(widget, CTkEntry)
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
def set_windows_dark_titlebar(window):
    """Устанавливает темную тему заголовка окна на Windows"""
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
def setup_icon(root: tk.Tk):
    """Устанавливает иконку приложения"""
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

# ====== КЛАССЫ ======
class AppCache:
    _instance = None
    def __new__(cls):
        if not cls._instance: cls._instance = super(AppCache, cls).__new__(cls)
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
                except Exception as e: print(f"Failed to load language {code}: {e}"); continue
        if not loaded: self.texts = {"lang_load_error_title": "Language Error", "lang_load_error_message": "Could not load any language files. Please ensure 'lang/en' directory exists."}
        return loaded
    def get(self, key, **kwargs):
        defaults = {
            "ok": "OK", "cancel": "Cancel", "yes": "Yes", "no": "No",
            "cut": "Cut", "copy": "Copy", "paste": "Paste", "select_all": "Select All",
            "undo": "Undo", "redo": "Redo"}
        if key in defaults and key not in self.texts: return defaults[key]
        template = self.texts.get(key, f"[{key.upper()}]")
        return template.format(**kwargs)

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
                for node in tree.body)
            if not main_found: return False, Lang.get("module_err_main_not_found")
            return True, Lang.get("module_validated")
        except SyntaxError as e: return False, Lang.get("module_err_syntax", e=e)
        except Exception as e: return False, Lang.get("module_err_generic", e=e)

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
        if not provider_dir.is_dir(): print("Warning: 'model_providers' directory not found."); return
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
                    "params": params}
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
            found_funcs = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
            has_token_limit = False
            has_params_in_connect = False
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if target.id == 'token_limit': has_token_limit = True
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == "connect":
                    for stmt in node.body:
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if isinstance(target, ast.Name) and target.id == 'params':
                                    has_params_in_connect = True
                                    break
            return required_funcs.issubset(found_funcs), found_funcs, has_token_limit, has_params_in_connect
        except Exception as e: print(f"Error validating provider {path}: {e}"); return False, set(), False, False
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
                                            if key_name is None: continue
                                            default_value = get_ast_value(value)
                                            param_data = {
                                                'name': key_name,
                                                'default': default_value,
                                                'is_file': 'path' in key_name.lower() or 
                                                        'file' in key_name.lower() or 
                                                        'dir' in key_name.lower()}
                                            params_info.append(param_data)
            return params_info
        except Exception as e: print(f"Could not parse params from connect function for {path.name}: {e}"); return []
    def get_providers(self): return self.providers

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
            self.loaded = False
    def load_modules(self, backend, reload_m=False):
        if self.loaded and not reload_m:
            print("Modules already loaded, skipping")
            return
        try:
            self.default_modules = backend.get_default_mods()
            self.custom_modules = backend.get_custom_mods()
            self.loaded = True
            print(f"Loaded {len(self.default_modules)} default modules and {len(self.custom_modules)} custom modules to cache")
        except Exception as e:
            print(f"Error loading modules: {e}")
            self.default_modules = []
            self.custom_modules = []
    def get_default_modules(self): return self.default_modules
    def get_custom_modules(self): return self.custom_modules
    def update_custom_modules(self, backend): self.custom_modules = backend.get_custom_mods() # Обновляет только кастомные модули (при добавлении/удалении)

class Backend:
    def __init__(self):
        self.db_path = resource_path(os.path.join("data", "settings.db"))
        self.cache = AppCache()
        self.init_settings_db()
    def sql_exec(self, db_path, query, params=(), fetchone=False, fetchall=False, commit=True):
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
    # ИСПРАВЛЕНО: добавлен параметр lang, используется переданный язык или глобальный
    def _get_localized_doc(self, mod_path: Path, lang=None):
        localized_name, localized_desc = None, None
        if lang is None:
            lang = Lang.current_language
        lang_file = mod_path.with_name(f"{mod_path.stem}_lang.py")
        if lang_file.exists() and lang and lang != "en":
            try:
                spec = importlib.util.spec_from_file_location("lang_module", str(lang_file))
                lang_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(lang_module)
                if hasattr(lang_module, 'locales') and lang in lang_module.locales:
                    locale_data = lang_module.locales[lang]
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
        system_os = platform.system().lower()
        # Используем текущий язык интерфейса для всех операций
        current_language = Lang.current_language
        self._check_existing_modules(db_path, current_language)
        self._scan_default_tools(db_path, system_os, current_language)
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
                new_name, new_desc = self._get_localized_doc(mod_path, lang=current_language)
                self.sql_exec(db_path, "UPDATE default_mods SET name = ?, description = ?, lang = ? WHERE id = ?", (new_name, new_desc, current_language, mod_id))
        current_customs = self.sql_exec(db_path, "SELECT id, adress, lang FROM custom_mods", fetchall=True) or []
        for mod_id, mod_adress, mod_lang in current_customs:
            mod_path = Path(mod_adress)
            if not mod_path.exists():
                self.sql_exec(db_path, "DELETE FROM custom_mods WHERE id = ?", (mod_id,))
                continue
            if mod_lang != current_language:
                new_name, new_desc = self._get_localized_doc(mod_path, lang=current_language)
                self.sql_exec(db_path, "UPDATE custom_mods SET name = ?, description = ?, lang = ? WHERE id = ?", (new_name, new_desc, current_language, mod_id))
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
            if not valid: print(f"Ошибка в модуле по умолчанию {mod_file.name}: {msg}"); return
            name, description = self._get_localized_doc(mod_file, lang=current_language)
            self.sql_exec(db_path, "INSERT OR IGNORE INTO default_mods (name, description, adress, enabled, lang) VALUES (?, ?, ?, ?, ?)", (name, description, relative_path_str, 0, current_language))
        for item in default_mods_dir.rglob("*.py"):
            if item.is_file():
                relative_path = item.relative_to(default_mods_dir)
                process_mod_file(item, str(relative_path))
    def init_settings_db(self):
        Path("data").mkdir(exist_ok=True)
        db_path = self.db_path
        self.sql_exec(db_path, "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
        self.sql_exec(db_path, """CREATE TABLE IF NOT EXISTS default_mods (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, description TEXT, adress TEXT UNIQUE, enabled INTEGER DEFAULT 0, lang TEXT DEFAULT 'en')""")
        self.sql_exec(db_path, """CREATE TABLE IF NOT EXISTS custom_mods (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, description TEXT, adress TEXT UNIQUE, enabled INTEGER DEFAULT 1, lang TEXT DEFAULT 'en')""")
        self.sql_exec(db_path, "INSERT OR IGNORE INTO settings (key, value) VALUES ('language', 'en')")
        providers = ProviderManager().get_providers()
        default_provider = list(providers.keys())[0] if providers else ""
        defaults = {
            "token_limit": "8192", "model_provider_params": "",
            "model_type": default_provider, "use_rag": "1", 
            "filter_generations": "0", "hierarchy_limit": "0",
            "write_log": "1", "write_results": "0", "max_critic_reactions": "2"}
        for key, value in defaults.items(): self.sql_exec(db_path, "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (key, value))
    def generate_id(self, length=12): return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    def _load_chats_from_db(self):
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
    def get_chats(self): return self.cache.get_chats(self)
    def _load_global_settings_from_db(self):
        rows = self.sql_exec(self.db_path, "SELECT key, value FROM settings", fetchall=True) or []
        return {k: v for k, v in rows}
    def get_global_settings(self): return self.cache.get_global_settings(self)
    def update_global_settings(self, settings):
        for key, value in settings.items(): self.sql_exec(self.db_path, "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
        self.cache.update_global_settings({**self.cache.get_global_settings(self), **settings})
        return True
    def create_chat(self, chat_name, settings_data):
        existing_chats = self.cache.get_chats(self)
        if any(chat['name'].lower() == chat_name.lower() for chat in existing_chats):  return None
        chat_id = self.generate_id()
        while (Path(resource_path(os.path.join("data", "chats"))) / chat_id).exists():
            chat_id = self.generate_id()
        chat_path = Path(resource_path(os.path.join("data", "chats"))) / chat_id
        chat_path.mkdir(parents=True, exist_ok=True)
        new_chat = {"id": chat_id, "name": chat_name}
        updated_chats = [new_chat] + existing_chats
        self.cache.update_chats(updated_chats)
        default_mods = ModuleManager().get_default_modules()
        create_file_mod = next((mod for mod in default_mods if mod['adress'] == 'create_file.py'), None)
        if create_file_mod:
            mod_enabled = settings_data.get('default_mods_config', {}).get(create_file_mod['id'], False)
            if mod_enabled: (chat_path / "files").mkdir(exist_ok=True)
        cmd_mod = next((mod for mod in default_mods if mod['adress'].endswith('cmd.py')), None)
        if cmd_mod:
            mod_enabled = settings_data.get('default_mods_config', {}).get(cmd_mod['id'], False)
            if mod_enabled: (chat_path / "console_folders").mkdir(exist_ok=True)
        create_report_mod = next((mod for mod in default_mods if mod['adress'] == 'create_report.py'), None)
        if create_report_mod:
            mod_enabled = settings_data.get('default_mods_config', {}).get(create_report_mod['id'], False)
            if mod_enabled: (chat_path / "reports").mkdir(exist_ok=True)
        write_results = settings_data.get('chat_config', {}).get('write_results', '0') == '1'
        if write_results: (chat_path / "results").mkdir(exist_ok=True)
        settings_db = str(chat_path / "chatsettings.db")
        self.sql_exec(settings_db, "PRAGMA max_page_count = 2147483647")
        self.sql_exec(settings_db, "CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT)")
        self.sql_exec(settings_db, "CREATE TABLE default_mods (id INTEGER PRIMARY KEY, name TEXT, adress TEXT, enabled INTEGER)")
        self.sql_exec(settings_db, "CREATE TABLE custom_mods (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, description TEXT, adress TEXT UNIQUE)")
        all_settings = {**settings_data.get('model_config', {}), **settings_data.get('chat_config', {})}
        all_settings['chat_name'] = chat_name
        for key, value in all_settings.items(): self.sql_exec(settings_db, "INSERT INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
        default_mods = ModuleManager().get_default_modules()
        enabled_defaults = settings_data.get('default_mods_config', {})
        for mod in default_mods: self.sql_exec(settings_db, "INSERT INTO default_mods (id, name, adress, enabled) VALUES (?, ?, ?, ?)", (mod['id'], mod['name'], mod['adress'], 1 if enabled_defaults.get(mod['id'], False) else 0))
        for mod in settings_data.get('custom_mods_list', []): self.sql_exec(settings_db, "INSERT INTO custom_mods (name, description, adress) VALUES (?, ?, ?)", (mod['name'], mod['description'], mod['adress']))
        dialog_db = str(chat_path / "chatsettings.db")
        self.sql_exec(dialog_db, "CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, is_my INTEGER, attachments TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
        return new_chat
    def delete_chat(self, chat_id):
        chat_path = Path(resource_path(os.path.join("data", "chats"))) / chat_id
        if not chat_path.exists(): return False
        shutil.rmtree(chat_path)
        existing_chats = self.cache.get_chats(self)
        updated_chats = [chat for chat in existing_chats if chat['id'] != chat_id]
        self.cache.update_chats(updated_chats)
        return True
    def get_messages(self, chat_id):
        db = Path(resource_path(os.path.join("data", "chats", chat_id, "chatsettings.db")))
        if not db.exists(): return []
        rows = self.sql_exec(str(db), "SELECT text, is_my, attachments FROM messages ORDER BY timestamp", fetchall=True) or []
        return [{"text": r[0], "isMy": bool(r[1]), "attachments": json.loads(r[2]) if r[2] else []} for r in rows]
    def add_message(self, chat_id, text, is_my, attachments=None):
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
        name, description = self._get_localized_doc(Path(file_path), lang=Lang.current_language)
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
            f = io.StringIO()
            with redirect_stdout(f): valid, tokens, _ = provider_module.connect(connection_string)
            if hasattr(provider_module, 'disconnect'): provider_module.disconnect()
            if valid: return True, Lang.get("model_validated_success", tokens=tokens), tokens
            else: return False, Lang.get("custom_api_fail"), max_tokens
        except ImportError as e: print(e); return False, Lang.get("model_err_provider_missing", provider=model_type), max_tokens
        except Exception as e: return False, Lang.get("model_err_validation_generic", e=str(e)), max_tokens

# ====== БАЗОВОЕ ОКНО ======
class BaseTopLevel(CTkToplevel):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        setup_window_geometry(self)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after_idle(self.setup_and_center)
    def setup_and_center(self):
        setup_icon(self)
        self.lift()
        self.grab_set()
    def on_close(self):
        self.grab_release()
        self.destroy()

# ====== ДИНАМИЧЕСКИЙ UI МОДЕЛИ ======
class DynamicModelUI:
    def __init__(self):
        self.provider_param_full_paths = {}
        self.model_frames = {}
        self.provider_manager = ProviderManager()
        self.providers = self.provider_manager.get_providers()
    def _create_model_ui(self, parent):
        model_type_frame = create_styled_frame(parent)
        model_type_frame.pack(fill="x", pady=10)
        create_styled_label(model_type_frame, text=Lang.get("model_type")).pack(side="top")
        radio_frame = create_styled_frame(parent)
        radio_frame.pack(fill="x")
        provider_items = list(self.providers.items())
        if not provider_items: create_styled_label(radio_frame, text=Lang.get("no_providers_found")).pack(); return
        for i, (module_name, data) in enumerate(provider_items):
            display_name = data['name']
            CTkRadioButton(radio_frame, text=display_name, variable=self.settings_vars['model_type'], value=module_name, command=self.toggle_model_frames, fg_color=PURPLE_ACCENT, font=FONT_REGULAR).grid(row=i // 4, column=i % 4, sticky="w", padx=5, pady=2)
        self.frames_container = create_styled_frame(parent)
        self.frames_container.pack(fill="x", expand=True, pady=10)
        self.model_frames = {}
        self._create_specific_model_frames(self.frames_container)
        token_frame = create_styled_frame(parent)
        token_frame.pack(fill="x", pady=5)
        self.token_label = create_styled_label(token_frame, text=Lang.get("token_limit"))
        self.token_label.pack(side="left", padx=(0,10))
        self.token_entry = create_styled_entry(token_frame, textvariable=self.settings_vars['token_limit'])
        self.token_entry.pack(side="left", fill="x", expand=True)
        if provider_items and not self.settings_vars['model_type'].get(): self.settings_vars['model_type'].set(provider_items[0][0])
        self.toggle_model_frames()
    def _create_specific_model_frames(self, container):
        try:
            for module_name, p_data in self.providers.items():
                main_frame = create_styled_frame(container, border_color=WHITE, border_width=1)
                self.model_frames[module_name] = main_frame
                params = p_data.get('params', [])
                has_scroll = len(params) > 6
                content_parent = main_frame
                if has_scroll:
                    scroll_frame = create_scrollable_frame(main_frame, fg_color=DARK_BG)
                    scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)
                    content_parent = scroll_frame
                else: main_frame.pack(padx=5, pady=5)
                self.provider_param_full_paths.setdefault(module_name, {})
                self.settings_vars.setdefault(module_name, {})
                for param in params: create_param_widget(content_parent, param, self.settings_vars[module_name], self.provider_param_full_paths[module_name])
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
        try:
            if hasattr(self, 'save_btn') and self.save_btn.winfo_exists(): self.save_btn.configure(state="disabled")
        except (tk.TclError, AttributeError): pass
        selected_type = self.settings_vars['model_type'].get()
        for name, frame in self.model_frames.items():
            try:
                if name == selected_type and frame.winfo_exists(): frame.pack(fill="x", expand=True, padx=5, pady=5)
                elif frame.winfo_exists(): frame.pack_forget()
            except (tk.TclError, AttributeError): continue
        self._load_provider_params_from_string()

# ====== БАЗОВЫЙ КЛАСС ДЛЯ НАСТРОЕК ======
class BaseSettingsWindow(BaseTopLevel, DynamicModelUI):
    def __init__(self, master, backend, title_key="settings_title", geometry="500x420"):  # УМЕНЬШЕНА ВЫСОТА С 500x500
        BaseTopLevel.__init__(self, master)
        DynamicModelUI.__init__(self)
        self.master = master
        self.backend = backend
        self.title(Lang.get(title_key))
        self.geometry(geometry)
        self.configure(fg_color=DARK_BG)
        self.max_tokens = 8192
        self.validated = True
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
    def setup_model_tab(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=0)  # Изменено для прилипания к верху
        main_frame = create_styled_frame(parent)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)  # Уменьшены отступы
        main_frame.grid_columnconfigure(0, weight=1)
        self._create_model_ui(main_frame)
    def setup_chat_settings_tab(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=0)  # Изменено для прилипания к верху
        # Определяем какие настройки булевы, а какие числовые на основе данных из БД
        all_settings = self.backend.get_global_settings()
        boolean_keys = ['use_rag', 'filter_generations', 'write_log', 'write_results']
        numeric_keys = ['hierarchy_limit', 'max_critic_reactions']
        scrollable_frame = create_scrollable_frame(parent, fg_color="transparent")
        scrollable_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        scrollable_frame.grid_columnconfigure(0, weight=1)
        row = 0
        # Булевы настройки (переключатели)
        for key in boolean_keys:
            if key not in self.settings_vars: self.settings_vars[key] = tk.StringVar(value=all_settings.get(key, '1' if key == 'use_rag' or key == 'write_log' else '0'))
            frame = create_styled_frame(scrollable_frame)
            frame.pack(fill="x", pady=2)
            create_styled_label(frame, text=Lang.get(key)).pack(side="left", padx=(0, 10))
            switch = CTkSwitch(frame, text="", variable=self.settings_vars[key], onvalue="1", offvalue="0", switch_width=50, switch_height=25, progress_color=PURPLE_ACCENT, font=FONT_REGULAR)
            switch.pack(side="right")
            row += 1
        # Числовые настройки (поля ввода)
        for key in numeric_keys:
            if key not in self.settings_vars: self.settings_vars[key] = tk.StringVar(value=all_settings.get(key, '0' if key == 'hierarchy_limit' else '2'))
            frame = create_styled_frame(scrollable_frame)
            frame.pack(fill="x", pady=2)
            frame.grid_columnconfigure(1, weight=1)
            create_styled_label(frame, text=Lang.get(key)).grid(row=0, column=0, sticky="w", padx=(0, 10))
            entry = create_styled_entry(frame, textvariable=self.settings_vars[key])
            entry.grid(row=0, column=1, sticky="ew")
            row += 1
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
        else:
            self.validated = False
            showerror(self, Lang.get("validation_error"), msg)
    def _get_default_settings(self): settings = self.backend.get_global_settings(); return {key: tk.StringVar(value=val) for key, val in settings.items()}

# ====== ГЛАВНОЕ ОКНО ЧАТА ======
class ChatApp(CTk):
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
        self.log_queues = {}
        self.log_windows = {}
        self.chat_blink_states = {}
        self.blink_timer_id = None
        self.settings_window = None
        self.create_chat_window = None
        self.attachment_overlay_frame = None
        self.title(Lang.get("app_title"))
        self.geometry("600x350")
        self.minsize(600, 350)
        self.after(0, lambda: set_windows_dark_titlebar(self))
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        if not self.backend.is_main_config_complete(): self.after(0, self.show_initial_settings)
        else:
            self.backend.rescan_and_localize_modules()
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
                class FLASHWINFO(ctypes.Structure): _fields_ = [("cbSize", ctypes.c_uint), ("hwnd", ctypes.c_void_p), ("dwFlags", ctypes.c_uint), ("uCount", ctypes.c_uint), ("dwTimeout", ctypes.c_uint)]
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
            if askyesno(self, Lang.get("active_chats_on_close_title"), Lang.get("active_chats_on_close_message", count=len(active_chats))):
                for chat_id in active_chats:
                    self.terminate_chat_process(chat_id)
                self.destroy()
            else: return
        else: self.destroy()
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
        for d in [self.chat_processes, self.input_queues, self.output_queues, self.chat_blink_states, self.log_queues]:
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
            if not isinstance(row_frame, CTkFrame) or not hasattr(row_frame, 'winfo_children') or not row_frame.winfo_children(): continue
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
    def setup_main_ui(self):
        for widget in self.winfo_children(): widget.destroy()
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        left_panel_container = create_styled_frame(self, width=200)
        left_panel_container.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        left_panel_container.grid_rowconfigure(0, weight=1)
        left_panel_container.grid_rowconfigure(1, weight=0)
        # Внешняя белая рамка
        chats_bordered_frame = create_styled_frame(left_panel_container, fg_color=DARK_BG, border_color=WHITE, border_width=1, corner_radius=CORNER_RADIUS)
        chats_bordered_frame.grid(row=0, column=0, sticky="nsew")
        # Внутренний скроллируемый фрейм
        self.chats_list_frame = CTkScrollableFrame(chats_bordered_frame, scrollbar_button_color=PURPLE_ACCENT, scrollbar_button_hover_color=WHITE, fg_color="transparent", border_width=0, corner_radius=0)
        self.chats_list_frame.pack(fill="both", expand=True, padx=10, pady=1.5)
        if hasattr(self.chats_list_frame, '_scrollbar'): self.chats_list_frame._scrollbar.configure(width=12)
        # ====== КНОПКИ "НОВЫЙ ЧАТ" И "НАСТРОЙКИ" ======
        bottom_buttons_frame = create_styled_frame(left_panel_container)
        bottom_buttons_frame.grid(row=1, column=0, sticky="ew", pady=(3,0))  # ← row=1 под списком чатов
        bottom_buttons_frame.grid_columnconfigure((0,1), weight=1)
        self.new_chat_btn = create_styled_button(bottom_buttons_frame, text=Lang.get("new_chat"), command=self.create_chat_window_show, width=30)
        self.new_chat_btn.grid(row=0, column=0, padx=(0, 2), sticky="ew")
        self.settings_btn = create_styled_button(bottom_buttons_frame, text=Lang.get("settings"), command=self.open_settings, width=30)
        self.settings_btn.grid(row=0, column=1, padx=(2, 0), sticky="ew")
        # ===== ПРАВАЯ ПАНЕЛЬ (СООБЩЕНИЯ) =====
        right_panel_container = create_styled_frame(self)
        right_panel_container.grid(row=0, column=1, sticky="nsew", padx=(0, 5), pady=5)
        right_panel_container.grid_columnconfigure(0, weight=1)
        right_panel_container.grid_rowconfigure(0, weight=1)  # для messages_bordered_frame
        right_panel_container.grid_rowconfigure(1, weight=0)  # для input_outer_frame
        self.messages_bordered_frame = create_styled_frame(right_panel_container, fg_color=DARK_BG, border_color=WHITE, border_width=1, corner_radius=CORNER_RADIUS)
        self.messages_bordered_frame.grid(row=0, column=0, sticky="nsew", pady=(0,5))
        self.messages_bordered_frame.grid_rowconfigure(0, weight=1)
        self.messages_bordered_frame.grid_columnconfigure(0, weight=1)
        self.messages_frame = CTkScrollableFrame(self.messages_bordered_frame, scrollbar_button_color=PURPLE_ACCENT, scrollbar_button_hover_color=WHITE, fg_color="transparent", border_width=0, corner_radius=0)
        self.messages_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=1.5)
        if hasattr(self.messages_frame, '_scrollbar'): self.messages_frame._scrollbar.configure(width=12)
        self.input_outer_frame = create_styled_frame(right_panel_container)
        self.input_outer_frame.grid(row=1, column=0, sticky="ew")
        self.input_outer_frame.grid_columnconfigure(1, weight=1)
        self.left_controls_frame = create_styled_frame(self.input_outer_frame)
        self.left_controls_frame.grid(row=0, column=0, padx=(0, 2), sticky="n")  # Уменьшен отступ и изменено выравнивание
        self.stop_btn = create_styled_button(self.left_controls_frame, text="☐", width=30, height=30, command=self.stop_chat)
        self.play_btn = create_styled_button(self.left_controls_frame, text="▷", width=30, height=30, command=self.resume_chat)
        self.log_btn = create_styled_button(self.left_controls_frame, text="Log", width=30, height=30, command=self.open_log_window)
        self.input_text = CTkTextbox(self.input_outer_frame, corner_radius=CORNER_RADIUS, border_color=PURPLE_ACCENT, border_width=1, fg_color=DARK_SECONDARY, font=FONT_REGULAR, wrap="word")
        self.input_text._textbox.configure(borderwidth=0, padx=0, pady=0)
        self.input_text.grid(row=0, column=1, sticky="nsew", padx=2)
        self.input_text.bind("<Return>", self.on_enter_pressed)
        self.input_text.bind("<KeyRelease>", self.adjust_input_height, add=True)
        enhance_text_widget(self.input_text)
        self.adjust_input_height()
        self.right_controls_frame = create_styled_frame(self.input_outer_frame)
        self.right_controls_frame.grid(row=0, column=2, padx=(2,0), sticky="n")  # Уменьшен отступ и изменено выравнивание
        self.send_btn = create_styled_button(self.right_controls_frame, text="↑", width=30, height=30, command=self.send_message)
        self.send_btn.pack(side=tk.TOP, anchor="ne")
        self.attach_btn = create_styled_button(self.right_controls_frame, text=" + ", width=30, height=30, command=self.add_attachment)
        self.attach_btn.pack(side=tk.TOP, anchor="ne", pady=(5,0))
        self.load_chats()
        self.start_chat_blinking()
        self.update_chat_controls()
    def _on_mousewheel(self, event, canvas):
        if event.delta:
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        else:
            if event.num == 4: canvas.yview_scroll(-1, "units")
            elif event.num == 5: canvas.yview_scroll(1, "units")
    def update_chat_controls(self):
        for btn in [self.stop_btn, self.play_btn, self.log_btn]:
            btn.pack_forget()
        if not self.current_chat_id:
            self.play_btn.pack(side=tk.TOP)
            self.play_btn.configure(state="disabled")
            return
        has_messages = len(self.backend.get_messages(self.current_chat_id)) > 0
        is_active = self.current_chat_id in self.chat_processes
        if is_active:
            self.stop_btn.pack(side=tk.TOP)
            self.log_btn.pack(side=tk.TOP, pady=(5,0))
            self.play_btn.pack_forget()
        else:
            self.play_btn.pack(side=tk.TOP)
            self.play_btn.configure(state="normal" if has_messages else "disabled")
            self.log_btn.pack_forget()
    def stop_chat(self):
        if self.current_chat_id and self.current_chat_id in self.chat_processes:
            self.terminate_chat_process(self.current_chat_id)
            self.update_chat_controls()
    def resume_chat(self):
        if self.current_chat_id and self.current_chat_id not in self.chat_processes:
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
        files = filedialog.askopenfilenames()
        if files:
            self.attachments.extend(Path(file) for file in files)
            self.show_attachments()
    def show_attachments(self):
        if hasattr(self, 'attachment_overlay_frame') and self.attachment_overlay_frame and self.attachment_overlay_frame.winfo_exists(): self.attachment_overlay_frame.destroy()
        self.attachment_overlay_frame = None
        if not self.attachments: return
        self.attachment_overlay_frame = create_styled_frame(self.messages_bordered_frame, fg_color=DARK_BG, border_color=PURPLE_ACCENT, border_width=1, corner_radius=CORNER_RADIUS)
        self.attachment_overlay_frame.place(relx=0.5, y=5, anchor='n', relwidth=0.75)  # УЖЕ ЕСТЬ y=10
        header_text = f"{Lang.get('attachments')}"
        inner_frame = create_styled_frame(self.attachment_overlay_frame, fg_color=DARK_SECONDARY, corner_radius=CORNER_RADIUS)  # ДОБАВЛЕН corner_radius
        inner_frame.pack(fill="both", expand=True, padx=0, pady=0)
        scrollable_container = create_scrollable_frame(inner_frame, fg_color="transparent", label_text=header_text, label_text_color=WHITE)
        scrollable_container.pack(fill="both", expand=True, padx=5, pady=5)
        for i, attachment in enumerate(self.attachments):
            att_frame = create_styled_frame(scrollable_container)
            att_frame.pack(fill="x", pady=2, padx=2)
            CTkButton(att_frame, text="X", width=22, height=22, fg_color=DARK_SECONDARY, hover_color=PURPLE_ACCENT, text_color=WHITE, command=lambda idx=i: self.remove_attachment(idx)).pack(side=tk.LEFT)
            create_styled_label(att_frame, text=attachment.name, font=("Georgia", 11, "bold"), anchor="w").pack(side=tk.LEFT, padx=5, expand=True, fill="x")
            
        def _update_height():
            try:
                if not self.attachment_overlay_frame or not self.attachment_overlay_frame.winfo_exists(): return
                self.attachment_overlay_frame.update_idletasks()
                required_height = 35 + len(self.attachments) * 30
                parent_height = self.messages_bordered_frame.winfo_height()
                max_height = parent_height * 0.5
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
        chats = self.backend.get_chats()
        first_chat_id = chats[0]["id"] if chats else None
        for chat in chats:
            row_frame = create_styled_frame(self.chats_list_frame)
            row_frame.pack(fill="x", pady=1)
            row_frame.grid_columnconfigure(0, weight=1)
            chat_button = create_styled_button(row_frame, text=chat['name'], anchor="center", fg_color="transparent", border_width=0, command=lambda c_id=chat["id"]: self.on_chat_select(c_id))
            chat_button.grid(row=0, column=0, sticky="ew")
            setattr(chat_button, "chat_id", chat["id"])
            files_path = Path(resource_path(os.path.join("data", "chats", chat["id"], "files")))
            console_folders_path = Path(resource_path(os.path.join("data", "chats", chat["id"], "console_folders")))
            reports_path = Path(resource_path(os.path.join("data", "chats", chat["id"], "reports")))
            results_path = Path(resource_path(os.path.join("data", "chats", chat["id"], "results")))
            has_files = files_path.exists() and files_path.is_dir()
            has_console_folders = console_folders_path.exists() and console_folders_path.is_dir()
            has_reports = reports_path.exists() and reports_path.is_dir()
            has_results = results_path.exists() and results_path.is_dir()
            column_offset = 1
            if has_files:
                files_button = CTkButton(row_frame, text="💾", width=20, fg_color="transparent", hover_color=PURPLE_ACCENT, command=lambda c_id=chat["id"]: self.open_folder(c_id, "files"))
                files_button.grid(row=0, column=column_offset, padx=(2, 0))
                column_offset += 1
            if has_console_folders:
                console_button = CTkButton(row_frame, text=">_", width=20, fg_color="transparent", hover_color=PURPLE_ACCENT, command=lambda c_id=chat["id"]: self.open_folder(c_id, "console_folders"))
                console_button.grid(row=0, column=column_offset, padx=(2, 0))
                column_offset += 1
            if has_reports:
                reports_button = CTkButton(row_frame, text="📄", width=20, fg_color="transparent", hover_color=PURPLE_ACCENT, command=lambda c_id=chat["id"]: self.open_folder(c_id, "reports"))
                reports_button.grid(row=0, column=column_offset, padx=(2, 0))
                column_offset += 1
            if has_results:
                results_button = CTkButton(row_frame, text="📊", width=20, fg_color="transparent", hover_color=PURPLE_ACCENT, command=lambda c_id=chat["id"]: self.open_folder(c_id, "results"))
                results_button.grid(row=0, column=column_offset, padx=(2, 0))
                column_offset += 1
            delete_button = CTkButton(row_frame, text="X", width=20, fg_color="transparent", hover_color=PURPLE_ACCENT, command=lambda c_id=chat["id"]: self.delete_selected_chat(c_id))
            delete_button.grid(row=0, column=column_offset, padx=(2,0))
        self.chats_list_frame.update_idletasks()
        if hasattr(self.chats_list_frame, '_parent_canvas'):
            self.chats_list_frame._parent_canvas.configure(scrollregion=self.chats_list_frame._parent_canvas.bbox("all"))
        chat_ids = [c["id"] for c in chats]
        if current_selection not in chat_ids:
            self.on_chat_select(first_chat_id) if first_chat_id else self.clear_chat_view()
        self.update_chat_list_colors()
    def open_folder(self, chat_id, folder_name):
        folder_path = Path(resource_path(os.path.join("data", "chats", chat_id, folder_name)))
        if not folder_path.exists():
            showinfo(self, Lang.get("info"), Lang.get("folder_not_found", folder_name=folder_name))
            return
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
        for chat in self.backend.get_chats():
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
        if hasattr(self.messages_frame, '_parent_canvas'): self.messages_frame._parent_canvas.configure(scrollregion=self.messages_frame._parent_canvas.bbox("all"))
        self.after(100, lambda: self.messages_frame._parent_canvas.yview_moveto(1))
    def copy_text_to_clipboard(self, text):
        self.clipboard_clear()
        self.clipboard_append(text)
    def add_message_to_ui(self, text, is_my, is_question=False, attachments=None):
        bubble, msg_text_widget = create_chat_message_bubble(self.messages_frame, text, is_my, attachments, is_question)
        setup_message_wraplength(msg_text_widget, self.messages_frame)
        def create_context_menu(event):
            menu = tk.Menu(self, tearoff=0, bg=DARK_SECONDARY, fg=WHITE)
            menu.add_command(label=Lang.get("copy"), command=lambda: self.copy_text_to_clipboard(text))
            menu.tk_popup(event.x_root, event.y_root)
        msg_text_widget.bind("<Button-3>", create_context_menu)
        if attachments:
            att_container = create_styled_frame(bubble)
            att_container.pack(fill="x", pady=(8, 5), padx=5)
            for att in attachments:
                att_frame = create_styled_frame(att_container)
                att_frame.pack(fill=tk.X, pady=1, anchor='w')
                create_styled_label(att_frame, text=Path(att).name).pack(side=tk.LEFT)
                CTkButton(att_frame, text="📂", font=("Georgia", 12), width=25, height=25,
                    fg_color="transparent", hover_color=PURPLE_ACCENT, command=lambda a=att: self.open_attachment(a)).pack(side=tk.RIGHT)
        self.messages_frame.update_idletasks()
        if hasattr(self.messages_frame, '_parent_canvas'):
            self.messages_frame._parent_canvas.configure(scrollregion=self.messages_frame._parent_canvas.bbox("all"))
        self.after(100, lambda: self.messages_frame._parent_canvas.yview_moveto(1.0))
    def open_attachment(self, file_path):
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
        input_queue = multiprocessing.Queue()
        output_queue = multiprocessing.Queue()
        log_queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=initialize_work, args=(BASE_DIR, chat_id, input_queue, output_queue, log_queue))
        p.start()
        self.chat_processes[chat_id] = p
        self.input_queues[chat_id] = input_queue
        self.output_queues[chat_id] = output_queue
        self.log_queues[chat_id] = log_queue
        self.active_chats.add(chat_id)
        self.check_chat_responses(chat_id)
        self.update_chat_controls()
    def check_chat_responses(self, chat_id):
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
            self.create_chat_window.destroy()
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
                bubble = next((w for w in row_frame.winfo_children() if isinstance(w, CTkFrame)), None)
                if not bubble: continue
                content_frame = next((w for w in bubble.winfo_children() if isinstance(w, CTkFrame)), None)
                if not content_frame: continue
                for widget in content_frame.winfo_children():
                    if isinstance(widget, CTkTextbox): self._adjust_textbox_height(widget)
            except (IndexError, tk.TclError, StopIteration): continue

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
        main_frame = create_styled_frame(self)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        message_label = create_styled_label(main_frame, text=message, wraplength=width - 60, justify="left", font=("Georgia", 13, "bold"))
        message_label.grid(row=0, column=0, sticky="nsew")
        btn_frame = create_styled_frame(self)
        btn_frame.grid(row=1, column=0, sticky="se", padx=20, pady=(0, 20))
        for text_key, value in buttons:
            btn_text = Lang.get(text_key.lower())
            btn = create_styled_button(btn_frame, text=btn_text, command=lambda v=value: self.set_result(v))
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

class InitialSettingsWindow(BaseTopLevel, DynamicModelUI):
    def __init__(self, master, backend):
        BaseTopLevel.__init__(self, master)
        DynamicModelUI.__init__(self)
        self.master = master
        self.backend = backend
        self.title("Setup") 
        self.geometry("500x450")  # Уменьшен размер окна
        self.minsize(500, 450)   # Уменьшены минимальные размеры
        self.configure(fg_color=DARK_BG)
        self.max_tokens = 8192
        self.validated = False
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.show_step1_language()
    def show_step1_language(self):
        for widget in self.winfo_children(): widget.destroy()
        self.lang_var = tk.StringVar(value=Lang.current_language or "en")
        container = create_styled_frame(self)
        container.pack(expand=True, fill='both')
        lang_frame = create_styled_frame(container)
        lang_frame.pack(pady=10)
        create_styled_label(lang_frame, text=f"{Lang.get('language')}:").pack(side='left', padx=(0, 10))
        lang_combo = CTkOptionMenu(
            lang_frame, variable=self.lang_var, 
            values=list(Lang.available_languages.keys()), **OPTIONMENU_THEME)
        lang_combo.pack(side='left')
        create_styled_button(container, text="→", command=self.show_step2_model).pack(pady=20)
    def show_step2_model(self):
        Lang.load_language(self.lang_var.get())
        self.title(Lang.get("initial_settings_title"))
        self.backend.rescan_and_localize_modules()
        ModuleManager().load_modules(self.backend)
        for widget in self.winfo_children(): widget.destroy()
        self.settings_vars = self._get_default_settings()
        btn_frame = create_styled_frame(self)
        btn_frame.pack(side="bottom", fill="x", pady=(0, 20), padx=20)
        self.validate_btn = create_styled_button(btn_frame, text=Lang.get("validate_model"), command=self.validate_model)
        self.validate_btn.pack(side="left")
        back_btn = create_styled_button(btn_frame, text="←", command=self.show_step1_language)
        back_btn.pack(side="left", padx=(10, 10))
        self.save_btn = create_styled_button(btn_frame, text=Lang.get("save_and_continue"), command=self.save_settings, state="disabled")
        self.save_btn.pack(side="right")
        main_frame = create_styled_frame(self)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        self._create_model_ui(main_frame)
        self._load_provider_params_from_string()
    def _get_default_settings(self):
        settings = self.backend.get_global_settings()
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
            try:
                if hasattr(self, 'save_btn') and self.save_btn.winfo_exists(): self.save_btn.configure(state="normal")
            except (tk.TclError, AttributeError): pass
        else:
            self.validated = False
            showerror(self, Lang.get("validation_error"), msg)
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

class SettingsWindow(BaseSettingsWindow):
    def __init__(self, master, backend):
        super().__init__(master, backend, "settings_title", "500x450")  # УМЕНЬШЕНА ВЫСОТА С 500x500
        tabview = CTkTabview(self, **TAB_VIEW_THEME)
        tabview.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)  # Уменьшены отступы
        main_tab = tabview.add(Lang.get("tab_main"))
        chat_settings_tab = tabview.add(Lang.get("tab_chat_settings"))
        mods_tab = tabview.add(Lang.get("tab_modules"))
        self.setup_main_tab(main_tab)
        self.setup_chat_settings_tab(chat_settings_tab)
        self.setup_mods_tab(mods_tab)
    def setup_main_tab(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=0)  # Изменено для прилипания к верху
        self.max_tokens = int(self.backend.get_global_settings().get("token_limit", 8192))
        self.validated = True
        self.settings_vars = self._get_default_settings()
        self.original_language = self.settings_vars['language'].get()
        self.original_model_type = self.settings_vars['model_type'].get()
        self.original_connection_string = self.settings_vars['model_provider_params'].get()
        main_frame = create_styled_frame(parent)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)  # Уменьшены отступы
        main_frame.grid_columnconfigure(0, weight=1)
        lang_frame = create_styled_frame(main_frame)
        lang_frame.pack(fill='x', pady=5)
        lang_frame.grid_columnconfigure(0, weight=1)
        lang_frame.grid_columnconfigure(2, weight=1)
        lang_content = create_styled_frame(lang_frame)
        lang_content.grid(row=0, column=1, sticky="ew")
        create_styled_label(lang_content, text=f"{Lang.get('language')}:").grid(row=0, column=0, padx=(0, 10), sticky="w")
        self.lang_combo = CTkOptionMenu(
            lang_content, variable=self.settings_vars['language'], 
            values=list(Lang.available_languages.keys()), **OPTIONMENU_THEME)
        self.lang_combo.grid(row=0, column=1, sticky="ew")
        lang_content.grid_columnconfigure(1, weight=1)
        self._create_model_ui(main_frame)
        btn_frame = create_styled_frame(parent)
        btn_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))  # Уменьшены отступы
        create_styled_button(btn_frame, text=Lang.get("validate_model"), command=self.validate_model).pack(side="left", padx=5)
        self.save_btn_settings = create_styled_button(btn_frame, text=Lang.get("save"), command=self.save_settings)
        self.save_btn_settings.pack(side="left", padx=5)
        create_styled_button(btn_frame, text=Lang.get("reset_settings_button"), command=self.reset_settings).pack(side="right", padx=5)
        self._load_provider_params_from_string()
    def save_settings(self):
        try:
            if self.master.winfo_exists():
                if hasattr(self.master, 'new_chat_btn'): self.master.new_chat_btn.configure(state="disabled")
                if hasattr(self.master, 'settings_btn'): self.master.settings_btn.configure(state="disabled")
                if hasattr(self.master, 'send_btn'): self.master.send_btn.configure(state="disabled")
            current_model_type = self.settings_vars['model_type'].get()
            current_connection_string = self._build_connection_string()
            settings_changed = (current_model_type != self.original_model_type or current_connection_string != self.original_connection_string)
            if not self.validated and settings_changed:
                if not askyesno(self.master, Lang.get("warning"), Lang.get("model_not_validated_continue")): return
            try:
                token_limit = int(self.settings_vars['token_limit'].get())
                if not (1 <= token_limit <= self.max_tokens): raise ValueError
            except (ValueError, TypeError):
                showerror(self.master, Lang.get("error"), Lang.get("token_limit_info", max_tokens=self.max_tokens))
                return
            settings_to_save = {
                'language': self.settings_vars['language'].get(),
                'model_type': self.settings_vars['model_type'].get(),
                'token_limit': self.settings_vars['token_limit'].get(),
                'model_provider_params': self._build_connection_string(),
                'use_rag': self.settings_vars['use_rag'].get(),
                'filter_generations': self.settings_vars['filter_generations'].get(),
                'hierarchy_limit': self.settings_vars['hierarchy_limit'].get(),
                'write_log': self.settings_vars['write_log'].get(),
                'write_results': self.settings_vars['write_results'].get(),
                'max_critic_reactions': self.settings_vars['max_critic_reactions'].get()
            }
            self.backend.update_global_settings(settings_to_save)
            new_language = self.settings_vars['language'].get()
            if new_language != self.original_language: Lang.load_language(new_language)
            ModuleManager().load_modules(self.backend, reload_m=True)
            self.on_close()
        finally:
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
    def setup_mods_tab(self, parent):
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        self.scrollable_frame = create_scrollable_frame(parent, fg_color="transparent", label_text="")
        self.scrollable_frame.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)  # Уменьшены отступы
        self.rebuild_mods_list()
        create_styled_button(parent, text=Lang.get("add_module"), command=self.add_custom_mod).grid(row=1, column=0, pady=5, padx=5)  # Уменьшены отступы
    def rebuild_mods_list(self):
        for widget in self.scrollable_frame.winfo_children(): widget.destroy()
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        module_manager = ModuleManager()
        default_mods = module_manager.get_default_modules()
        custom_mods = module_manager.get_custom_modules()
        if default_mods:
            create_styled_label(self.scrollable_frame, text=Lang.get("system_modules"), font=("Georgia", 14, "bold")).pack(anchor="w", padx=5, pady=(5,2))  # Уменьшены отступы
            for mod in default_mods: self.create_mod_ui(self.scrollable_frame, mod, is_default=True)
        create_styled_label(self.scrollable_frame, text=Lang.get("global_custom_modules"), font=("Georgia", 14, "bold")).pack(anchor="w", padx=5, pady=(10,2))  # Уменьшены отступы
        for mod in custom_mods: self.create_mod_ui(self.scrollable_frame, mod, is_default=False)
    def create_mod_ui(self, parent, mod_data, is_default):
        enabled_var = tk.BooleanVar(value=mod_data["enabled"])
        def toggle_callback(): self.toggle_default_mod(mod_data["id"], enabled_var.get())
        def remove_callback():
            if askyesno(self, Lang.get("warning"), Lang.get("remove_module_confirm")):
                self.backend.remove_custom_mod(mod_data["id"])
                ModuleManager().update_custom_modules(self.backend)
                self.rebuild_mods_list()
        create_module_ui_item(parent, mod_data, "default" if is_default else "custom", enabled_var=enabled_var if is_default else None, on_toggle=toggle_callback if is_default else None, on_remove=None if is_default else remove_callback, show_checkbox=is_default)
    def toggle_default_mod(self, mod_id, enabled): self.backend.update_default_mod_enabled(mod_id, enabled)
    def remove_custom_mod(self, mod_id):
        if askyesno(self, Lang.get("warning"), Lang.get("remove_module_confirm")):
            self.backend.remove_custom_mod(mod_id)
            ModuleManager().update_custom_modules(self.backend)
            self.rebuild_mods_list()

    def add_custom_mod(self):
        path = filedialog.askopenfilename(filetypes=[(Lang.get("python_files"), "*.py")])
        if not path: return
        try:
            self.backend.add_custom_mod(str(Path(path).resolve()))
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
        self.messages_frame = create_scrollable_frame(self, fg_color="transparent", label_text="")
        self.messages_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.messages_frame.grid_columnconfigure(0, weight=1)
        self.log_message_widgets = []
        self.check_log_queue()
    def add_log_message_to_ui(self, text):
        bubble = create_styled_frame(self.messages_frame, border_width=1, border_color=PURPLE_ACCENT, corner_radius=CORNER_RADIUS, fg_color=DARK_BG)
        bubble.pack(fill=tk.X, padx=5, pady=3, ipady=5)
        msg_widget = create_styled_label(bubble, text=text, wraplength=self.messages_frame.winfo_width() - 50, justify="left", fg_color="transparent")
        msg_widget.pack(fill=tk.X, padx=10, pady=5)
        ChatApp.add_label_context_menu(self.master, msg_widget)
        self.log_message_widgets.append(bubble)
        if len(self.log_message_widgets) > 200: self.log_message_widgets.pop(0).destroy()
        self.after(50, lambda: self.messages_frame._parent_canvas.yview_moveto(1.0))
    def check_log_queue(self):
        try:
            while True: self.add_log_message_to_ui(self.log_queue.get_nowait())
        except queue.Empty: pass
        if self.winfo_exists(): self.after(250, self.check_log_queue)

class CreateChatWindow(BaseSettingsWindow):
    def __init__(self, master, backend):
        super().__init__(master, backend, "create_chat_title", "500x550")  # УМЕНЬШЕНА ВЫСОТА С 500x600
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
        top_frame = create_styled_frame(self)
        top_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)  # Уменьшены отступы
        create_styled_label(top_frame, text=Lang.get("chat_name")).pack(side='left', padx=(0,10))
        e = create_styled_entry(top_frame, textvariable=self.settings_vars['chat_name'])
        e.pack(fill='x', expand=True)
        tabview = CTkTabview(self, **TAB_VIEW_THEME)
        tabview.grid(row=1, column=0, sticky="nsew", padx=5, pady=2)  # Уменьшены отступы
        model_tab = tabview.add(Lang.get("tab_model"))
        chat_tab = tabview.add(Lang.get("tab_chat_settings"))
        mods_tab = tabview.add(Lang.get("tab_modules"))
        self.setup_model_tab(model_tab)
        self.setup_chat_settings_tab(chat_tab)
        self.setup_mods_tab(mods_tab)
        bottom_frame = create_styled_frame(self)
        bottom_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=(5, 5))  # Уменьшены отступы
        self.create_btn = create_styled_button(bottom_frame, text=Lang.get("create"), command=self.create_chat_finalize)
        self.create_btn.pack(side='left')
        create_styled_button(bottom_frame, text=Lang.get("validate_model"), command=self.validate_model).pack(side='left', padx=5)
        create_styled_button(bottom_frame, text=Lang.get("cancel"), command=self.destroy).pack(side='left', padx=5)
        self._load_provider_params_from_string()
    def _get_default_settings(self):
        settings = self.backend.get_global_settings()
        s_vars = {key: tk.StringVar(value=val) for key, val in settings.items()}
        s_vars['chat_name'] = tk.StringVar(value=f"{Lang.get('chat_prefix')} {self.backend.generate_id(4)}")
        module_manager = ModuleManager()
        default_mods = module_manager.get_default_modules()
        s_vars['default_mods'] = { mod['id']: tk.BooleanVar(value=mod['enabled']) for mod in default_mods }
        return s_vars
    def setup_mods_tab(self, parent):
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        self.mods_scrollable_frame = create_scrollable_frame(parent, fg_color="transparent")
        self.mods_scrollable_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)  # Уменьшены отступы
        self.rebuild_mods_list()
    def rebuild_mods_list(self):
        for widget in self.mods_scrollable_frame.winfo_children(): widget.destroy()
        self.mods_scrollable_frame.grid_columnconfigure(0, weight=1)
        module_manager = ModuleManager()
        default_mods = module_manager.get_default_modules()
        if default_mods:
            create_styled_label(self.mods_scrollable_frame, text=Lang.get("system_modules"), font=("Georgia", 14, "bold")).pack(anchor="w", padx=5, pady=(5,2))  # Уменьшены отступы
            for mod in default_mods: self.create_mod_ui(self.mods_scrollable_frame, mod, "default")
        if self.custom_mods_for_chat:
            create_styled_label(self.mods_scrollable_frame, text=Lang.get("global_custom_modules"), font=("Georgia", 14, "bold")).pack(anchor="w", padx=5, pady=(10,2))  # Уменьшены отступы
            for mod in self.custom_mods_for_chat: self.create_mod_ui(self.mods_scrollable_frame, mod, "global_custom")
        header_frame = create_styled_frame(self.mods_scrollable_frame)
        header_frame.pack(fill='x', pady=(10,2))  # Уменьшены отступы
        create_styled_label(header_frame, text=Lang.get("chat_specific_modules"), font=("Georgia", 14, "bold")).pack(side='left', anchor="w", padx=5)
        create_styled_button(header_frame, text="+", width=30, command=self.add_new_local_mod).pack(side='left', padx=5)
        if self.newly_added_mods:
            for mod in self.newly_added_mods: self.create_mod_ui(self.mods_scrollable_frame, mod, "new_custom")
    def create_mod_ui(self, parent, mod_data, mod_type):
        if mod_type == "default":
            enabled_var = self.settings_vars['default_mods'][mod_data["id"]]
            create_module_ui_item(parent, mod_data, mod_type, enabled_var=enabled_var, show_checkbox=True)
        else:
            def remove_callback():
                if mod_type == "global_custom": self.remove_mod_from_chat_list(mod_data["id"], self.custom_mods_for_chat)
                elif mod_type == "new_custom": self.remove_mod_from_chat_list(mod_data["id"], self.newly_added_mods)
            create_module_ui_item(parent, mod_data, mod_type, on_remove=remove_callback, show_checkbox=False)
    def remove_mod_from_chat_list(self, mod_id_to_remove, mod_list):
        mod_list[:] = [m for m in mod_list if m.get("id") != mod_id_to_remove]
        self.rebuild_mods_list()
    def add_new_local_mod(self):
        path_str = filedialog.askopenfilename(filetypes=[(Lang.get("python_files"), "*.py")])
        if not path_str: return
        path = str(Path(path_str).resolve())
        valid, msg = ModuleValidator.validate_module(path)
        if not valid:
            showerror(self, Lang.get("error"), msg)
            return
        name, description = self.backend._get_localized_doc(Path(path), lang=Lang.current_language)
        new_mod = {"id": self.backend.generate_id(6), "name": name, "description": description, "adress": path}
        self.newly_added_mods.append(new_mod)
        self.rebuild_mods_list()
    def create_chat_finalize(self):
        chat_name = self.settings_vars['chat_name'].get().strip()
        if not chat_name:
            showerror(self, Lang.get("error"), Lang.get("enter_chat_name"))
            return
        current_model_type = self.settings_vars['model_type'].get()
        current_connection_string = self._build_connection_string()
        settings_changed = (current_model_type != self.original_model_type or current_connection_string != self.original_connection_string)
        if not self.validated and settings_changed:
            if not askyesno(self, Lang.get("warning"), Lang.get("model_not_validated_continue")):
                return
        model_config = {
            'model_type': self.settings_vars['model_type'].get(),
            'model_provider_params': self._build_connection_string(),
            'token_limit': self.settings_vars['token_limit'].get()
        }
        chat_config = {
            "language": Lang.current_language,
            "use_rag": self.settings_vars['use_rag'].get(),
            "filter_generations": self.settings_vars['filter_generations'].get(),
            "hierarchy_limit": self.settings_vars['hierarchy_limit'].get(),
            "write_log": self.settings_vars['write_log'].get(),
            "write_results": self.settings_vars['write_results'].get(),
            "max_critic_reactions": self.settings_vars['max_critic_reactions'].get()
        }
        module_manager = ModuleManager()
        default_mods = module_manager.get_default_modules()
        default_mods_config = {mid: var.get() for mid, var in self.settings_vars['default_mods'].items()}
        final_custom_mods = self.custom_mods_for_chat + self.newly_added_mods
        settings_bundle = {
            "model_config": model_config, "chat_config": chat_config, 
            "default_mods_config": default_mods_config, "custom_mods_list": final_custom_mods
        }
        self.on_close()
        chat_data = self.backend.create_chat(chat_name, settings_bundle)
        if not chat_data:
            showerror(self.master, Lang.get("error"), Lang.get("chat_name_exists"))
            return
        self.master.load_chats()
        self.master.on_chat_select(chat_data["id"])

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
        label = tk.Label(splash, text="Loading...", font=("Georgia", 24), bg='black', fg='white')
    x, y = (sw - w) // 2, (sh - h) // 2
    splash.geometry(f"{w}x{h}+{x}+{y}")
    label.pack()
    splash.attributes('-topmost', True)
    if sys.platform == "win32": splash.attributes('-transparentcolor', 'black')
    def poll():
        if app_ready_event.is_set(): splash.after(500, lambda: (splash.destroy(), root.quit()))
        else: splash.after(50, poll)
    splash.after(50, poll)
    root.mainloop()

def run_main_app(app_ready_event: multiprocessing.Event):
    from cross_gpt import initialize_work
    if sys.platform == "darwin":
        try:
            from AppKit import NSApp
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
    global Lang
    Lang = LanguageManager()
    customtkinter.set_appearance_mode("dark")
    selected_language = "en"
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
        except sqlite3.OperationalError: pass
        except Exception as e: print(f"Could not read language from DB: {e}")
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

if getattr(sys, 'frozen', False):
    import __main__
    for attr in ['run_main_app', 'show_splash_screen']:
        if hasattr(sys.modules[__name__], attr): setattr(__main__, attr, getattr(sys.modules[__name__], attr))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app_is_ready_event = multiprocessing.Event()
    import ui
    main_app_process = multiprocessing.Process(target=ui.run_main_app, args=(app_is_ready_event,))
    main_app_process.start()
    show_splash(app_is_ready_event)
    sys.exit(0)