import tkinter as tk
from PIL import Image, ImageTk
import multiprocessing
import sys
import os

# ====== БАЗОВЫЕ ПУТИ (остаются глобальными для заставки и других вызовов) ======
def get_base_dir():
    if getattr(sys, 'frozen', False): return os.path.dirname(os.path.abspath(sys.executable))
    else: return os.path.dirname(os.path.abspath(__file__))

def resource_path(relative_path): return os.path.join(get_base_dir(), relative_path)

# ====== ЗАСТАВКА (использует только лёгкие модули) ======
def show_splash(app_ready_event: multiprocessing.Event):
    if sys.platform.startswith("win32"):
        icon_path = resource_path(os.path.join("data", "icons", "icon.png"))
        if not os.path.exists(icon_path): print(f"Иконка для сплэша не найдена: {icon_path}"); return
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
        splash.attributes('-transparentcolor', 'black')
        def poll():
            if app_ready_event.is_set(): splash.after(500, lambda: (splash.destroy(), root.quit()))
            else: splash.after(50, poll)
        splash.after(50, poll)
        root.mainloop()
    else:
        icon_path = resource_path(os.path.join("data", "icons", "icon.png"))
        if not os.path.exists(icon_path): print(f"Иконка для сплэша не найдена: {icon_path}"); return
        splash = tk.Tk()
        splash.overrideredirect(True)
        bg_color = '#1da244'
        splash.configure(bg=bg_color)
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
            label = tk.Label(splash, image=img_tk, bg=bg_color)
            label.image = img_tk
        except Exception as e:
            print(f"Splash image load failed: {e}")
            w, h = 400, 300
            label = tk.Label(splash, text="Loading...", font=("Georgia", 24), bg=bg_color, fg='white')
        splash.configure(bg=bg_color)
        x, y = (sw - w) // 2, (sh - h) // 2
        splash.geometry(f"{w}x{h}+{x}+{y}")
        label.pack()
        splash.attributes('-topmost', True)
        def poll():
            if app_ready_event.is_set(): splash.after(500, splash.destroy)
            else: splash.after(50, poll)
        splash.after(50, poll)
        splash.mainloop()

# ====== ОСНОВНАЯ ФУНКЦИЯ (все тяжёлые импорты и логика приложения внутри) ======
def run_main_app(app_ready_event: multiprocessing.Event):
    import customtkinter
    import sqlite3
    import json
    import random
    import string
    import ast
    import queue
    import importlib.util
    import shutil
    import subprocess
    import platform
    from contextlib import redirect_stdout
    import io
    import encryption_utils
    from customtkinter import (
        CTkButton, CTkEntry, CTkFrame, CTkLabel, CTkScrollableFrame,
        CTkTabview, CTkRadioButton, CTkSwitch, CTkOptionMenu,
        CTkTextbox, CTkCheckBox, CTkToplevel, CTk)
    from pathlib import Path
    from tkinter import filedialog
    DARK_BG = "#000000"
    DARK_ENTRY_BG = "#1e1e1e"
    DARK_SECONDARY = "#2a2a2a"
    DARK_BORDER = "#333333"
    PURPLE_ACCENT = "#5200ff"
    ACTIVE_CHAT_COLOR = "#ff9900"
    WHITE = "#c7c7c7"
    DARK_TEXT_SECONDARY = "#b0b0b0"
    CORNER_RADIUS = 12
    FONT_FAMILY = "Georgia"
    FONT_REGULAR = (FONT_FAMILY, 12)
    BUTTON_THEME = {
        "fg_color": "transparent",
        "hover_color": PURPLE_ACCENT,
        "corner_radius": 50,
        "font": FONT_REGULAR,
        "width": 20,
        "height": 20}
    ENTRY_THEME = {
        "fg_color": PURPLE_ACCENT,
        "border_width": 0,
        "corner_radius": CORNER_RADIUS,
        "font": FONT_REGULAR,
        "text_color": WHITE,
        "height": 27}
    TAB_VIEW_THEME = {
        "segmented_button_selected_color": PURPLE_ACCENT,
        "segmented_button_unselected_color": DARK_SECONDARY,
        "segmented_button_selected_hover_color": PURPLE_ACCENT,
        "fg_color": DARK_BG}
    OPTIONMENU_THEME = {
        "fg_color": DARK_SECONDARY,
        "button_color": DARK_SECONDARY,
        "button_hover_color": PURPLE_ACCENT,
        "dropdown_fg_color": DARK_SECONDARY,
        "dropdown_hover_color": PURPLE_ACCENT,
        "corner_radius": CORNER_RADIUS,
        "font": FONT_REGULAR}
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
        default_kwargs = {"font": FONT_REGULAR}
        default_kwargs.update(kwargs)
        if "fg_color" not in default_kwargs: default_kwargs["fg_color"] = "transparent"
        if "height" not in default_kwargs: default_kwargs["height"] = 0
        return CTkLabel(parent, text=text, **default_kwargs)
    def create_param_widget(parent, param_info, settings_vars_dict, path_vars_dict, on_change_callback=None):
        param_name = param_info['name']
        default_val = param_info.get('default')
        is_file = param_info['is_file']
        param_frame = create_styled_frame(parent)
        param_frame.pack(fill="x", pady=2, padx=5)
        param_frame.grid_columnconfigure(1, weight=1)
        label_text = param_name
        if default_val is not None and str(default_val).strip() != '': label_text += f" {default_val}"
        create_styled_label(param_frame, label_text).grid(row=0, column=0, sticky="w", padx=(0, 10))
        input_frame = create_styled_frame(param_frame)
        input_frame.grid(row=0, column=1, sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)
        is_secret = param_name.lower() in ["api_token", "password", "token"]
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
            if on_change_callback: settings_vars_dict[param_name].trace_add("write", lambda *args: on_change_callback(param_name))
            entry_kwargs = {}
            if is_secret: entry_kwargs["show"] = "•"
            entry = create_styled_entry(input_frame, textvariable=settings_vars_dict[param_name], **entry_kwargs)
            entry.grid(row=0, column=0, sticky="ew")
        return param_frame
    def create_module_ui_item(parent, module_data, module_type, enabled_var=None, on_toggle=None, on_remove=None, show_checkbox=True):
        frame = create_styled_frame(parent, border_width=1, border_color=DARK_BORDER, corner_radius=CORNER_RADIUS)
        frame.pack(fill="x", padx=5, pady=3, ipady=5)
        frame.grid_columnconfigure(1, weight=1)
        if show_checkbox and enabled_var is not None:
            cb = CTkCheckBox(frame, variable=enabled_var, text="", border_color=WHITE, checkmark_color=WHITE, fg_color=(DARK_SECONDARY, PURPLE_ACCENT), command=lambda: on_toggle() if on_toggle else None)
            cb.grid(row=0, column=0, rowspan=1, padx=10)
        info_frame = create_styled_frame(frame)
        info_frame.grid(row=0, column=1, rowspan=1, sticky="ew", padx=10)
        create_styled_label(info_frame, module_data["name"], font=FONT_REGULAR).pack(anchor="w")
        desc_label = create_styled_label(info_frame, text=module_data["description"], text_color=DARK_TEXT_SECONDARY, wraplength=400, justify="left")
        desc_label.pack(anchor="w", fill="x")
        if hasattr(parent, 'master') and hasattr(parent.master, 'add_label_context_menu'): parent.master.add_label_context_menu(parent.master, desc_label)
        elif hasattr(parent.winfo_toplevel(), 'add_label_context_menu'): parent.winfo_toplevel().add_label_context_menu(parent.winfo_toplevel(), desc_label)
        if on_remove:
            remove_btn = CTkButton(frame, text="X", width=25, height=25, fg_color="transparent", hover_color=PURPLE_ACCENT, text_color=WHITE, command=on_remove)
            remove_btn.grid(row=0, column=2, rowspan=1, padx=10)
        return frame
    def create_chat_message_bubble(parent, text, is_my, attachments=None, is_question=False):
        row_frame = create_styled_frame(parent)
        row_frame.pack(fill=tk.X, pady=2, padx=10, anchor="center")
        if is_my: bubble = create_styled_frame(row_frame, border_width=0, corner_radius=CORNER_RADIUS, fg_color=DARK_BG)
        else: bubble = create_styled_frame(row_frame, border_width=0, corner_radius=CORNER_RADIUS, fg_color=PURPLE_ACCENT)
        bubble.pack(expand=False, anchor="center")
        msg_text_widget = CTkLabel(bubble, text=text, justify="left", anchor="w", fg_color="transparent", text_color=WHITE, font=FONT_REGULAR, height=0)
        if is_question and not is_my: msg_text_widget.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 8), pady=6)
        else: msg_text_widget.pack(fill=tk.X, expand=True, padx=8, pady=6)
        return bubble, msg_text_widget

    def setup_message_wraplength(widget, messages_frame):
        def update_wraplength(event=None):
            try:
                if widget.winfo_exists():
                    available_width = messages_frame.winfo_width() * 0.905
                    if available_width > 50: widget.configure(wraplength=available_width)
            except Exception: pass
        messages_frame.bind("<Configure>", update_wraplength)
        widget.after(100, update_wraplength)
        return update_wraplength
    def setup_window_geometry(window, width=600, height=500):
        window.geometry(f"{width}x{height}")
        window.minsize(width, height)
        window.after(10, lambda: set_windows_dark_titlebar(window))
        window.after_idle(lambda: center_window(window))
        return window
    def center_window(window):
        window.update_idletasks()
        try:
            width, height = window.winfo_width(), window.winfo_height()
            x = (window.winfo_screenwidth() // 2) - (width // 2)
            y = (window.winfo_screenheight() // 2) - (height // 2)
            window.geometry(f'{width}x{height}+{x}+{y}')
        except tk.TclError: pass
    def create_tabbed_interface(parent, tabs_config):
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
    def load_settings_from_backend(backend, additional_settings=None):
        settings = backend.get_global_settings()
        if additional_settings: settings.update(additional_settings)
        settings_vars = {}
        for key, val in settings.items(): settings_vars[key] = tk.StringVar(value=val)
        return settings_vars
    def save_settings_to_backend(backend, settings_vars, keys_to_save=None):
        if keys_to_save is None: keys_to_save = settings_vars.keys()
        settings_to_save = {}
        for key in keys_to_save:
            if key in settings_vars: settings_to_save[key] = settings_vars[key].get()
        backend.update_global_settings(settings_to_save)
        return True
    def create_scrollable_frame(parent, **kwargs):
        scroll_frame = CTkScrollableFrame(parent, scrollbar_button_color=PURPLE_ACCENT, scrollbar_button_hover_color=WHITE, **kwargs)
        if hasattr(scroll_frame, '_scrollbar'):
            scroll_frame._scrollbar.configure(width=12)
            try: scroll_frame._scrollbar.configure(corner_radius=50)
            except: pass
        if hasattr(scroll_frame, '_scrollbar_horizontal'):
            scroll_frame._scrollbar_horizontal.configure(width=12)
            try: scroll_frame._scrollbar_horizontal.configure(corner_radius=50)
            except: pass
        return scroll_frame
    def browse_file_dialog(entry_widget, entry_var, full_path_var, filetypes=None):
        if filetypes is None: filetypes = [("All files", "*.*")]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            full_path_var.set(path)
            entry_var.set(Path(path).name)
            entry_widget.delete(0, "end")
            entry_widget.insert(0, Path(path).name)
    def show_message_dialog(parent, title, message, buttons): dialog = CustomMessageBox(parent, title, message, buttons); parent.wait_window(dialog); return dialog.result
    def showinfo(parent, title, message): return show_message_dialog(parent, title, message, [("OK", True)])
    def showerror(parent, title, message): return show_message_dialog(parent, title, message, [("OK", True)])
    def showwarning(parent, title, message): return show_message_dialog(parent, title, message, [("OK", True)])
    def askyesno(parent, title, message): return show_message_dialog(parent, title, message, [("Yes", True), ("No", False)])
    def enhance_text_widget(widget, on_change=None):
        is_textbox = isinstance(widget, CTkTextbox)
        is_entry = isinstance(widget, CTkEntry)
        def select_all(event=None):
            if is_textbox: widget.tag_add("sel", "1.0", "end")
            elif is_entry: widget.select_range(0, 'end')
            return "break"
        def copy_action(event=None):
            try:
                if is_textbox and widget.tag_ranges("sel"): selected_text = widget.get(tk.SEL_FIRST, tk.SEL_LAST); widget.clipboard_clear(); widget.clipboard_append(selected_text)
                elif is_entry and widget.select_present(): selected_text = widget.selection_get(); widget.clipboard_clear(); widget.clipboard_append(selected_text)
            except (tk.TclError, AttributeError): pass
            return "break"
        def cut_action(event=None):
            is_disabled = hasattr(widget, '_state') and widget._state == 'disabled'
            if is_disabled: return "break"
            try:
                copy_action(event)
                if is_textbox and widget.tag_ranges("sel"): widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
                elif is_entry and widget.select_present(): widget.delete(widget.index(tk.SEL_FIRST), widget.index(tk.SEL_LAST))
                # Вызываем callback изменения содержимого, если он предоставлен
                if callable(on_change): on_change()
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
                # Вызываем callback изменения содержимого, если он предоставлен
                if callable(on_change): on_change()
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
        menu = tk.Menu(widget, tearoff=0, bg=DARK_SECONDARY, fg=WHITE, relief="flat", borderwidth=0, font=(FONT_FAMILY, 8))
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
        base = Path(__file__).resolve().parent
        png_path = str(base / "data" / "icons" / "icon.png")
        ico_path = str(base / "data" / "icons" / "icon.ico")
        if not Path(png_path).exists(): return
        try: img = tk.PhotoImage(file=png_path); root.iconphoto(True, img)
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
    # ------------------------------------------------------------
    # 4. Классы (AppCache, LanguageManager, ModuleValidator, ...)
    # ------------------------------------------------------------
    class AppCache:
        _instance = None
        def __new__(cls):
            if not cls._instance: cls._instance = super(AppCache, cls).__new__(cls)
            return cls._instance
        def __init__(self):
            if not hasattr(self, 'initialized'):
                self.chats = None
                self.global_settings = None
                self.settings_metadata = None   # кэш для метаданных настроек (ключ -> widget_type)
                self.chats_loaded = False
                self.settings_loaded = False
                self.metadata_loaded = False
                self.initialized = True
        def clear_chats_cache(self):
            self.chats = None
            self.chats_loaded = False
        def clear_settings_cache(self):
            self.global_settings = None
            self.settings_loaded = False
        def clear_metadata_cache(self):
            self.settings_metadata = None
            self.metadata_loaded = False
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
        def get_settings_metadata(self, backend):
            if not self.metadata_loaded or self.settings_metadata is None:
                self.settings_metadata = backend._load_settings_metadata_from_db()
                self.metadata_loaded = True
            return self.settings_metadata
        def update_settings_metadata(self, metadata):
            self.settings_metadata = metadata
            self.metadata_loaded = True
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
            if not loaded:
                self.texts = {"lang_load_error_title": "Language Error", "lang_load_error_message": "Could not load any language files. Please ensure 'lang/en' directory exists."}
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
                    self.providers[module_name] = {"path": py_file, "name": display_name, "params": params}
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
                                    if isinstance(target, ast.Name) and target.id == 'params': has_params_in_connect = True; break
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
                                                key_name = self._get_ast_value(key)
                                                if key_name is None: continue
                                                default_value = self._get_ast_value(value)
                                                param_data = {'name': key_name, 'default': default_value, 'is_file': 'path' in key_name.lower() or 'file' in key_name.lower() or 'dir' in key_name.lower()}
                                                params_info.append(param_data)
                return params_info
            except Exception as e: print(f"Could not parse params from connect function for {path.name}: {e}"); return []
        @staticmethod
        def _get_ast_value(node):
            if isinstance(node, ast.Constant): return node.value
            elif isinstance(node, ast.Str): return node.s
            elif isinstance(node, ast.Num): return node.n
            elif isinstance(node, ast.NameConstant): return node.value
            else: return None
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
            if self.loaded and not reload_m: print("Modules already loaded, skipping"); return
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
        def update_custom_modules(self, backend): self.custom_modules = backend.get_custom_mods()

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
            except Exception as e: print(f"[SQL Error] {query} | {params} -> {e}"); return None
            finally:
                if 'conn' in locals(): conn.close()
        def _get_localized_doc(self, mod_path: Path, lang=None):
            localized_name, localized_desc = None, None
            if lang is None: lang = Lang.current_language
            lang_file = mod_path.with_name(f"{mod_path.stem}_lang.py")
            if lang_file.exists() and lang and lang != "en":
                try:
                    spec = importlib.util.spec_from_file_location("lang_module", str(lang_file))
                    lang_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(lang_module)
                    if hasattr(lang_module, 'locales') and lang in lang_module.locales:
                        locale_data = lang_module.locales[lang]
                        if 'module_doc' in locale_data and len(locale_data['module_doc']) >= 4: localized_name = locale_data['module_doc'][2]; localized_desc = locale_data['module_doc'][3]
                except Exception as e: print(f"Ошибка загрузки локализации для {mod_path.name}: {e}")
            with open(mod_path, 'r', encoding='utf-8') as f: source = f.read()
            tree = ast.parse(source)
            docstring = ast.get_docstring(tree) or ""
            doc_lines = docstring.strip().split('\n')
            name = localized_name or (doc_lines[2].strip() if len(doc_lines) > 2 else mod_path.stem)
            description = localized_desc or (doc_lines[3].strip() if len(doc_lines) > 3 else Lang.get("module_desc_missing"))
            return name, description
        def rescan_and_localize_modules(self):
            db_path = self.db_path
            system_os = platform.system().lower()
            current_language = Lang.current_language
            self._check_existing_modules(db_path, current_language)
            self._scan_default_tools(db_path, system_os, current_language)
            ModuleManager().load_modules(self)
            return True
        def _check_existing_modules(self, db_path, current_language):
            current_defaults = self.sql_exec(db_path, "SELECT id, adress, lang FROM default_mods", fetchall=True) or []
            for mod_id, mod_adress, mod_lang in current_defaults:
                mod_path = Path(resource_path(os.path.join("default_tools", mod_adress)))
                if not mod_path.exists(): self.sql_exec(db_path, "DELETE FROM default_mods WHERE id = ?", (mod_id,)); continue
                if mod_lang != current_language: new_name, new_desc = self._get_localized_doc(mod_path, lang=current_language); self.sql_exec(db_path, "UPDATE default_mods SET name = ?, description = ?, lang = ? WHERE id = ?", (new_name, new_desc, current_language, mod_id))
            current_customs = self.sql_exec(db_path, "SELECT id, adress, lang FROM custom_mods", fetchall=True) or []
            for mod_id, mod_adress, mod_lang in current_customs:
                mod_path = Path(mod_adress)
                if not mod_path.exists(): self.sql_exec(db_path, "DELETE FROM custom_mods WHERE id = ?", (mod_id,)); continue
                if mod_lang != current_language: new_name, new_desc = self._get_localized_doc(mod_path, lang=current_language); self.sql_exec(db_path, "UPDATE custom_mods SET name = ?, description = ?, lang = ? WHERE id = ?", (new_name, new_desc, current_language, mod_id))
        def _scan_default_tools(self, db_path, system_os, current_language):
            default_mods_dir = Path(resource_path("default_tools"))
            if not default_mods_dir.exists(): return
            def process_mod_file(mod_file, relative_path_str):
                if mod_file.name.endswith("_lang.py"): return
                mod_name_stem = mod_file.stem.lower()
                if mod_name_stem in ['windows_cmd', 'linux_cmd', 'macos_cmd']:
                    if not (system_os == 'windows' and mod_name_stem == 'windows_cmd') or (system_os == 'linux' and mod_name_stem == 'linux_cmd') or (system_os == 'darwin' and mod_name_stem == 'macos_cmd'): return
                existing = self.sql_exec(db_path, "SELECT id FROM default_mods WHERE adress = ?", (relative_path_str,), fetchone=True)
                if existing: return
                valid, msg = ModuleValidator.validate_module(str(mod_file.resolve()))
                if not valid: print(f"Ошибка в модуле по умолчанию {mod_file.name}: {msg}"); return
                name, description = self._get_localized_doc(mod_file, lang=current_language)
                self.sql_exec(db_path, "INSERT OR IGNORE INTO default_mods (name, description, adress, enabled, lang) VALUES (?, ?, ?, ?, ?)", (name, description, relative_path_str, 0, current_language))
            for item in default_mods_dir.rglob("*.py"):
                if item.is_file(): relative_path = item.relative_to(default_mods_dir); process_mod_file(item, str(relative_path))
        def init_settings_db(self):
            Path("data").mkdir(exist_ok=True)
            db_path = self.db_path
            self.sql_exec(db_path, "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT, widget_type TEXT DEFAULT 'entry')")
            self.sql_exec(db_path, """CREATE TABLE IF NOT EXISTS default_mods (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, description TEXT, adress TEXT UNIQUE, enabled INTEGER DEFAULT 0, lang TEXT DEFAULT 'en')""")
            self.sql_exec(db_path, """CREATE TABLE IF NOT EXISTS custom_mods (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, description TEXT, adress TEXT UNIQUE, enabled INTEGER DEFAULT 1, lang TEXT DEFAULT 'en')""")
            self.sql_exec(db_path, "INSERT OR IGNORE INTO settings (key, value) VALUES ('language', 'en')")
            providers = ProviderManager().get_providers()
            default_provider = list(providers.keys())[0] if providers else ""
            # Изменение 1: проверка AVX2 на Linux для allow_ocr
            if sys.platform.startswith("linux"):
                try:
                    import subprocess
                    has_avx2 = subprocess.run(['grep', '-q', 'avx2', '/proc/cpuinfo'], capture_output=True).returncode == 0
                    allow_ocr = "1" if has_avx2 else "0"
                except:
                    allow_ocr = "0"
            else:
                allow_ocr = "1"
            defaults = {
                "token_limit": "8192", "model_provider_params": "",
                "model_type": default_provider, "use_rag": "1",
                "filter_generations": "0", "hierarchy_limit": "0",
                "write_log": "1", "write_results": "0", "max_critic_reactions": "2",
                "max_token_limit": "8192", "use_librarian": "1",
                "recreate_agents": "0",
                "skip_nested_images": "0",
                "cut_wrong_command_history": "1",
                "allow_ocr": allow_ocr}
            for key, value in defaults.items(): self.sql_exec(db_path, "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (key, value))
            # Устанавливаем widget_type для известных ключей
            widget_type_map = {
                "use_rag": "switch",
                "filter_generations": "switch",
                "write_log": "switch",
                "write_results": "switch",
                "use_librarian": "switch",
                "recreate_agents": "switch",
                "skip_nested_images": "switch",
                "cut_wrong_command_history": "switch",
                "allow_ocr": "switch",
                "hierarchy_limit": "entry",
                "max_critic_reactions": "entry",}
            for key, wtype in widget_type_map.items(): self.sql_exec(db_path, "UPDATE settings SET widget_type = ? WHERE key = ?", (wtype, key))
        def _load_settings_metadata_from_db(self): # Возвращает список кортежей (key, widget_type) для всех записей settings.
            rows = self.sql_exec(self.db_path, "SELECT key, widget_type FROM settings", fetchall=True) or []
            return {row[0]: row[1] for row in rows}
        def get_settings_metadata(self): return self.cache.get_settings_metadata(self)
        def generate_id(self, length=12): return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
        def _load_chats_from_db(self):
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
            if any(chat['name'].lower() == chat_name.lower() for chat in existing_chats): return None
            chat_id = self.generate_id()
            while (Path(resource_path(os.path.join("data", "chats"))) / chat_id).exists(): chat_id = self.generate_id()
            chat_path = Path(resource_path(os.path.join("data", "chats"))) / chat_id
            chat_path.mkdir(parents=True, exist_ok=True)
            # Изменение 2: всегда создаём папку files
            (chat_path / "files").mkdir(exist_ok=True)
            new_chat = {"id": chat_id, "name": chat_name}
            updated_chats = [new_chat] + existing_chats
            self.cache.update_chats(updated_chats)
            default_mods = ModuleManager().get_default_modules()
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
        def update_default_mod_enabled(self, mod_id, enabled): self.sql_exec(self.db_path, "UPDATE default_mods SET enabled = ? WHERE id = ?", (1 if enabled else 0, mod_id)); return True
        def remove_custom_mod(self, mod_id): self.sql_exec(self.db_path, "DELETE FROM custom_mods WHERE id = ?", (mod_id,)); return True
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
        def validate_model_settings(self, model_type, connection_string, plain_password=None):
            max_tokens = 8192
            try:
                if not model_type: return False, Lang.get("model_err_no_provider"), max_tokens
                provider_manager = ProviderManager()
                provider_data = provider_manager.providers.get(model_type)
                if not provider_data: return False, Lang.get("model_err_provider_missing", provider=model_type), max_tokens
                provider_module = importlib.import_module(f"model_providers.{model_type}")
                f = io.StringIO()
                with redirect_stdout(f):
                    try:
                        import inspect
                        sig = inspect.signature(provider_module.connect)
                        if '_decrypted_password' in sig.parameters: valid, tokens, _, *rest = provider_module.connect(connection_string, _decrypted_password=plain_password)
                        else: valid, tokens, _, *rest = provider_module.connect(connection_string)
                    except Exception as ex:
                        try: valid, tokens, _, error_text = provider_module.connect(connection_string); return False, Lang.get("model_err_validation_generic", e=error_text), max_tokens
                        except: return False, Lang.get("model_err_validation_generic", e=str(ex)), max_tokens
                if hasattr(provider_module, 'disconnect'): provider_module.disconnect()
                if valid: return True, Lang.get("model_validated_success", tokens=tokens), tokens
                else: err_msg = rest[0] if rest else Lang.get("custom_api_fail"); return False, err_msg, max_tokens
            except ImportError as e: print(e); return False, Lang.get("model_err_provider_missing", provider=model_type), max_tokens
            except Exception as e: return False, Lang.get("model_err_validation_generic", e=str(e)), max_tokens
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
        def on_close(self): self.grab_release(); self.destroy()
    # ===== Общая функция для построения UI настроек чата =====
    def build_chat_settings_ui(parent, settings_vars, metadata):
        """
        Строит виджеты для настроек чата на основе переданных метаданных.
        metadata: словарь {key: widget_type}
        settings_vars: словарь, где для каждого ключа уже должен быть создан tk.StringVar
        Возвращает словарь созданных виджетов (на случай, если понадобится дополнительная настройка)
        """
        created_widgets = {}
        # Разделяем настройки: сначала поля ввода (entry), затем переключатели (switch)
        entry_items = [(k, v) for k, v in metadata.items() if v != 'switch']
        switch_items = [(k, v) for k, v in metadata.items() if v == 'switch']
        for key, wtype in entry_items + switch_items:
            # Пропускаем служебные ключи, которые не должны отображаться в настройках чата
            if key in ('language', 'model_type', 'model_provider_params', 'token_limit', 'max_token_limit', 'chat_name'): continue
            frame = create_styled_frame(parent)
            frame.pack(fill="x", pady=2)
            frame.grid_columnconfigure(1, weight=1)
            create_styled_label(frame, text=Lang.get(key, default=key)).grid(row=0, column=0, sticky="w", padx=(0, 10))
            if wtype == 'switch':
                # Создаем Switch
                var = settings_vars[key]
                switch = CTkSwitch(frame, text="", variable=var, onvalue="1", offvalue="0", switch_width=50, switch_height=25, progress_color=PURPLE_ACCENT, font=FONT_REGULAR)
                switch.grid(row=0, column=1, sticky="e")
                created_widgets[key] = switch
            else: # 'entry' или любой другой тип, используем entry
                entry = create_styled_entry(frame, textvariable=settings_vars[key])
                entry.grid(row=0, column=1, sticky="ew")
                created_widgets[key] = entry
        return created_widgets
    class DynamicModelUI:
        def __init__(self):
            self.provider_param_full_paths = {}
            self.model_frames = {}
            self.provider_manager = ProviderManager()
            self.providers = self.provider_manager.get_providers()
            self.real_token_values = {}
            self.real_pwd_status = {}
        def _create_model_ui(self, parent):
            model_type_frame = create_styled_frame(parent)
            model_type_frame.pack(fill="x", pady=10)
            create_styled_label(model_type_frame, text=Lang.get("model_type")).pack(side="top")
            radio_scroll = CTkScrollableFrame(parent, orientation="horizontal", fg_color="transparent", height=50, scrollbar_button_color=PURPLE_ACCENT, scrollbar_button_hover_color=WHITE)
            if hasattr(radio_scroll, '_scrollbar'):
                try: radio_scroll._scrollbar.configure(corner_radius=50)
                except: pass
            if hasattr(radio_scroll, '_scrollbar_horizontal'):
                radio_scroll._scrollbar_horizontal.configure(width=12)
                try: radio_scroll._scrollbar_horizontal.configure(corner_radius=50)
                except: pass
            radio_scroll.pack(fill="x", pady=5)
            inner_frame = create_styled_frame(radio_scroll, fg_color="transparent")
            inner_frame.pack(fill="both", expand=True)
            provider_items = list(self.providers.items())
            if not provider_items: create_styled_label(radio_scroll, text=Lang.get("no_providers_found")).pack()
            for module_name, data in provider_items: display_name = data['name']; CTkRadioButton(inner_frame, text=display_name, variable=self.settings_vars['model_type'], value=module_name, command=self.toggle_model_frames, fg_color=PURPLE_ACCENT, font=FONT_REGULAR).pack(side="left", padx=5, pady=2)
            self.frames_container = create_styled_frame(parent)
            self.frames_container.pack(fill="x", pady=10)
            self.model_frames = {}
            self._create_specific_model_frames(self.frames_container)
            token_frame = create_styled_frame(parent)
            token_frame.pack(fill="x", pady=5)
            self.token_label = create_styled_label(token_frame, text=Lang.get("token_limit"))
            self.token_label.pack(side="left", padx=(0,10))
            self.token_entry = create_styled_entry(token_frame, textvariable=self.settings_vars['token_limit'])
            self.token_entry.pack(side="left", fill="x", expand=True)
            self.max_token_label = create_styled_label(token_frame, text="", text_color=WHITE)
            self.max_token_label.pack(side="left", padx=(5,0))
            self.update_max_token_label()
            if provider_items and not self.settings_vars['model_type'].get(): self.settings_vars['model_type'].set(provider_items[0][0])
            self.toggle_model_frames()
        def _create_specific_model_frames(self, container):
            try:
                for module_name, p_data in self.providers.items():
                    main_frame = create_styled_frame(container, fg_color=DARK_BG, border_color=WHITE, border_width=1, corner_radius=CORNER_RADIUS)
                    self.model_frames[module_name] = main_frame
                    main_frame.pack_propagate(False)
                    main_frame.configure(height=170)
                    params = p_data.get('params', [])
                    scroll_frame = CTkScrollableFrame(main_frame, scrollbar_button_color=PURPLE_ACCENT, scrollbar_button_hover_color=WHITE, fg_color="transparent", border_width=0, corner_radius=0)
                    scroll_frame.pack(fill="both", expand=True, padx=10, pady=1.5)
                    if hasattr(scroll_frame, '_scrollbar'):
                        scroll_frame._scrollbar.configure(width=12)
                        try: scroll_frame._scrollbar.configure(corner_radius=50)
                        except: pass
                    if hasattr(scroll_frame, '_scrollbar_horizontal'):
                        scroll_frame._scrollbar_horizontal.configure(width=12)
                        try: scroll_frame._scrollbar_horizontal.configure(corner_radius=50)
                        except: pass
                    content_parent = scroll_frame
                    self.provider_param_full_paths.setdefault(module_name, {})
                    self.settings_vars.setdefault(module_name, {})
                    for param in params:
                        def make_callback(m_name=module_name): return lambda p_name: self.on_param_change(p_name, m_name)
                        create_param_widget(content_parent, param, self.settings_vars[module_name], self.provider_param_full_paths[module_name], make_callback())
                    main_frame.pack(fill="x", padx=5, pady=5)
            except Exception as e: print(f"Error creating specific model frames: {e}")
        def on_param_change(self, param_name, provider_name):
            if param_name not in ["api_token", "token", "password"]: return
            val = self.settings_vars[provider_name][param_name].get()
            if val == "•••": return
            if param_name in ["api_token", "token"]:
                pwd_var = self.settings_vars[provider_name].get("password")
                if pwd_var and pwd_var.get() == "•••": pwd_var.set("")
            if param_name == "password":
                tok_var = self.settings_vars[provider_name].get("api_token")
                if not tok_var: tok_var = self.settings_vars[provider_name].get("token")
                if tok_var and tok_var.get() == "•••": tok_var.set("")
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
                        if param_name in provider_ui_vars:
                            if param_name in ["api_token", "token"]:
                                if value:
                                    self.real_token_values[current_provider_module] = value
                                    provider_ui_vars[param_name].set("•••")
                                else: provider_ui_vars[param_name].set("")
                            elif param_name == "password":
                                self.real_pwd_status[current_provider_module] = value
                                if value == "set": provider_ui_vars[param_name].set("•••")
                                else: provider_ui_vars[param_name].set("")
                            else: provider_ui_vars[param_name].set(value)
            except (ValueError, KeyError) as e: print(f"Warning: Could not parse provider params string: {params_str}. Error: {e}")
        def _build_connection_string(self) -> str:
            provider_module_name = self.settings_vars['model_type'].get()
            if not provider_module_name: return ""
            provider_data = self.providers.get(provider_module_name)
            if not provider_data: return ""
            parts = []
            self._last_plain_password = None
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
                if param_name in ["api_token", "token"]:
                    if value == "•••": value = self.real_token_values.get(provider_module_name, "")
                    else:
                        raw_token = value
                        raw_pwd = provider_ui_vars.get("password", tk.StringVar()).get()
                        if raw_pwd == "•••": raw_pwd = ""
                        if raw_pwd and raw_token: value = encryption_utils.encrypt_token(raw_token, raw_pwd); self._last_plain_password = raw_pwd
                        else: value = raw_token
                elif param_name == "password":
                    if value == "•••": value = self.real_pwd_status.get(provider_module_name, "")
                    else:
                        if value: self._last_plain_password = value; value = "set"
                        else: value = "empty"
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
                    if name == selected_type and frame.winfo_exists(): frame.pack(fill="x", padx=5, pady=5)
                    elif frame.winfo_exists(): frame.pack_forget()
                except (tk.TclError, AttributeError): continue
            self._load_provider_params_from_string()
        def update_max_token_label(self):
            if hasattr(self, 'max_token_label') and self.max_token_label.winfo_exists():
                max_limit = self.settings_vars.get('max_token_limit', tk.StringVar(value="8192")).get()
                try: max_int = int(max_limit); self.max_token_label.configure(text=f"{max_int}max")
                except: self.max_token_label.configure(text="")
        def enforce_token_limit(self):
            max_str = self.settings_vars.get('max_token_limit', tk.StringVar(value="8192")).get()
            cur_str = self.settings_vars['token_limit'].get().strip()
            try:
                max_limit = int(max_str) if max_str else 8192
                cur_limit = int(cur_str) if cur_str else max_limit
                if cur_limit < 1: cur_limit = max_limit
                elif cur_limit > max_limit: cur_limit = max_limit
                self.settings_vars['token_limit'].set(str(cur_limit))
            except ValueError: self.settings_vars['token_limit'].set(max_str if max_str else "8192")
            self.update_max_token_label()

    class BaseSettingsWindow(BaseTopLevel, DynamicModelUI):
        def __init__(self, master, backend, title_key="settings_title", geometry="500x420"):
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
            parent.grid_rowconfigure(0, weight=0)
            main_frame = create_styled_frame(parent)
            main_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
            main_frame.grid_columnconfigure(0, weight=1)
            self._create_model_ui(main_frame)
        def setup_chat_settings_tab(self, parent):
            parent.grid_columnconfigure(0, weight=1)
            parent.grid_rowconfigure(0, weight=1)
            scrollable_frame = create_scrollable_frame(parent, fg_color="transparent")
            scrollable_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
            scrollable_frame.grid_columnconfigure(0, weight=1)
            # Получаем метаданные из кэша
            metadata = self.backend.get_settings_metadata()
            # Строим UI динамически
            build_chat_settings_ui(scrollable_frame, self.settings_vars, metadata)
        def validate_model(self):
            model_type = self.settings_vars['model_type'].get()
            connection_string = self._build_connection_string()
            plain_password = getattr(self, '_last_plain_password', None)
            has_token_param = False
            has_pwd_param = False
            provider_data = self.providers.get(model_type, {})
            for param in provider_data.get('params', []):
                p_name = param['name'].lower()
                if "api_token" in p_name or "token" in p_name: has_token_param = True
                if "password" in p_name: has_pwd_param = True
            if has_token_param and not has_pwd_param: showerror(self, Lang.get("error", "Error"), "Провайдер не валиден: отсутствует параметр password при наличии api_token."); return
            valid, msg, max_tokens = self.backend.validate_model_settings(model_type, connection_string, plain_password)
            if valid:
                self.max_tokens = max_tokens
                self.validated = True
                showinfo(self, Lang.get("success"), msg)
                self.settings_vars['max_token_limit'].set(str(self.max_tokens))
                self.update_max_token_label()
                self.enforce_token_limit()
                self.token_label.configure(text=Lang.get("token_limit_info", max_tokens=self.max_tokens))
                if plain_password: encryption_utils.SESSION_PASSWORDS[model_type] = plain_password; self.valid_password = plain_password
                self.settings_vars['model_provider_params'].set(connection_string)
            else: self.validated = False; showerror(self, Lang.get("validation_error"), msg)
        def _get_default_settings(self):
            settings = self.backend.get_global_settings()
            if 'max_token_limit' not in settings: settings['max_token_limit'] = '8192'
            try:
                cur = int(settings.get('token_limit', '8192'))
                mx = int(settings['max_token_limit'])
                if cur > mx: settings['token_limit'] = str(mx)
            except: pass
            return {key: tk.StringVar(value=val) for key, val in settings.items()}

    class ChatApp(CTk):
        def __init__(self, backend):
            super().__init__(fg_color=DARK_BG)
            if sys.platform.startswith("linux"): self.withdraw()
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
            self._wraplength_update_pending = False
            self._last_available_width = 0
        @staticmethod
        def add_label_context_menu(master, widget):
            menu = tk.Menu(master, tearoff=0, bg=DARK_SECONDARY, fg=WHITE, relief="flat", borderwidth=0, font=(FONT_FAMILY, 8))
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
                    for chat_id in active_chats: self.terminate_chat_process(chat_id)
                    self.destroy()
                else: return
            else: self.destroy()
        def terminate_chat_process(self, chat_id):
            if chat_id in self.chat_processes:
                process = self.chat_processes[chat_id]
                if process.is_alive(): process.terminate(); process.join()
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
                if hasattr(chat_button, '_hover') and chat_button._hover: continue
                is_blinking = self.chat_blink_states.get(chat_id, False)
                if self.current_chat_id == chat_id: color = PURPLE_ACCENT
                elif is_blinking: color = ACTIVE_CHAT_COLOR
                else: color = "transparent"
                chat_button.configure(fg_color=color)
        def show_initial_settings(self): self.withdraw(); InitialSettingsWindow(self, self.backend)
        def setup_main_ui(self):
            for widget in self.winfo_children(): widget.destroy()
            self.grid_rowconfigure(0, weight=1)
            self.grid_columnconfigure(0, weight=0)
            self.grid_columnconfigure(1, weight=1)
            self.bind("<Control-o>", self.add_attachment)
            if sys.platform == "darwin": self.bind("<Command-o>", self.add_attachment)
            left_panel_container = create_styled_frame(self, width=200)
            left_panel_container.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
            left_panel_container.grid_rowconfigure(0, weight=1)
            left_panel_container.grid_rowconfigure(1, weight=0)
            chats_bordered_frame = create_styled_frame(left_panel_container, fg_color=DARK_BG, border_color=WHITE, border_width=1, corner_radius=CORNER_RADIUS)
            chats_bordered_frame.grid(row=0, column=0, sticky="nsew")
            self.chats_list_frame = CTkScrollableFrame(chats_bordered_frame, scrollbar_button_color=PURPLE_ACCENT, scrollbar_button_hover_color=WHITE, fg_color="transparent", border_width=0, corner_radius=0)
            self.chats_list_frame.pack(fill="both", expand=True, padx=10, pady=1.5)
            if hasattr(self.chats_list_frame, '_scrollbar'):
                self.chats_list_frame._scrollbar.configure(width=12)
                try: self.chats_list_frame._scrollbar.configure(corner_radius=50)
                except: pass
            if hasattr(self.chats_list_frame, '_scrollbar_horizontal'):
                self.chats_list_frame._scrollbar_horizontal.configure(width=12)
                try: self.chats_list_frame._scrollbar_horizontal.configure(corner_radius=50)
                except: pass
            bottom_buttons_frame = create_styled_frame(left_panel_container)
            bottom_buttons_frame.grid(row=1, column=0, sticky="ew", pady=(3,0))
            self.settings_btn = create_styled_button(bottom_buttons_frame, text="☰", command=self.open_settings, width=20, height=20)
            self.settings_btn.pack(side=tk.LEFT, padx=(0, 2))
            self.new_chat_btn = create_styled_button(bottom_buttons_frame, text="+↑", command=self.create_chat_window_show, width=20, height=20)
            self.new_chat_btn.pack(side=tk.LEFT, padx=(2, 0))
            self.control_buttons_frame = create_styled_frame(bottom_buttons_frame, fg_color="transparent")
            self.control_buttons_frame.pack(side=tk.LEFT, padx=(2, 0))
            self.stop_btn = create_styled_button(self.control_buttons_frame, text="◯", width=20, height=20, command=self.stop_chat)
            self.stop_btn.pack(side=tk.LEFT, padx=2)
            self.play_btn = create_styled_button(self.control_buttons_frame, text="ᗞ", width=20, height=20, command=self.resume_chat)
            self.play_btn.pack(side=tk.LEFT, padx=2)
            self.log_btn = create_styled_button(self.control_buttons_frame, text="log", width=20, height=20, command=self.open_log_window)
            self.log_btn.pack(side=tk.LEFT, padx=2)
            right_panel_container = create_styled_frame(self)
            right_panel_container.grid(row=0, column=1, sticky="nsew", padx=(0, 5), pady=5)
            right_panel_container.grid_columnconfigure(0, weight=1)
            right_panel_container.grid_rowconfigure(0, weight=1)
            right_panel_container.grid_rowconfigure(1, weight=0)
            self.messages_bordered_frame = create_styled_frame(right_panel_container, fg_color=DARK_BG, border_color=WHITE, border_width=1, corner_radius=CORNER_RADIUS)
            self.messages_bordered_frame.grid(row=0, column=0, sticky="nsew", pady=(0,5))
            self.messages_bordered_frame.grid_rowconfigure(0, weight=1)
            self.messages_bordered_frame.grid_columnconfigure(0, weight=1)
            self.messages_frame = CTkScrollableFrame(self.messages_bordered_frame, scrollbar_button_color=PURPLE_ACCENT, scrollbar_button_hover_color=WHITE, fg_color="transparent", border_width=0, corner_radius=0)
            self.messages_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=1.5)
            if hasattr(self.messages_frame, '_scrollbar'):
                self.messages_frame._scrollbar.configure(width=12)
                try: self.messages_frame._scrollbar.configure(corner_radius=50)
                except: pass
            if hasattr(self.messages_frame, '_scrollbar_horizontal'):
                self.messages_frame._scrollbar_horizontal.configure(width=12)
                try: self.messages_frame._scrollbar_horizontal.configure(corner_radius=50)
                except: pass
            self.messages_bordered_frame.bind("<Configure>", self._on_message_container_resize)
            self.input_outer_frame = create_styled_frame(right_panel_container)
            self.input_outer_frame.grid(row=1, column=0, sticky="ew")
            self.input_outer_frame.grid_columnconfigure(0, weight=0)
            self.input_outer_frame.grid_columnconfigure(1, weight=1)
            self.input_outer_frame.grid_columnconfigure(2, weight=0)
            self.left_bar_canvas = tk.Canvas(self.input_outer_frame, width=6, bg=DARK_BG, highlightthickness=0)
            self.left_bar_canvas.configure(height=45)
            self.left_bar_canvas.grid(row=0, column=0, sticky="nsw", padx=(0, 0))
            def draw_left_wall(event=None):
                self.left_bar_canvas.delete("all")
                h = self.left_bar_canvas.winfo_height()
                r = 6
                color = WHITE
                if h > r * 2:
                    self.left_bar_canvas.create_arc(0, 0, r*2, r*2, start=90, extent=90, style="arc", outline=color, width=1)
                    self.left_bar_canvas.create_line(0, r, 0, h - r, fill=color, width=1)
                    self.left_bar_canvas.create_arc(0, h - r*2, r*2, h, start=180, extent=90, style="arc", outline=color, width=1)
            self.left_bar_canvas.bind("<Configure>", draw_left_wall)
            self.input_text = CTkTextbox(
                self.input_outer_frame,
                corner_radius=0,
                border_width=0,
                fg_color="transparent",
                font=FONT_REGULAR,
                wrap="word",
                text_color=WHITE,
                scrollbar_button_color=PURPLE_ACCENT,
                scrollbar_button_hover_color=WHITE)
            if hasattr(self.input_text, '_scrollbar'):
                try: self.input_text._scrollbar.configure(corner_radius=50)
                except: pass
            if hasattr(self.input_text, '_scrollbar_horizontal'):
                self.input_text._scrollbar_horizontal.configure(width=12)
                try: self.input_text._scrollbar_horizontal.configure(corner_radius=50)
                except: pass
            self.input_text._textbox.configure(borderwidth=0, padx=0, pady=0, selectbackground=PURPLE_ACCENT)
            self.input_text.grid(row=0, column=1, sticky="nsew")
            self.input_text.bind("<Return>", self.on_enter_pressed)
            self.input_text.bind("<KeyRelease>", self.adjust_input_height, add=True)
            enhance_text_widget(self.input_text, on_change=self.adjust_input_height)
            self.adjust_input_height()
            self.right_controls_frame = create_styled_frame(self.input_outer_frame)
            self.right_controls_frame.grid(row=0, column=2, padx=(0,0), sticky="se")
            self.send_btn = create_styled_button(self.right_controls_frame, text="↑", width=20, height=20, command=self.send_message)
            self.send_btn.pack(side=tk.TOP)
            self.attach_btn = create_styled_button(self.right_controls_frame, text="+", width=20, height=20, command=self.add_attachment)
            self.attach_btn.pack(side=tk.TOP, pady=(5,0))
            self.load_chats()
            self.start_chat_blinking()
            self.update_chat_controls()
        def _on_mousewheel(self, event, canvas):
            if event.delta: canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            else:
                if event.num == 4: canvas.yview_scroll(-1, "units")
                elif event.num == 5: canvas.yview_scroll(1, "units")
        def update_chat_controls(self):
            self.stop_btn.pack_forget()
            self.play_btn.pack_forget()
            self.log_btn.pack_forget()
            if not self.current_chat_id:
                self.play_btn.pack(side=tk.LEFT, padx=2)
                self.play_btn.configure(state="disabled")
                return
            has_messages = len(self.backend.get_messages(self.current_chat_id)) > 0
            is_active = self.current_chat_id in self.chat_processes
            if is_active:
                self.stop_btn.pack(side=tk.LEFT, padx=2)
                self.log_btn.pack(side=tk.LEFT, padx=2)
                self.play_btn.pack_forget()
            else:
                self.play_btn.pack(side=tk.LEFT, padx=2)
                self.play_btn.configure(state="normal" if has_messages else "disabled")
                self.log_btn.pack_forget()
        def stop_chat(self):
            if self.current_chat_id and self.current_chat_id in self.chat_processes: self.terminate_chat_process(self.current_chat_id); self.update_chat_controls()
        def resume_chat(self):
            if self.current_chat_id and self.current_chat_id not in self.chat_processes: self.start_chat_process(self.current_chat_id)
            self.update_chat_controls()
        def adjust_input_height(self, event=None):
            text_widget = self.input_text._textbox
            display_lines = text_widget.count("1.0", "end-1c", "displaylines")
            if isinstance(display_lines, tuple): display_lines = display_lines[0]
            if display_lines is None: display_lines = 1
            display_lines = max(1, display_lines)
            max_lines = 6
            lines_to_show = min(display_lines, max_lines)
            new_height = lines_to_show * 20 + 10
            self.input_text.configure(height=new_height)
        def on_enter_pressed(self, event):
            if not event.state & 0x1: self.send_message(); return "break"
            return None
        def add_attachment(self, event=None):
            files = filedialog.askopenfilenames()
            if files: self.attachments.extend(Path(file) for file in files); self.show_attachments()
        def show_attachments(self):
            if hasattr(self, 'attachment_overlay_frame') and self.attachment_overlay_frame and self.attachment_overlay_frame.winfo_exists(): self.attachment_overlay_frame.destroy()
            self.attachment_overlay_frame = None
            if not self.attachments: return
            self.attachment_overlay_frame = create_styled_frame(self.messages_bordered_frame, fg_color=DARK_BG, border_color=PURPLE_ACCENT, border_width=1, corner_radius=CORNER_RADIUS)
            self.attachment_overlay_frame.place(relx=0.5, y=5, anchor='n', relwidth=0.75)
            header_text = f"{Lang.get('attachments')}"
            inner_frame = create_styled_frame(self.attachment_overlay_frame, fg_color=DARK_SECONDARY, corner_radius=CORNER_RADIUS)
            inner_frame.pack(fill="both", expand=True, padx=0, pady=0)
            scrollable_container = create_scrollable_frame(inner_frame, fg_color="transparent", label_text=header_text, label_text_color=WHITE)
            scrollable_container.pack(fill="both", expand=True, padx=5, pady=5)
            for i, attachment in enumerate(self.attachments):
                att_frame = create_styled_frame(scrollable_container)
                att_frame.pack(fill="x", pady=2, padx=2)
                CTkButton(att_frame, text="X", width=22, height=22, fg_color="transparent", hover_color=PURPLE_ACCENT, text_color=WHITE, command=lambda idx=i: self.remove_attachment(idx)).pack(side=tk.LEFT)
                create_styled_label(att_frame, text=attachment.name, font=FONT_REGULAR, anchor="w").pack(side=tk.LEFT, padx=5, expand=True, fill="x")
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
                chat_button._hover = False
                chat_button.bind("<Enter>", lambda e: setattr(e.widget, '_hover', True))
                chat_button.bind("<Leave>", lambda e: setattr(e.widget, '_hover', False))
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
                    files_button = CTkButton(row_frame, text="ƒ", width=20, height=20, fg_color="transparent", hover_color=PURPLE_ACCENT, corner_radius=50, command=lambda c_id=chat["id"]: self.open_folder(c_id, "files"))
                    files_button.grid(row=0, column=column_offset, padx=(2, 0))
                    column_offset += 1
                if has_console_folders:
                    console_button = CTkButton(row_frame, text=">_", width=20, height=20, fg_color="transparent", hover_color=PURPLE_ACCENT, corner_radius=50, command=lambda c_id=chat["id"]: self.open_folder(c_id, "console_folders"))
                    console_button.grid(row=0, column=column_offset, padx=(2, 0))
                    column_offset += 1
                if has_reports:
                    reports_button = CTkButton(row_frame, text="📄", width=20, height=20, fg_color="transparent", hover_color=PURPLE_ACCENT, corner_radius=50, command=lambda c_id=chat["id"]: self.open_folder(c_id, "reports"))
                    reports_button.grid(row=0, column=column_offset, padx=(2, 0))
                    column_offset += 1
                if has_results:
                    results_button = CTkButton(row_frame, text="✔", width=20, height=20, fg_color="transparent", hover_color=PURPLE_ACCENT, corner_radius=50, command=lambda c_id=chat["id"]: self.open_folder(c_id, "results"))
                    results_button.grid(row=0, column=column_offset, padx=(2, 0))
                    column_offset += 1
                delete_button = CTkButton(row_frame, text="✘", width=20, height=20, fg_color="transparent", hover_color=PURPLE_ACCENT, corner_radius=50, command=lambda c_id=chat["id"]: self.delete_selected_chat(c_id))
                delete_button.grid(row=0, column=column_offset, padx=(2,0))
            self.chats_list_frame.update_idletasks()
            if hasattr(self.chats_list_frame, '_parent_canvas'): self.chats_list_frame._parent_canvas.configure(scrollregion=self.chats_list_frame._parent_canvas.bbox("all"))
            chat_ids = [c["id"] for c in chats]
            if current_selection not in chat_ids: self.on_chat_select(first_chat_id) if first_chat_id else self.clear_chat_view()
            self.update_chat_list_colors()
        def open_folder(self, chat_id, folder_name):
            folder_path = Path(resource_path(os.path.join("data", "chats", chat_id, folder_name)))
            if not folder_path.exists(): showinfo(self, Lang.get("info"), Lang.get("folder_not_found", folder_name=folder_name)); return
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
                if chat['id'] == chat_id: chat_name = chat['name']; break
            if chat_id in self.chat_processes: showwarning(self, Lang.get("active_chat_delete_title"), Lang.get("active_chat_delete_message", chat_name=chat_name)); return
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
            self.after(0, lambda: self.messages_frame._parent_canvas.yview_moveto(1))
        def copy_text_to_clipboard(self, text): self.clipboard_clear(); self.clipboard_append(text)
        def add_message_to_ui(self, text, is_my, is_question=False, attachments=None):
            bubble, msg_text_widget = create_chat_message_bubble(self.messages_frame, text, is_my, attachments, is_question)
            def create_context_menu(event):
                menu = tk.Menu(self, tearoff=0, bg=DARK_SECONDARY, fg=WHITE, font=(FONT_FAMILY, 8))
                menu.add_command(label=Lang.get("copy"), command=lambda: self.copy_text_to_clipboard(text))
                menu.tk_popup(event.x_root, event.y_root)
            msg_text_widget.bind("<Button-3>", create_context_menu)
            if attachments:
                att_container = create_styled_frame(bubble)
                att_container.pack(fill=tk.X, pady=(8, 5), padx=5)
                for att in attachments:
                    att_frame = create_styled_frame(att_container)
                    att_frame.pack(fill=tk.X, pady=1, anchor='w')
                    create_styled_label(att_frame, text=Path(att).name).pack(side=tk.LEFT)
                    hover_color = PURPLE_ACCENT if is_my else DARK_BG
                    CTkButton(att_frame, text="📂", font=FONT_REGULAR, width=25, height=25, fg_color="transparent", hover_color=hover_color, command=lambda a=att: self.open_attachment(a)).pack(side=tk.RIGHT)
            self.messages_frame.update_idletasks()
            if hasattr(self.messages_frame, '_parent_canvas'): self.messages_frame._parent_canvas.configure(scrollregion=self.messages_frame._parent_canvas.bbox("all"))
            self.after(0, lambda: self.messages_frame._parent_canvas.yview_moveto(1.0))
            self._update_message_wraplengths(force=True)
        def open_attachment(self, file_path):
            try:
                if sys.platform == "win32": os.startfile(file_path)
                elif sys.platform == "darwin": subprocess.run(["open", file_path])
                else: subprocess.run(["xdg-open", file_path])
            except Exception as e: showerror(self, Lang.get("error"), Lang.get("attachment_open_error", e=str(e)))
        def send_message(self):
            text = self.input_text.get("1.0", "end-1c").strip()
            if not text and not self.attachments: return
            if not self.current_chat_id: self.create_chat_window_show(); return
            attachments_paths = [str(a.resolve()) for a in self.attachments] if self.attachments else []
            if self.waiting_for_answer.get(self.current_chat_id): self.waiting_for_answer[self.current_chat_id] = False
            if self.backend.add_message(self.current_chat_id, text, True, attachments_paths):
                self.attachments.clear()
                self.show_attachments()
                self.add_message_to_ui(text, True, attachments=attachments_paths)
                self.input_text.delete("1.0", "end")
                self.adjust_input_height()
                message_data = {'text': text, 'attachments': attachments_paths or None, 'command': 'answer_user' if self.waiting_for_answer.get(self.current_chat_id) else None}
                if self.current_chat_id in self.input_queues: self.input_queues[self.current_chat_id].put(message_data)
                if self.current_chat_id not in self.chat_processes: self.resume_chat()
                else: self.update_chat_controls()
        def start_chat_process(self, chat_id):
            chat_settings = self.backend.get_chat_settings(chat_id)
            params_str = chat_settings.get('model_provider_params', '')
            model_type = chat_settings.get('model_type', '')
            params_map = dict(part.split('=', 1) for part in params_str.split(';') if '=' in part)
            if params_map.get('password') == 'set':
                pwd = encryption_utils.SESSION_PASSWORDS.get(chat_id)
                if not pwd and model_type in encryption_utils.SESSION_PASSWORDS: pwd = encryption_utils.SESSION_PASSWORDS.get(model_type)
                encrypted_token = params_map.get('api_token') or params_map.get('token')
                valid_cached = False
                if pwd and encrypted_token:
                    try:
                        encryption_utils.decrypt_token(encrypted_token, pwd)
                        valid_cached = True
                        encryption_utils.SESSION_PASSWORDS[chat_id] = pwd
                    except Exception: valid_cached = False
                if not valid_cached:
                    while True:
                        pwd_input = encryption_utils.ask_for_password(self)
                        if not pwd_input: return
                        try:
                            if encrypted_token: encryption_utils.decrypt_token(encrypted_token, pwd_input)
                            encryption_utils.SESSION_PASSWORDS[chat_id] = pwd_input
                            break
                        except Exception: showerror(self, Lang.get("error", "Error"), "Неверный пароль!")
            input_queue = multiprocessing.Queue()
            output_queue = multiprocessing.Queue()
            log_queue = multiprocessing.Queue()
            current_passwords = encryption_utils.SESSION_PASSWORDS.copy()
            from cross_gpt import initialize_work
            p = multiprocessing.Process(target=initialize_work, args=(get_base_dir(), chat_id, input_queue, output_queue, log_queue, current_passwords))
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
                    if isinstance(response, dict):
                        message_text = response.get('text', '')
                        attachments = response.get('attachments')
                        is_question = response.get('command') == 'ask_user'
                    else:
                        message_text = str(response)
                        attachments = None
                        is_question = False
                    if not message_text: continue
                    self.backend.add_message(chat_id, message_text, False, attachments)
                    if chat_id == self.current_chat_id: self.add_message_to_ui(message_text, False, is_question=is_question, attachments=attachments)
                    else: self.chat_blink_states[chat_id] = True
                    if not self.focus_get(): self.flash_window()
            except queue.Empty: pass
            if chat_id in self.chat_processes and self.chat_processes[chat_id].is_alive(): self.after(500, lambda c=chat_id: self.check_chat_responses(c))
            else:
                if chat_id in self.active_chats:
                    self._cleanup_chat_process_data(chat_id)
                    if chat_id == self.current_chat_id: self.update_chat_controls()
        def create_chat_window_show(self):
            if self.create_chat_window and self.create_chat_window.winfo_exists(): self.create_chat_window.destroy()
            self.create_chat_window = CreateChatWindow(self, self.backend)
        def open_settings(self):
            if self.settings_window and self.settings_window.winfo_exists(): self.settings_window.lift(); return
            self.settings_window = SettingsWindow(self, self.backend)
        def open_log_window(self):
            if not self.current_chat_id or self.current_chat_id not in self.chat_processes: return
            if self.current_chat_id in self.log_windows and self.log_windows[self.current_chat_id].winfo_exists(): self.log_windows[self.current_chat_id].lift(); return
            log_queue = self.log_queues.get(self.current_chat_id)
            if log_queue: log_win = LogWindow(self, self.current_chat_id, log_queue); self.log_windows[self.current_chat_id] = log_win
        def _on_message_container_resize(self, event=None):
            if not hasattr(self, '_wraplength_update_pending'): return # Защита от вызова с неверным self (например, корневым окном)
            if self._wraplength_update_pending: return
            self._wraplength_update_pending = True
            self.after(50, self._update_message_wraplengths)
        def _update_message_wraplengths(self, force=False):
            self._wraplength_update_pending = False
            if not hasattr(self, 'messages_frame') or not self.messages_frame.winfo_exists(): return
            if not hasattr(self, 'messages_bordered_frame') or not self.messages_bordered_frame.winfo_exists(): return
            available_width = self.messages_bordered_frame.winfo_width() - 70
            if available_width < 50: return
            if not force and available_width == self._last_available_width: return
            self._last_available_width = available_width
            for row_frame in self.messages_frame.winfo_children():
                try:
                    bubble = next((w for w in row_frame.winfo_children() if isinstance(w, CTkFrame)), None)
                    if not bubble: continue
                    for child in bubble.winfo_children():
                        if isinstance(child, CTkLabel): child.configure(wraplength=available_width)
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
            message_label = create_styled_label(main_frame, text=message, wraplength=width - 60, justify="left", font=FONT_REGULAR)
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
            if self.result is None: is_yesno = any(b[0].lower() == 'no' for b in []); self.result = False if is_yesno else None
            super().on_close()

    class InitialSettingsWindow(BaseTopLevel, DynamicModelUI):
        def __init__(self, master, backend):
            BaseTopLevel.__init__(self, master)
            DynamicModelUI.__init__(self)
            self.master = master
            self.backend = backend
            self.title("Setup")
            self.geometry("500x450")
            self.minsize(500, 450)
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
            lang_combo = CTkOptionMenu(lang_frame, variable=self.lang_var, values=list(Lang.available_languages.keys()), **OPTIONMENU_THEME)
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
            if 'max_token_limit' not in settings: settings['max_token_limit'] = '8192'
            return {key: tk.StringVar(value=val) for key, val in settings.items()}
        def validate_model(self):
            model_type = self.settings_vars['model_type'].get()
            connection_string = self._build_connection_string()
            plain_password = getattr(self, '_last_plain_password', None)
            has_token_param = False
            has_pwd_param = False
            provider_data = self.providers.get(model_type, {})
            for param in provider_data.get('params', []):
                p_name = param['name'].lower()
                if "api_token" in p_name or "token" in p_name: has_token_param = True
                if "password" in p_name: has_pwd_param = True
            if has_token_param and not has_pwd_param: showerror(self, Lang.get("error", "Error"), "Провайдер не валиден: отсутствует параметр password при наличии api_token."); return
            valid, msg, max_tokens = self.backend.validate_model_settings(model_type, connection_string, plain_password)
            if valid:
                self.max_tokens = max_tokens
                self.validated = True
                showinfo(self, Lang.get("success"), msg)
                self.settings_vars['max_token_limit'].set(str(self.max_tokens))
                self.update_max_token_label()
                self.enforce_token_limit()
                self.token_label.configure(text=Lang.get("token_limit_info", max_tokens=self.max_tokens))
                if plain_password: encryption_utils.SESSION_PASSWORDS[model_type] = plain_password; self.valid_password = plain_password
                self.settings_vars['model_provider_params'].set(connection_string)
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
            self.enforce_token_limit()
            try:
                token_limit = int(self.settings_vars['token_limit'].get())
                max_limit = int(self.settings_vars['max_token_limit'].get())
                if not (1 <= token_limit <= max_limit): raise ValueError
            except (ValueError, TypeError): showerror(self, Lang.get("error"), Lang.get("token_limit_info", max_tokens=self.max_tokens)); return
            settings_to_save = {
                'language': self.lang_var.get(),
                'model_type': self.settings_vars['model_type'].get(),
                'token_limit': self.settings_vars['token_limit'].get(),
                'max_token_limit': self.settings_vars['max_token_limit'].get(),
                'model_provider_params': self.settings_vars['model_provider_params'].get()}
            self.backend.update_global_settings(settings_to_save)
            self.on_close()
        def on_close(self):
            super().on_close()
            if self.master.winfo_exists():
                if self.backend.is_main_config_complete(): self.master.deiconify(); self.master.setup_main_ui()
                else: self.master.destroy()
    class SettingsWindow(BaseSettingsWindow):
        def __init__(self, master, backend):
            super().__init__(master, backend, "settings_title", "500x450")
            self.bind("<Control-o>", self.add_custom_mod)
            if sys.platform == "darwin": self.bind("<Command-o>", self.add_custom_mod)
            tabview = CTkTabview(self, **TAB_VIEW_THEME)
            tabview.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
            main_tab = tabview.add(Lang.get("tab_main"))
            chat_settings_tab = tabview.add(Lang.get("tab_chat_settings"))
            mods_tab = tabview.add(Lang.get("tab_modules"))
            self.setup_main_tab(main_tab)
            self.setup_chat_settings_tab(chat_settings_tab)
            self.setup_mods_tab(mods_tab)
        def setup_main_tab(self, parent):
            parent.grid_columnconfigure(0, weight=1)
            parent.grid_rowconfigure(0, weight=1)
            parent.grid_rowconfigure(1, weight=0)
            self.max_tokens = int(self.backend.get_global_settings().get("token_limit", 8192))
            self.validated = True
            self.settings_vars = self._get_default_settings()
            self.original_language = self.settings_vars['language'].get()
            self.original_model_type = self.settings_vars['model_type'].get()
            self.original_connection_string = self.settings_vars['model_provider_params'].get()
            main_frame = create_styled_frame(parent)
            main_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
            main_frame.grid_columnconfigure(0, weight=1)
            lang_frame = create_styled_frame(main_frame)
            lang_frame.pack(fill='x', pady=5)
            lang_frame.grid_columnconfigure(0, weight=1)
            lang_frame.grid_columnconfigure(2, weight=1)
            lang_content = create_styled_frame(lang_frame)
            lang_content.grid(row=0, column=1, sticky="ew")
            create_styled_label(lang_content, text=f"{Lang.get('language')}:").grid(row=0, column=0, padx=(0, 10), sticky="w")
            self.lang_combo = CTkOptionMenu(lang_content, variable=self.settings_vars['language'], values=list(Lang.available_languages.keys()), **OPTIONMENU_THEME)
            self.lang_combo.grid(row=0, column=1, sticky="ew")
            lang_content.grid_columnconfigure(1, weight=1)
            self._create_model_ui(main_frame)
            btn_frame = create_styled_frame(parent)
            btn_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))
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
                self.enforce_token_limit()
                try:
                    token_limit = int(self.settings_vars['token_limit'].get())
                    max_limit = int(self.settings_vars['max_token_limit'].get())
                    if not (1 <= token_limit <= max_limit): raise ValueError
                except (ValueError, TypeError): showerror(self.master, Lang.get("error"), Lang.get("token_limit_info", max_tokens=self.max_tokens)); return
                # Собираем значения всех настроек, включая динамические из метаданных
                settings_to_save = {
                    'language': self.settings_vars['language'].get(),
                    'model_type': self.settings_vars['model_type'].get(),
                    'token_limit': self.settings_vars['token_limit'].get(),
                    'max_token_limit': self.settings_vars['max_token_limit'].get(),
                    'model_provider_params': self.settings_vars['model_provider_params'].get(),}
                # Добавляем все настройки чата из метаданных
                metadata = self.backend.get_settings_metadata()
                for key in metadata:
                    if key in self.settings_vars: settings_to_save[key] = self.settings_vars[key].get()
                if settings_changed: settings_to_save['model_provider_params'] = self._build_connection_string()
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
            self.scrollable_frame.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
            self.rebuild_mods_list()
            create_styled_button(parent, text=Lang.get("add_module"), command=self.add_custom_mod).grid(row=1, column=0, pady=5, padx=5)
        def rebuild_mods_list(self):
            for widget in self.scrollable_frame.winfo_children(): widget.destroy()
            self.scrollable_frame.grid_columnconfigure(0, weight=1)
            module_manager = ModuleManager()
            default_mods = module_manager.get_default_modules()
            custom_mods = module_manager.get_custom_modules()
            if default_mods:
                create_styled_label(self.scrollable_frame, text=Lang.get("system_modules"), font=FONT_REGULAR).pack(anchor="w", padx=5, pady=(5,2))
                for mod in default_mods: self.create_mod_ui(self.scrollable_frame, mod, is_default=True)
            create_styled_label(self.scrollable_frame, text=Lang.get("global_custom_modules"), font=FONT_REGULAR).pack(anchor="w", padx=5, pady=(10,2))
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
        def toggle_default_mod(self, mod_id, enabled):
            self.backend.update_default_mod_enabled(mod_id, enabled)
            ModuleManager().load_modules(self.backend, reload_m=True)
        def remove_custom_mod(self, mod_id):
            if askyesno(self, Lang.get("warning"), Lang.get("remove_module_confirm")):
                self.backend.remove_custom_mod(mod_id)
                ModuleManager().update_custom_modules(self.backend)
                self.rebuild_mods_list()
        def add_custom_mod(self, event=None):
            path = filedialog.askopenfilename(filetypes=[(Lang.get("python_files"), "*.py")])
            if not path: return
            try:
                self.backend.add_custom_mod(str(Path(path).resolve()))
                ModuleManager().update_custom_modules(self.backend)
                self.rebuild_mods_list()
            except ValueError as e: showerror(self, Lang.get("error"), str(e))

    class LogWindow(BaseTopLevel):
        MAX_LOG_MESSAGES = 10
        def __init__(self, master, chat_id, log_queue):
            super().__init__(master, fg_color=DARK_BG)
            self.geometry("400x200")
            self.minsize(400, 200)
            self.master = master
            self.chat_id = chat_id
            self.log_queue = log_queue
            chat_name = next((chat['name'] for chat in self.master.backend.get_chats() if chat['id'] == chat_id), chat_id)
            self.title(f"log {chat_name}")
            self.grid_rowconfigure(0, weight=1)
            self.grid_columnconfigure(0, weight=1)
            self.messages_frame = CTkScrollableFrame(
                self,
                scrollbar_button_color=PURPLE_ACCENT,
                scrollbar_button_hover_color=WHITE,
                fg_color="transparent",
                border_width=0,
                corner_radius=0)
            self.messages_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
            if hasattr(self.messages_frame, '_scrollbar'):
                self.messages_frame._scrollbar.configure(width=12)
                try: self.messages_frame._scrollbar.configure(corner_radius=50)
                except: pass
            self.log_message_widgets = []
            self.check_log_queue()
        def setup_and_center(self):
            setup_icon(self)
            self.lift()
            center_window(self)
        def add_log_message_to_ui(self, text):
            try:
                if len(self.log_message_widgets) >= self.MAX_LOG_MESSAGES:
                    for widget in self.messages_frame.winfo_children(): widget.destroy()
                    self.log_message_widgets.clear()
                    bubble, msg_text = create_chat_message_bubble(self.messages_frame, text, is_my=False, is_question=False)
                    setup_message_wraplength(msg_text, self.messages_frame)
                    self.log_message_widgets.append(bubble)
                    def copy_text(): self.master.clipboard_clear(); self.master.clipboard_append(text)
                    menu = tk.Menu(self, tearoff=0, bg=DARK_SECONDARY, fg=WHITE, font=(FONT_FAMILY, 8))
                    menu.add_command(label=Lang.get("copy"), command=copy_text)
                    msg_text.bind("<Button-3>", lambda e: menu.tk_popup(e.x_root, e.y_root))
                    if sys.platform == "darwin": msg_text.bind("<Button-2>", lambda e: menu.tk_popup(e.x_root, e.y_root))
                else:
                    bubble, msg_text = create_chat_message_bubble(self.messages_frame, text, is_my=False, is_question=False)
                    setup_message_wraplength(msg_text, self.messages_frame)
                    def copy_text():
                        self.master.clipboard_clear()
                        self.master.clipboard_append(text)
                    menu = tk.Menu(self, tearoff=0, bg=DARK_SECONDARY, fg=WHITE, font=(FONT_FAMILY, 8))
                    menu.add_command(label=Lang.get("copy"), command=copy_text)
                    msg_text.bind("<Button-3>", lambda e: menu.tk_popup(e.x_root, e.y_root))
                    if sys.platform == "darwin": msg_text.bind("<Button-2>", lambda e: menu.tk_popup(e.x_root, e.y_root))
                    self.log_message_widgets.append(bubble)
                self.messages_frame.update_idletasks()
                canvas = self.messages_frame._parent_canvas
                canvas.configure(scrollregion=canvas.bbox("all"))
                canvas.yview_moveto(1.0)
            except Exception as e: print(f"Ошибка при добавлении лога в UI: {e}")
        def check_log_queue(self):
            try:
                while True:
                    msg = self.log_queue.get_nowait()
                    self.add_log_message_to_ui(msg)
            except queue.Empty: pass
            except Exception as e: print(f"Ошибка в check_log_queue: {e}")
            if self.winfo_exists(): self.after(250, self.check_log_queue)
    class CreateChatWindow(BaseSettingsWindow):
        def __init__(self, master, backend):
            super().__init__(master, backend, "create_chat_title", "500x550")
            self.bind("<Control-o>", self.add_new_local_mod)
            if sys.platform == "darwin": self.bind("<Command-o>", self.add_new_local_mod)
            module_manager = ModuleManager()
            self.custom_mods_for_chat = module_manager.get_custom_modules().copy()
            self.newly_added_mods = []
            self.max_tokens = int(self.backend.get_global_settings().get("token_limit", 8192))
            self.validated = True
            self.settings_vars = self._get_default_settings()
            self.original_model_type = self.settings_vars['model_type'].get()
            self.original_connection_string = self.settings_vars['model_provider_params'].get()
            self.grid_columnconfigure(0, weight=1)
            self.grid_rowconfigure(1, weight=1)
            top_frame = create_styled_frame(self)
            top_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
            create_styled_label(top_frame, text=Lang.get("chat_name")).pack(side='left', padx=(0,10))
            e = create_styled_entry(top_frame, textvariable=self.settings_vars['chat_name'])
            e.pack(fill='x', expand=True)
            tabview = CTkTabview(self, **TAB_VIEW_THEME)
            tabview.grid(row=1, column=0, sticky="nsew", padx=5, pady=2)
            model_tab = tabview.add(Lang.get("tab_model"))
            chat_tab = tabview.add(Lang.get("tab_chat_settings"))
            mods_tab = tabview.add(Lang.get("tab_modules"))
            self.setup_model_tab(model_tab)
            self.setup_chat_settings_tab(chat_tab)
            self.setup_mods_tab(mods_tab)
            bottom_frame = create_styled_frame(self)
            bottom_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=(5, 5))
            self.create_btn = create_styled_button(bottom_frame, text=Lang.get("create"), command=self.create_chat_finalize)
            self.create_btn.pack(side='left')
            create_styled_button(bottom_frame, text=Lang.get("validate_model"), command=self.validate_model).pack(side='left', padx=5)
            create_styled_button(bottom_frame, text=Lang.get("cancel"), command=self.destroy).pack(side='left', padx=5)
            self._load_provider_params_from_string()
        def _get_default_settings(self):
            settings = self.backend.get_global_settings()
            s_vars = {key: tk.StringVar(value=val) for key, val in settings.items()}
            s_vars['chat_name'] = tk.StringVar(value=self.backend.generate_id(4))
            module_manager = ModuleManager()
            default_mods = module_manager.get_default_modules()
            s_vars['default_mods'] = { mod['id']: tk.BooleanVar(value=mod['enabled']) for mod in default_mods }
            return s_vars
        def setup_mods_tab(self, parent):
            parent.grid_rowconfigure(0, weight=1)
            parent.grid_columnconfigure(0, weight=1)
            self.mods_scrollable_frame = create_scrollable_frame(parent, fg_color="transparent")
            self.mods_scrollable_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
            self.rebuild_mods_list()
        def rebuild_mods_list(self):
            for widget in self.mods_scrollable_frame.winfo_children(): widget.destroy()
            self.mods_scrollable_frame.grid_columnconfigure(0, weight=1)
            module_manager = ModuleManager()
            default_mods = module_manager.get_default_modules()
            if default_mods:
                create_styled_label(self.mods_scrollable_frame, text=Lang.get("system_modules"), font=FONT_REGULAR).pack(anchor="w", padx=5, pady=(5,2))
                for mod in default_mods: self.create_mod_ui(self.mods_scrollable_frame, mod, "default")
            if self.custom_mods_for_chat:
                create_styled_label(self.mods_scrollable_frame, text=Lang.get("global_custom_modules"), font=FONT_REGULAR).pack(anchor="w", padx=5, pady=(10,2))
                for mod in self.custom_mods_for_chat: self.create_mod_ui(self.mods_scrollable_frame, mod, "global_custom")
            header_frame = create_styled_frame(self.mods_scrollable_frame)
            header_frame.pack(fill='x', pady=(10,2))
            create_styled_label(header_frame, text=Lang.get("chat_specific_modules"), font=FONT_REGULAR).pack(side='left', anchor="w", padx=5)
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
        def remove_mod_from_chat_list(self, mod_id_to_remove, mod_list): mod_list[:] = [m for m in mod_list if m.get("id") != mod_id_to_remove]; self.rebuild_mods_list()
        def add_new_local_mod(self, event=None):
            path_str = filedialog.askopenfilename(filetypes=[(Lang.get("python_files"), "*.py")])
            if not path_str: return
            path = str(Path(path_str).resolve())
            valid, msg = ModuleValidator.validate_module(path)
            if not valid: showerror(self, Lang.get("error"), msg); return
            name, description = self.backend._get_localized_doc(Path(path), lang=Lang.current_language)
            new_mod = {"id": self.backend.generate_id(6), "name": name, "description": description, "adress": path}
            self.newly_added_mods.append(new_mod)
            self.rebuild_mods_list()
        def create_chat_finalize(self):
            chat_name = self.settings_vars['chat_name'].get().strip()
            if not chat_name: showerror(self, Lang.get("error"), Lang.get("enter_chat_name")); return
            current_model_type = self.settings_vars['model_type'].get()
            current_connection_string = self._build_connection_string()
            settings_changed = (current_model_type != self.original_model_type or current_connection_string != self.original_connection_string)
            if not self.validated and settings_changed:
                if not askyesno(self, Lang.get("warning"), Lang.get("model_not_validated_continue")): return
            self.enforce_token_limit()
            try:
                token_limit = int(self.settings_vars['token_limit'].get())
                max_limit = int(self.settings_vars['max_token_limit'].get())
                if not (1 <= token_limit <= max_limit): raise ValueError
            except (ValueError, TypeError): showerror(self, Lang.get("error"), Lang.get("token_limit_info", max_tokens=self.max_tokens)); return
            model_config = {'model_type': self.settings_vars['model_type'].get(), 'model_provider_params': self.settings_vars['model_provider_params'].get(), 'token_limit': self.settings_vars['token_limit'].get()}
            if settings_changed: model_config['model_provider_params'] = self._build_connection_string()
            # Собираем настройки чата динамически из метаданных
            metadata = self.backend.get_settings_metadata()
            chat_config = {"language": Lang.current_language}
            for key in metadata:
                if key in self.settings_vars: chat_config[key] = self.settings_vars[key].get()
            module_manager = ModuleManager()
            default_mods = module_manager.get_default_modules()
            default_mods_config = {mid: var.get() for mid, var in self.settings_vars['default_mods'].items()}
            final_custom_mods = self.custom_mods_for_chat + self.newly_added_mods
            settings_bundle = {"model_config": model_config, "chat_config": chat_config, "default_mods_config": default_mods_config, "custom_mods_list": final_custom_mods}
            self.on_close()
            chat_data = self.backend.create_chat(chat_name, settings_bundle)
            if not chat_data: showerror(self.master, Lang.get("error"), Lang.get("chat_name_exists")); return
            if hasattr(self, 'valid_password') and self.valid_password: encryption_utils.SESSION_PASSWORDS[chat_data["id"]] = self.valid_password
            elif model_config['model_type'] in encryption_utils.SESSION_PASSWORDS: encryption_utils.SESSION_PASSWORDS[chat_data["id"]] = encryption_utils.SESSION_PASSWORDS[model_config['model_type']]
            self.master.load_chats()
            self.master.on_chat_select(chat_data["id"])
    # ------------------------------------------------------------
    # 5. Инициализация приложения
    # ------------------------------------------------------------
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
        if sys.platform.startswith("linux"): app.deiconify()
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

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app_is_ready_event = multiprocessing.Event() # Импортируем текущий модуль как ui (для запуска процесса)
    import ui
    main_app_process = multiprocessing.Process(target=ui.run_main_app, args=(app_is_ready_event,))
    main_app_process.start()
    show_splash(app_is_ready_event)
    sys.exit(0)