import tkinter as tk
from PIL import Image, ImageTk
import multiprocessing
import sys
import os
import json

# ====== БАЗОВЫЕ ПУТИ (остаются глобальными для заставки и других вызовов) ======
def get_base_dir():
    if getattr(sys, 'frozen', False): return os.path.dirname(os.path.abspath(sys.executable))
    else: return os.path.dirname(os.path.abspath(__file__))

def resource_path(relative_path): return os.path.join(get_base_dir(), relative_path)

def bundled_image_models_available():
    """Local BLIP+EasyOCR under data/models (optional installer component)."""
    try:
        from info_loaders import bundled_image_models_available as _check
        return bool(_check())
    except Exception:
        root = get_base_dir()
        blip = os.path.join(root, "data", "models", "blip")
        easy = os.path.join(root, "data", "models", "easyocr")
        blip_ok = (
            os.path.isfile(os.path.join(blip, "config.json"))
            or os.path.isfile(os.path.join(blip, "model.safetensors"))
        )
        easy_ok = (
            os.path.isfile(os.path.join(easy, "craft_mlt_25k.pth"))
            and os.path.isfile(os.path.join(easy, "cyrillic_g2.pth"))
        )
        return blip_ok and easy_ok

def suggest_default_chats_dir():
    """Windows: D:/Milana/chats если D: есть; иначе data/chats. Mac/Linux: data/chats."""
    default_local = resource_path(os.path.join("data", "chats"))
    if sys.platform == "win32":
        for drive in ("D:\\", "D:/"):
            if os.path.exists(drive):
                return os.path.join(drive, "Milana", "chats")
    return default_local

def normalize_chats_dir(path_value):
    """Нормализует путь к корню чатов; пустой → suggest_default_chats_dir()."""
    if path_value is None or str(path_value).strip() == "":
        return suggest_default_chats_dir()
    p = os.path.expanduser(str(path_value).strip())
    if not os.path.isabs(p):
        p = resource_path(p)
    return os.path.normpath(p)

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
        CTkTabview, CTkRadioButton, CTkSwitch, CTkOptionMenu, CTkSlider,
        CTkTextbox, CTkCheckBox, CTkToplevel, CTk)
    from pathlib import Path
    from tkinter import filedialog
    # Theme colors (mutated by apply_ui_theme). Names keep legacy dark_* labels.
    DARK_BG = "#000000"
    DARK_ENTRY_BG = "#1e1e1e"
    DARK_SECONDARY = "#2a2a2a"
    DARK_BORDER = "#333333"
    PURPLE_ACCENT = "#5200ff"
    ACTIVE_CHAT_COLOR = "#5200ff"
    BLINK_CHAT_COLOR_OFF = "#000000"
    WHITE = "#c7c7c7"  # primary text (light gray on dark / dark on light)
    DARK_TEXT_SECONDARY = "#b0b0b0"
    # Desired on-screen thickness (matches former ~110% look); code divides by scale so
    # 100% and 110% render the same physical width after CTk widget_scaling.
    SCROLLBAR_WIDTH = 13
    # UI scale: discrete positions 100%..170% (slider index → percent)
    UI_SCALE_STEPS = (100, 110, 120, 130, 140, 150, 160, 170)
    UI_LIGHT_THEME = False
    CORNER_RADIUS = 12
    FONT_FAMILY = "Georgia"
    # Base font size at 100% scale (min = current = 100%; max 170%)
    FONT_SIZE_BASE = 12
    UI_SCALE_PCT = 100
    FONT_REGULAR = (FONT_FAMILY, FONT_SIZE_BASE)
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
        "segmented_button_fg_color": DARK_BG,
        "segmented_button_selected_color": PURPLE_ACCENT,
        "segmented_button_selected_hover_color": PURPLE_ACCENT,
        "segmented_button_unselected_color": DARK_BG,
        "segmented_button_unselected_hover_color": WHITE,
        "text_color": WHITE,
        "fg_color": DARK_BG,
        "border_width": 0,
        # rounded tab switcher (content pad stripped separately in flush_tabview_content)
        "corner_radius": CORNER_RADIUS,
    }
    OPTIONMENU_THEME = {
        "fg_color": DARK_SECONDARY,
        "button_color": DARK_SECONDARY,
        "button_hover_color": PURPLE_ACCENT,
        "dropdown_fg_color": DARK_SECONDARY,
        "dropdown_hover_color": PURPLE_ACCENT,
        "text_color": WHITE,
        "corner_radius": CORNER_RADIUS,
        "font": FONT_REGULAR}

    def apply_ui_theme(light=False):
        """Dark (default) or light white-orange. Updates palette + shared theme dicts."""
        nonlocal DARK_BG, DARK_ENTRY_BG, DARK_SECONDARY, DARK_BORDER
        nonlocal PURPLE_ACCENT, ACTIVE_CHAT_COLOR, BLINK_CHAT_COLOR_OFF
        nonlocal WHITE, DARK_TEXT_SECONDARY, UI_LIGHT_THEME
        UI_LIGHT_THEME = bool(light)
        if light:
            # white–orange light theme (primary text must stay dark, not pale)
            DARK_BG = "#ffffff"
            DARK_ENTRY_BG = "#fff8f0"
            DARK_SECONDARY = "#fff0e0"
            DARK_BORDER = "#ffb366"
            PURPLE_ACCENT = "#ff6a00"
            ACTIVE_CHAT_COLOR = "#ff6a00"
            BLINK_CHAT_COLOR_OFF = "#ffffff"
            WHITE = "#1a1a1a"
            DARK_TEXT_SECONDARY = "#3d2e22"
        else:
            DARK_BG = "#000000"
            DARK_ENTRY_BG = "#1e1e1e"
            DARK_SECONDARY = "#2a2a2a"
            DARK_BORDER = "#333333"
            PURPLE_ACCENT = "#5200ff"
            ACTIVE_CHAT_COLOR = "#5200ff"
            BLINK_CHAT_COLOR_OFF = "#000000"
            WHITE = "#c7c7c7"
            DARK_TEXT_SECONDARY = "#b0b0b0"
        BUTTON_THEME["hover_color"] = PURPLE_ACCENT
        ENTRY_THEME["fg_color"] = PURPLE_ACCENT
        ENTRY_THEME["text_color"] = WHITE if not light else "#ffffff"
        TAB_VIEW_THEME.update({
            "segmented_button_fg_color": DARK_BG,
            "segmented_button_selected_color": PURPLE_ACCENT,
            "segmented_button_selected_hover_color": PURPLE_ACCENT,
            "segmented_button_unselected_color": DARK_BG,
            "segmented_button_unselected_hover_color": DARK_SECONDARY,
            "text_color": WHITE,
            "fg_color": DARK_BG,
        })
        OPTIONMENU_THEME.update({
            "fg_color": DARK_SECONDARY,
            "button_color": DARK_SECONDARY,
            "button_hover_color": PURPLE_ACCENT,
            "dropdown_fg_color": DARK_SECONDARY,
            "dropdown_hover_color": PURPLE_ACCENT,
            "text_color": WHITE,
        })
        BUTTON_THEME["text_color"] = WHITE
        try:
            customtkinter.set_appearance_mode("light" if light else "dark")
        except Exception:
            pass

    def clamp_ui_scale_pct(raw) -> int:
        try:
            v = int(float(str(raw).strip().replace('%', '')))
        except (TypeError, ValueError):
            v = 100
        v = max(100, min(170, v))
        # snap to nearest discrete step
        steps = UI_SCALE_STEPS
        return min(steps, key=lambda s: abs(s - v))

    def ui_scale_pct_to_index(pct) -> int:
        pct = clamp_ui_scale_pct(pct)
        try:
            return list(UI_SCALE_STEPS).index(pct)
        except ValueError:
            return 0

    def ui_scale_index_to_pct(idx) -> int:
        try:
            i = int(round(float(idx)))
        except (TypeError, ValueError):
            i = 0
        i = max(0, min(len(UI_SCALE_STEPS) - 1, i))
        return UI_SCALE_STEPS[i]

    def apply_ui_scale(pct=100):
        """Scale fonts + CTk widgets. 100% = current design size; max 170%.

        Must call after widgets exist; decreasing scale requires rebuild of main UI
        (CTk does not fully shrink already-built widgets on set_widget_scaling alone).
        """
        nonlocal FONT_REGULAR, UI_SCALE_PCT, BUTTON_THEME, ENTRY_THEME
        UI_SCALE_PCT = clamp_ui_scale_pct(pct)
        factor = UI_SCALE_PCT / 100.0
        size = max(1, int(round(FONT_SIZE_BASE * factor)))
        FONT_REGULAR = (FONT_FAMILY, size)
        BUTTON_THEME["font"] = FONT_REGULAR
        ENTRY_THEME["font"] = FONT_REGULAR
        ENTRY_THEME["height"] = max(27, int(round(27 * factor)))
        BUTTON_THEME["width"] = max(20, int(round(20 * factor)))
        BUTTON_THEME["height"] = max(20, int(round(20 * factor)))
        try:
            customtkinter.set_widget_scaling(factor)
        except Exception:
            pass
        try:
            customtkinter.set_window_scaling(factor)
        except Exception:
            pass
        # force layout refresh on existing roots when possible
        try:
            customtkinter.DrawEngine.preferred_drawing_method = customtkinter.DrawEngine.preferred_drawing_method
        except Exception:
            pass

    def create_styled_button(parent, text, command=None, width=None, height=None, **kwargs):
        default_kwargs = BUTTON_THEME.copy()
        default_kwargs.setdefault("text_color", WHITE)
        if width: default_kwargs["width"] = width
        if height: default_kwargs["height"] = height
        default_kwargs.update(kwargs)
        if "text_color" not in kwargs:
            default_kwargs["text_color"] = WHITE
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
        default_kwargs = {"font": FONT_REGULAR, "text_color": WHITE}
        default_kwargs.update(kwargs)
        if "fg_color" not in default_kwargs: default_kwargs["fg_color"] = "transparent"
        if "height" not in default_kwargs: default_kwargs["height"] = 0
        # ensure primary text follows theme (light: dark text, not CTk pale gray)
        if "text_color" not in kwargs:
            default_kwargs["text_color"] = WHITE
        return CTkLabel(parent, text=text, **default_kwargs)
    def _param_is_bool(param_info) -> bool:
        d = param_info.get('default')
        if isinstance(d, bool):
            return True
        if isinstance(d, str) and d.strip().lower() in ('true', 'false'):
            return True
        return False

    def _normalize_bool_param_value(value, default_val=None) -> str:
        """Map True/False/1/0/yes/no → 'true'|'false' for CTkSwitch + connect string."""
        if isinstance(value, bool):
            return 'true' if value else 'false'
        if value is None or (isinstance(value, str) and value.strip() == ''):
            if isinstance(default_val, bool):
                return 'true' if default_val else 'false'
            if isinstance(default_val, str) and default_val.strip().lower() in ('true', 'false'):
                return default_val.strip().lower()
            return 'false'
        s = str(value).strip().lower()
        if s in ('true', '1', 'yes', 'on', 'y'):
            return 'true'
        if s in ('false', '0', 'no', 'off', 'n'):
            return 'false'
        # already true/false or unknown → default off
        return 'true' if s == 'true' else 'false'

    def create_param_widget(parent, param_info, settings_vars_dict, path_vars_dict, on_change_callback=None):
        """Параметры провайдера — имя + поле/switch (bool → true/false в connect string)."""
        param_name = param_info['name']
        default_val = param_info.get('default')
        is_file = param_info['is_file']
        is_bool = _param_is_bool(param_info)
        param_frame = create_styled_frame(parent)
        param_frame.pack(fill="x", pady=2, padx=5)
        param_frame.grid_columnconfigure(1, weight=1)
        label_text = param_name
        # for bools don't clutter label with default; switch shows state
        if (not is_bool) and default_val is not None and str(default_val).strip() != '':
            label_text += f" {default_val}"
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
        elif is_bool:
            init = _normalize_bool_param_value(default_val, default_val)
            settings_vars_dict[param_name] = tk.StringVar(value=init)
            if on_change_callback:
                settings_vars_dict[param_name].trace_add("write", lambda *args: on_change_callback(param_name))
            sw = CTkSwitch(
                input_frame, text="", variable=settings_vars_dict[param_name],
                onvalue="true", offvalue="false",
                switch_width=50, switch_height=25,
                progress_color=PURPLE_ACCENT, font=FONT_REGULAR)
            sw.grid(row=0, column=0, sticky="e")
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
    def create_chat_message_bubble(parent, text, is_my, attachments=None, is_question=False, timestamp=None, show_datetime=False):
        row_frame = create_styled_frame(parent)
        row_frame.pack(fill=tk.X, pady=2, padx=10, anchor="center")
        if is_my: bubble = create_styled_frame(row_frame, border_width=0, corner_radius=CORNER_RADIUS, fg_color=DARK_BG)
        else: bubble = create_styled_frame(row_frame, border_width=0, corner_radius=CORNER_RADIUS, fg_color=PURPLE_ACCENT)
        bubble.pack(expand=False, anchor="center")
        if show_datetime and timestamp:
            ts_label = CTkLabel(bubble, text=str(timestamp), justify="left", anchor="w", fg_color="transparent", text_color=DARK_TEXT_SECONDARY, font=(FONT_FAMILY, 9), height=0)
            ts_label.pack(fill=tk.X, expand=True, padx=8, pady=(4, 0))
        msg_text_widget = CTkLabel(bubble, text=text, justify="left", anchor="w", fg_color="transparent", text_color=WHITE, font=FONT_REGULAR, height=0)
        if is_question and not is_my: msg_text_widget.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 8), pady=6)
        else: msg_text_widget.pack(fill=tk.X, expand=True, padx=8, pady=6)
        return bubble, msg_text_widget

    def setup_message_wraplength(widget, messages_frame):
        def update_wraplength(event=None):
            try:
                if widget.winfo_exists():
                    fw = messages_frame.winfo_width()
                    pad = max(48, int(20 * (UI_SCALE_PCT / 100.0)) + SCROLLBAR_WIDTH * 2)
                    available_width = max(60, fw - pad)
                    if available_width > 50: widget.configure(wraplength=available_width)
            except Exception: pass
        messages_frame.bind("<Configure>", update_wraplength)
        widget.after(100, update_wraplength)
        return update_wraplength
    def _parse_window_size(window, fallback=(600, 500)):
        """Reliable width/height for unmapped/preloaded windows (winfo_* can be 1x1)."""
        fw, fh = fallback
        try:
            stored = getattr(window, '_default_geometry', None)
            if stored and 'x' in str(stored):
                part = str(stored).lower().split('+')[0].split('-')[0]
                w, h = part.split('x', 1)
                fw, fh = int(w), int(h)
        except Exception:
            pass
        try:
            geo = window.geometry()  # e.g. 720x560+10+10 or 720x560
            if geo and 'x' in geo:
                part = geo.split('+')[0].split('-')[0]
                w, h = part.split('x', 1)
                w, h = int(w), int(h)
                if w >= 80 and h >= 80:
                    return w, h
        except Exception:
            pass
        try:
            w, h = int(window.winfo_width()), int(window.winfo_height())
            if w >= 80 and h >= 80:
                return w, h
        except Exception:
            pass
        try:
            w, h = int(window.winfo_reqwidth()), int(window.winfo_reqheight())
            if w >= 80 and h >= 80:
                return w, h
        except Exception:
            pass
        return fw, fh

    def setup_window_geometry(window, width=600, height=500, min_width=None, min_height=None):
        """Default size. Dialogs: min = start size (not half). ChatApp sets its own min."""
        window._default_geometry = f"{width}x{height}"
        window.geometry(f"{width}x{height}")
        mw = min_width if min_width is not None else max(200, int(width * 2 / 3))
        mh = min_height if min_height is not None else max(160, int(height * 2 / 3))
        window.minsize(mw, mh)
        try:
            window.resizable(True, True)
        except Exception:
            pass
        window.after(10, lambda: set_windows_dark_titlebar(window))
        # only center if already visible (preloaded windows stay withdrawn)
        def _maybe_center():
            try:
                if getattr(window, '_preload', False):
                    return
                if str(window.state()) == 'withdrawn':
                    return
                center_window(window)
            except Exception:
                pass
        window.after_idle(_maybe_center)
        return window

    def center_window(window, relative_to=None):
        """Center on screen (not on parent). relative_to ignored — keep API stable."""
        try:
            window.update_idletasks()
        except Exception:
            pass
        try:
            width, height = _parse_window_size(window)
            sw = window.winfo_screenwidth()
            sh = window.winfo_screenheight()
            x = (sw - width) // 2
            y = (sh - height) // 2
            x = max(0, min(int(x), max(0, sw - width)))
            y = max(0, min(int(y), max(0, sh - height)))
            window.geometry(f'{width}x{height}+{x}+{y}')
        except tk.TclError:
            pass
    def flush_tabview_content(tabview):
        """Rounded tab buttons; body flush to edges (no CTk corner_radius pad on tab content)."""
        if tabview is None:
            return tabview
        try:
            # Body/canvas: square + flush. Buttons: keep rounded separately.
            btn_cr = CORNER_RADIUS
            tabview._corner_radius = 0  # canvas + tab padx source
            tabview._border_width = 0
            tabview._outer_spacing = 0
            if hasattr(tabview, '_segmented_button') and tabview._segmented_button is not None:
                tabview._segmented_button.configure(corner_radius=btn_cr)
            if hasattr(tabview, '_configure_grid'):
                tabview._configure_grid()
            # Side inset of tab switcher used to mirror body radius — keep a little so buttons look inset
            try:
                if hasattr(tabview, '_segmented_button') and tabview._segmented_button is not None:
                    tabview._segmented_button.grid(
                        row=1, rowspan=2, column=0, columnspan=1,
                        padx=max(4, int(btn_cr // 2)), sticky="ns")
            except Exception:
                if hasattr(tabview, '_set_grid_segmented_button'):
                    tabview._set_grid_segmented_button()
            if hasattr(tabview, '_set_grid_canvas'):
                tabview._set_grid_canvas()
        except Exception:
            pass

        def _set_grid_current_tab_flush():
            name = getattr(tabview, '_current_name', '') or ''
            tabs = getattr(tabview, '_tab_dict', None) or {}
            frame = tabs.get(name)
            if frame is None:
                return
            try:
                frame.grid(row=3, column=0, sticky="nsew", padx=0, pady=0)
            except Exception:
                pass

        try:
            tabview._set_grid_current_tab = _set_grid_current_tab_flush
            _set_grid_current_tab_flush()
            tabview.after(50, _set_grid_current_tab_flush)
            tabview.after(150, _set_grid_current_tab_flush)
        except Exception:
            pass
        try:
            if hasattr(tabview, '_draw'):
                tabview._draw(no_color_updates=False)
        except Exception:
            pass
        return tabview

    def create_tabbed_interface(parent, tabs_config):
        tabview = CTkTabview(parent, **TAB_VIEW_THEME)
        tabview.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        tabs = {}
        for tab_name, setup_func in tabs_config.items():
            tab = tabview.add(Lang.get(tab_name))
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(0, weight=1)
            setup_func(tab)
            tabs[tab_name] = tab
        flush_tabview_content(tabview)
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
    def effective_scrollbar_width():
        """CTk multiplies width by widget_scaling — divide so physical thickness stays ~SCROLLBAR_WIDTH."""
        factor = max(0.5, UI_SCALE_PCT / 100.0)
        return max(8, int(round(SCROLLBAR_WIDTH / factor)))

    def _style_scrollbar_widget(sb, *, idle=False):
        """Unified thickness (scale-compensated); idle → blend with bg, active → accent."""
        if sb is None:
            return
        w = effective_scrollbar_width()
        try:
            sb.configure(width=w, corner_radius=50)
        except Exception:
            try:
                sb.configure(width=w)
            except Exception:
                pass
        try:
            if idle:
                sb.configure(button_color=DARK_BG, button_hover_color=DARK_BG, fg_color=DARK_BG)
            else:
                sb.configure(button_color=PURPLE_ACCENT, button_hover_color=WHITE, fg_color=DARK_BG)
        except Exception:
            pass

    def _refresh_scrollbar_fade(scroll_frame):
        """Update scrollbar colors from current content size (call after layout)."""
        try:
            canvas = getattr(scroll_frame, '_parent_canvas', None)
            if canvas is None:
                return
            try:
                scroll_frame.update_idletasks()
            except Exception:
                pass
            bbox = canvas.bbox('all')
            ch = (bbox[3] - bbox[1]) if bbox else 0
            cw = (bbox[2] - bbox[0]) if bbox else 0
            vh = max(int(canvas.winfo_height() or 0), 1)
            vw = max(int(canvas.winfo_width() or 0), 1)
            # if canvas not laid out yet (height 1), keep accent so bar is not "black hole"
            if vh <= 2 and ch > 0:
                need_v = True
            else:
                need_v = ch > vh + 2
            need_h = cw > vw + 2
            if hasattr(scroll_frame, '_scrollbar'):
                _style_scrollbar_widget(scroll_frame._scrollbar, idle=not need_v)
            if hasattr(scroll_frame, '_scrollbar_horizontal'):
                _style_scrollbar_widget(scroll_frame._scrollbar_horizontal, idle=not need_h)
        except Exception:
            pass

    def _bind_scrollbar_fade(scroll_frame):
        """Fade scrollbars when content fits (nothing to scroll)."""
        def _refresh(_event=None):
            _refresh_scrollbar_fade(scroll_frame)
        try:
            canvas = getattr(scroll_frame, '_parent_canvas', None)
            if canvas is not None:
                canvas.bind('<Configure>', _refresh, add='+')
            scroll_frame.bind('<Configure>', _refresh, add='+')
            scroll_frame.after(80, _refresh)
            scroll_frame.after(250, _refresh)
            scroll_frame.after(600, _refresh)
        except Exception:
            pass
        scroll_frame._refresh_scrollbar_fade = lambda: _refresh_scrollbar_fade(scroll_frame)
        return scroll_frame

    def create_scrollable_frame(parent, **kwargs):
        kwargs.setdefault('scrollbar_button_color', PURPLE_ACCENT)
        kwargs.setdefault('scrollbar_button_hover_color', WHITE)
        kwargs.setdefault('fg_color', kwargs.get('fg_color', 'transparent'))
        # CTkScrollableFrame pads canvas by corner_radius+border_width — keep 0 so content is flush
        kwargs.setdefault('corner_radius', 0)
        kwargs.setdefault('border_width', 0)
        scroll_frame = CTkScrollableFrame(parent, **kwargs)
        try:
            # ensure theme defaults didn't leave radius → re-grid with zero border_spacing
            if hasattr(scroll_frame, '_parent_frame'):
                scroll_frame._parent_frame.configure(corner_radius=0, border_width=0)
            if hasattr(scroll_frame, '_create_grid'):
                scroll_frame._create_grid()
        except Exception:
            pass
        if hasattr(scroll_frame, '_scrollbar'):
            # start accent (visible); fade will hide if no overflow after layout
            _style_scrollbar_widget(scroll_frame._scrollbar, idle=False)
        if hasattr(scroll_frame, '_scrollbar_horizontal'):
            _style_scrollbar_widget(scroll_frame._scrollbar_horizontal, idle=True)
        return _bind_scrollbar_fade(scroll_frame)

    def draw_brace_on_canvas(canvas, side='right', color=None):
        """Draw chat-list-style brace (arc+line) on a thin Canvas."""
        try:
            canvas.delete('all')
            h = canvas.winfo_height()
            w = max(canvas.winfo_width() - 1, 1)
            r = 6
            col = color if color is not None else WHITE
            if h <= r * 2:
                return
            if side == 'right':
                canvas.create_arc(w - r * 2, 0, w, r * 2, start=0, extent=90, style='arc', outline=col, width=1)
                canvas.create_line(w, r, w, h - r, fill=col, width=1)
                canvas.create_arc(w - r * 2, h - r * 2, w, h, start=270, extent=90, style='arc', outline=col, width=1)
            else:
                canvas.create_arc(0, 0, r * 2, r * 2, start=90, extent=90, style='arc', outline=col, width=1)
                canvas.create_line(0, r, 0, h - r, fill=col, width=1)
                canvas.create_arc(0, h - r * 2, r * 2, h, start=180, extent=90, style='arc', outline=col, width=1)
        except Exception:
            pass

    def create_braced_content(parent, height=132, side='left'):
        """
        Provider params container: brace (like chat list) instead of white border.
        Default height ~6 text rows (not 8).
        Returns (outer_frame, content_frame_for_children).
        """
        outer = create_styled_frame(parent, fg_color='transparent')
        outer.pack_propagate(False)
        try:
            outer.configure(height=height)
        except Exception:
            pass
        outer.grid_columnconfigure(1 if side == 'left' else 0, weight=1)
        outer.grid_rowconfigure(0, weight=1)
        brace = tk.Canvas(outer, width=6, bg=DARK_BG, highlightthickness=0)
        content = create_styled_frame(outer, fg_color=DARK_BG, border_width=0, corner_radius=0)
        if side == 'left':
            brace.grid(row=0, column=0, sticky='ns')
            content.grid(row=0, column=1, sticky='nsew')
        else:
            content.grid(row=0, column=0, sticky='nsew')
            brace.grid(row=0, column=1, sticky='ns')
        def _redraw(_e=None):
            try:
                brace.configure(bg=DARK_BG)
            except Exception:
                pass
            draw_brace_on_canvas(brace, side=side, color=WHITE)
        brace.bind('<Configure>', _redraw)
        outer.bind('<Configure>', _redraw)
        outer.after(30, _redraw)
        return outer, content
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
        # Индексы, а не локализованные labels — иначе после смены языка entryconfigure падает
        MENU_IDX_CUT, MENU_IDX_COPY, MENU_IDX_PASTE, MENU_IDX_SELECT = 0, 1, 2, 4
        menu.add_command(label=Lang.get("cut"), command=lambda: cut_action(None))
        menu.add_command(label=Lang.get("copy"), command=lambda: copy_action(None))
        menu.add_command(label=Lang.get("paste"), command=lambda: paste_action(None))
        menu.add_separator()
        menu.add_command(label=Lang.get("select_all"), command=lambda: select_all(None))
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
            try:
                menu.entryconfigure(MENU_IDX_CUT, label=Lang.get("cut"), state="normal" if has_selection and not is_disabled else "disabled")
                menu.entryconfigure(MENU_IDX_COPY, label=Lang.get("copy"), state="normal" if has_selection else "disabled")
                menu.entryconfigure(MENU_IDX_PASTE, label=Lang.get("paste"), state="normal" if has_clipboard and not is_disabled else "disabled")
                menu.entryconfigure(MENU_IDX_SELECT, label=Lang.get("select_all"), state="normal")
            except tk.TclError:
                pass
            menu.tk_popup(event.x_root, event.y_root)
        widget.bind("<Button-3>", show_menu)
        if sys.platform == "darwin": widget.bind("<Button-2>", show_menu)
    def _hex_to_colorref(hex_color: str) -> int:
        """#RRGGBB → Windows COLORREF 0x00BBGGRR."""
        h = (hex_color or "#000000").lstrip("#")
        if len(h) != 6:
            h = "000000"
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return r | (g << 8) | (b << 16)

    def _x11_collect_window_ids(window):
        """Walk X11 parent chain (no systemd / DE-specific APIs)."""
        ids = []
        try:
            import subprocess
            wid = window.winfo_id()
            seen = set()
            cur = wid
            for _ in range(12):
                if cur in seen:
                    break
                seen.add(cur)
                ids.append(str(cur) if not str(cur).startswith("0x") else str(cur))
                hx = hex(int(cur)) if not isinstance(cur, str) else (
                    cur if str(cur).startswith("0x") else hex(int(cur)))
                if hx not in ids:
                    ids.append(hx)
                try:
                    out = subprocess.check_output(
                        ["xwininfo", "-id", str(cur)],
                        stderr=subprocess.DEVNULL, text=True, timeout=0.35)
                except Exception:
                    break
                parent = None
                for line in out.splitlines():
                    if "Parent window id:" in line:
                        part = line.split(":")[-1].strip().split()[0]
                        if part.startswith("0x"):
                            try:
                                parent = int(part, 16)
                            except Exception:
                                parent = part
                        break
                if parent is None or parent in (0, 1):
                    break
                cur = parent
        except Exception:
            try:
                ids.append(str(window.winfo_id()))
            except Exception:
                pass
        out, seen2 = [], set()
        for i in ids:
            if i not in seen2:
                seen2.add(i)
                out.append(i)
        return out

    def set_windows_dark_titlebar(window):
        """Keep system titlebar; recolor when the OS allows it (no custom CSD chrome)."""
        bg = DARK_BG
        if sys.platform == "win32":
            try:
                import ctypes
                hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
                DWMWA_USE_IMMERSIVE_DARK_MODE = 20
                value = ctypes.c_int(0 if UI_LIGHT_THEME else 1)
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(value), ctypes.sizeof(value))
                DWMWA_CAPTION_COLOR = 35
                color = ctypes.c_int(_hex_to_colorref(bg))
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, DWMWA_CAPTION_COLOR, ctypes.byref(color), ctypes.sizeof(color))
                ctypes.windll.user32.SetWindowPos(hwnd, None, 0, 0, 0, 0, 0x0027)
            except Exception as e:
                print(f"Failed to set title bar: {e}")
            return
        if sys.platform == "darwin":
            # macOS: no reliable per-window titlebar color from Tk; leave system bar
            return
        if sys.platform.startswith("linux"):
            # Best-effort only: ask WM/theme for dark/light decorations (exact hex color
            # is usually not available with server-side decorations).
            try:
                window.update_idletasks()
                import subprocess
                variant = "light" if UI_LIGHT_THEME else "dark"
                scheme = "BreezeLight" if UI_LIGHT_THEME else "BreezeDark"
                for prop_win in _x11_collect_window_ids(window):
                    for args in (
                        ["xprop", "-id", prop_win, "-f", "_GTK_THEME_VARIANT", "8u",
                         "-set", "_GTK_THEME_VARIANT", variant],
                        ["xprop", "-id", prop_win, "-f", "_KDE_NET_WM_COLOR_SCHEME", "8u",
                         "-set", "_KDE_NET_WM_COLOR_SCHEME", scheme],
                        ["xprop", "-id", prop_win, "-f", "GTK_THEME_VARIANT", "8u",
                         "-set", "GTK_THEME_VARIANT", variant],
                    ):
                        try:
                            subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        except Exception:
                            pass
            except Exception:
                pass
            if not getattr(window, '_titlebar_map_bound', False):
                try:
                    def _on_map(_e=None, w=window):
                        try:
                            w.after(50, lambda: set_windows_dark_titlebar(w))
                        except Exception:
                            pass
                    window.bind("<Map>", _on_map, add="+")
                    window._titlebar_map_bound = True
                except Exception:
                    pass

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
                self.messages_by_chat = {}      # chat_id -> list[msg dict] (RAM, variant A)
                self.chats_loaded = False
                self.settings_loaded = False
                self.metadata_loaded = False
                self.initialized = True
        def clear_chats_cache(self):
            self.chats = None
            self.chats_loaded = False
            self.messages_by_chat = {}
        def clear_settings_cache(self):
            self.global_settings = None
            self.settings_loaded = False
        def clear_metadata_cache(self):
            self.settings_metadata = None
            self.metadata_loaded = False
        def clear_messages_cache(self, chat_id=None):
            if chat_id is None:
                self.messages_by_chat = {}
            else:
                self.messages_by_chat.pop(chat_id, None)
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
        def get(self, key, default=None, **kwargs):
            """key; optional default if missing; kwargs for str.format."""
            defaults = {
                "ok": "OK", "cancel": "Cancel", "yes": "Yes", "no": "No",
                "cut": "Cut", "copy": "Copy", "paste": "Paste", "select_all": "Select All",
                "undo": "Undo", "redo": "Redo", "error": "Error"}
            if key in defaults and key not in self.texts:
                template = defaults[key]
            elif key in self.texts:
                template = self.texts[key]
            elif default is not None:
                template = default
            else:
                template = f"[{key.upper()}]"
            try:
                return template.format(**kwargs) if kwargs else template
            except (KeyError, ValueError, IndexError):
                return template
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
            # Thin re-export aliases (importable, not shown as separate UI providers)
            alias_skip = {"xai_provider"}
            for py_file in provider_dir.glob("*.py"):
                if py_file.name.startswith("_") or not py_file.is_file(): continue
                module_name = py_file.stem
                if module_name in alias_skip:
                    continue
                display_name = module_name.replace("_", " ").title()
                if module_name == "grok_provider":
                    display_name = "Grok (xAI)"
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
        """
        Singleton RAM cache of default/custom modules.
        Load once at startup from DB; Save updates DB + this RAM.
        Create-chat / settings UI read only from here (no re-SELECT spam).
        """
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
            """Load from DB into RAM. Only if not loaded, or reload_m=True (lang rescan)."""
            if self.loaded and not reload_m:
                return
            try:
                self.default_modules = backend.get_default_mods()
                self.custom_modules = backend.get_custom_mods()
                self.loaded = True
            except Exception as e:
                print(f"Error loading modules: {e}")
                self.default_modules = []
                self.custom_modules = []
        def get_default_modules(self):
            return self.default_modules
        def get_custom_modules(self):
            return self.custom_modules
        def set_default_enabled(self, mod_id, enabled):
            """Update enabled flag in RAM only (DB already written by Backend)."""
            for m in self.default_modules:
                if m.get('id') == mod_id:
                    m['enabled'] = bool(enabled)
                    return True
            return False
        def update_custom_modules(self, backend):
            """Refresh custom list from DB (after add/remove custom). Defaults stay in RAM."""
            try:
                self.custom_modules = backend.get_custom_mods()
            except Exception as e:
                print(f"update_custom_modules: {e}")

    class Backend:
        def __init__(self):
            self.db_path = resource_path(os.path.join("data", "settings.db"))
            self.cache = AppCache()
            self.init_settings_db()
        def sql_exec(self, db_path, query, params=(), fetchone=False, fetchall=False, commit=True, executemany=False):
            try:
                conn = sqlite3.connect(db_path, timeout=10)
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                except Exception:
                    pass
                cursor = conn.cursor()
                if executemany:
                    cursor.executemany(query, params)
                else:
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
            if lang_file.exists() and lang:
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
                # helpers / private files (not agent tools)
                if mod_file.name.startswith("_"): return
                mod_name_stem = mod_file.stem.lower()
                # Per-OS shells removed — only unified shell_cmd
                if mod_name_stem in ('windows_cmd', 'linux_cmd', 'macos_cmd'):
                    return
                existing = self.sql_exec(db_path, "SELECT id FROM default_mods WHERE adress = ?", (relative_path_str,), fetchone=True)
                if existing: return
                valid, msg = ModuleValidator.validate_module(str(mod_file.resolve()))
                if not valid: print(f"Ошибка в модуле по умолчанию {mod_file.name}: {msg}"); return
                name, description = self._get_localized_doc(mod_file, lang=current_language)
                self.sql_exec(db_path, "INSERT OR IGNORE INTO default_mods (name, description, adress, enabled, lang) VALUES (?, ?, ?, ?, ?)", (name, description, relative_path_str, 0, current_language))
            # drop legacy per-OS cmd modules from DB if still present
            for legacy in ('linux_cmd/linux_cmd.py', 'windows_cmd/windows_cmd.py', 'macos_cmd/macos_cmd.py',
                           'linux_cmd.py', 'windows_cmd.py', 'macos_cmd.py'):
                try:
                    self.sql_exec(db_path, "DELETE FROM default_mods WHERE adress = ? OR adress LIKE ?",
                                  (legacy, f"%{legacy.split('/')[-1]}"))
                except Exception:
                    pass
            # drop helper mistaken as module
            try:
                self.sql_exec(db_path, "DELETE FROM default_mods WHERE adress LIKE ? OR adress LIKE ?",
                              ("%_one_line_arg.py", "%/_one_line_arg.py"))
            except Exception:
                pass
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
            # allow_ocr: needs local models + (on Linux) AVX2
            models_ok = bundled_image_models_available()
            if not models_ok:
                allow_ocr = "0"
            elif sys.platform.startswith("linux"):
                try:
                    has_avx2 = subprocess.run(['grep', '-q', 'avx2', '/proc/cpuinfo'], capture_output=True).returncode == 0
                    allow_ocr = "1" if has_avx2 else "0"
                except Exception:
                    allow_ocr = "0"
            else:
                allow_ocr = "1"
            defaults = {
                "token_limit": "8192", "model_provider_params": "",
                "provider_params_by_type": "{}",
                "model_type": default_provider, "use_rag": "1",
                "filter_generations": "0", "hierarchy_limit": "0",
                "write_log": "1", "write_results": "0",
                "fs_copy_touched_on_end": "0",
                "copy_user_attachments_to_files": "0",
                "max_critic_reactions": "2",
                "max_token_limit": "8192", "use_librarian": "0",
                "recreate_agents": "0",
                "skip_nested_images": "0",
                "cut_wrong_command_history": "0",
                "allow_ocr": allow_ocr,
                "number_of_plan_items": "0",
                "do_translate": "0",
                "target_lang": "None",
                "local_and_tools_translate": "0",
                "use_local_cache": "0",
                "use_global_cache": "0",
                "use_psm": "0",
                "use_magical_prompt": "0",
                "use_gigo": "1",
                "use_old_gigo": "1",
                "gigo_idea_count": "2",
                "gigo_plan_items": "10",
                "gigo_use_entropy": "0",
                "gigo_use_filter": "1",
                "gigo_use_librarian": "0",
                "show_message_datetime": "0",
                "chats_dir": suggest_default_chats_dir(),
                "librarian_use_models": "0",
                "module_hints_for_operator": "0",
                "give_all_tools": "0",
                "critic_reuse_dialog": "1",
                "one_shot_intention_permission": "0",
                "max_messages_before_answer": "0",
                "librarian_use_web": "0",
                "save_emb_dialog": "1",
                "tools_no_examples": "0",
                "allow_command_not_at_start": "0",
                "give_operator_goal_to_executor": "0",
                "deliver_user_messages": "0",
                "use_small_model": "0",
                "small_model_type": "",
                "small_model_provider_params": "",
                "small_token_limit": "8192",
                "small_max_token_limit": "8192",
                "small_for_cutter_only": "1",
                "small_agent_until_protocol": "0",
                "text_cutter_token_limit": "2000",
                "max_incoming_tokens": "10000",
                "release_version": "2026-07",
                "shell_skip_confirm": "0",
                "mcp_url": "",
                "max_executor_recreates": "0",
                "ui_light_theme": "0",
                "ui_scale": "100",
                "gigo_role_dreamer": "1",
                "gigo_role_realist": "1",
                "gigo_role_critic": "1",
                "fs_use_git": "1",
                "small_protocol_drop_error": "0",
                }
            for key, value in defaults.items(): self.sql_exec(db_path, "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (key, value))
            # Устанавливаем widget_type для известных ключей
            widget_type_map = {
                "use_rag": "switch",
                "filter_generations": "switch",
                "write_log": "switch",
                "write_results": "switch",
                "fs_copy_touched_on_end": "switch",
                "copy_user_attachments_to_files": "switch",
                "use_librarian": "switch",
                "recreate_agents": "switch",
                "skip_nested_images": "switch",
                "cut_wrong_command_history": "switch",
                "allow_ocr": "switch",
                "do_translate": "switch",
                "local_and_tools_translate": "switch",
                "use_local_cache": "switch",
                "use_global_cache": "switch",
                "use_psm": "switch",
                "use_magical_prompt": "switch",
                "use_gigo": "switch",
                "use_old_gigo": "switch",
                "gigo_use_entropy": "switch",
                "gigo_use_filter": "switch",
                "gigo_use_librarian": "switch",
                "show_message_datetime": "switch",
                "librarian_use_models": "switch",
                "module_hints_for_operator": "switch",
                "give_all_tools": "switch",
                "critic_reuse_dialog": "switch",
                "one_shot_intention_permission": "switch",
                "target_lang": "entry",
                "gigo_plan_items": "entry",
                "hierarchy_limit": "entry",
                "max_critic_reactions": "entry",
                "max_messages_before_answer": "entry",
                "max_executor_recreates": "entry",
                "number_of_plan_items": "entry",
                "librarian_use_web": "switch",
                "save_emb_dialog": "switch",
                "tools_no_examples": "switch",
                "allow_command_not_at_start": "switch",
                "give_operator_goal_to_executor": "switch",
                "deliver_user_messages": "switch",
                "use_small_model": "switch",
                "small_for_cutter_only": "switch",
                "small_agent_until_protocol": "switch",
                "small_model_type": "entry",
                "small_model_provider_params": "entry",
                "small_token_limit": "entry",
                "small_max_token_limit": "entry",
                "text_cutter_token_limit": "entry",
                "max_incoming_tokens": "entry",
                "shell_skip_confirm": "switch",
                "mcp_url": "entry",
                "gigo_idea_count": "entry",
                "chats_dir": "entry",
                "ui_light_theme": "switch",
                "ui_scale": "slider",
                "gigo_role_dreamer": "switch",
                "gigo_role_realist": "switch",
                "gigo_role_critic": "switch",
                "fs_use_git": "switch",
                "small_protocol_drop_error": "switch",
                }
            for key, wtype in widget_type_map.items(): self.sql_exec(db_path, "UPDATE settings SET widget_type = ? WHERE key = ?", (wtype, key))
        def _load_settings_metadata_from_db(self): # Возвращает список кортежей (key, widget_type) для всех записей settings.
            rows = self.sql_exec(self.db_path, "SELECT key, widget_type FROM settings", fetchall=True) or []
            return {row[0]: row[1] for row in rows}
        def get_settings_metadata(self): return self.cache.get_settings_metadata(self)
        def generate_id(self, length=12): return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
        def get_chats_root(self):
            """Корень хранения чатов (id = имя подпапки)."""
            settings = self.get_global_settings()
            root = normalize_chats_dir(settings.get("chats_dir", ""))
            try: os.makedirs(root, exist_ok=True)
            except OSError: pass
            return root
        def chat_folder(self, chat_id):
            return Path(self.get_chats_root()) / chat_id
        def migrate_chats_dir(self, old_root, new_root):
            """Переносит папки чатов из old_root в new_root. Возвращает (ok, message)."""
            old_root = normalize_chats_dir(old_root)
            new_root = normalize_chats_dir(new_root)
            if os.path.normpath(old_root) == os.path.normpath(new_root):
                return True, "same"
            if not os.path.isdir(old_root):
                try: os.makedirs(new_root, exist_ok=True)
                except OSError as e: return False, str(e)
                return True, "empty"
            try:
                os.makedirs(new_root, exist_ok=True)
                moved = 0
                for name in os.listdir(old_root):
                    src = os.path.join(old_root, name)
                    dst = os.path.join(new_root, name)
                    if not os.path.isdir(src): continue
                    if os.path.exists(dst):
                        continue
                    shutil.move(src, dst)
                    moved += 1
                return True, f"moved:{moved}"
            except Exception as e:
                return False, str(e)
        def _load_chats_from_db(self):
            result = []
            chats_dir = Path(self.get_chats_root())
            if not chats_dir.exists(): return []
            # Oldest first (top) → newest at bottom (reversed from mtime-desc)
            for folder in sorted(chats_dir.iterdir(), key=os.path.getmtime, reverse=False):
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
        def update_global_settings(self, settings, persist=True):
            """
            Apply settings to RAM always. Disk write is one transaction (executemany).
            persist=False: only RAM (UI stays instant; call persist_global_settings later).
            """
            merged = {**self.cache.get_global_settings(self), **(settings or {})}
            self.cache.update_global_settings(merged)
            if not persist:
                return True
            return self.persist_global_settings(settings)

        def persist_global_settings(self, settings):
            """Single SQLite connection + one commit for all keys (avoids N× open/commit freezes)."""
            if not settings:
                return True
            rows = [(str(k), str(v)) for k, v in settings.items()]
            try:
                conn = sqlite3.connect(self.db_path, timeout=10)
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                except Exception:
                    pass
                conn.executemany(
                    "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                    rows,
                )
                conn.commit()
                conn.close()
                return True
            except Exception as e:
                print(f"[SQL Error] persist_global_settings: {e}")
                # fallback key-by-key
                for key, value in settings.items():
                    self.sql_exec(
                        self.db_path,
                        "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                        (key, str(value)),
                    )
                return True
        # ====== НОВЫЙ МЕТОД ДЛЯ ОБНОВЛЕНИЯ ИМЕНИ ЧАТА ======
        def update_chat_name(self, chat_id, new_name):
            db_path = self.chat_folder(chat_id) / "chatsettings.db"
            if not db_path.exists():
                return False
            self.sql_exec(str(db_path), "UPDATE settings SET value = ? WHERE key = 'chat_name'", (new_name,))
            # Обновляем кэш
            chats = self.cache.get_chats(self)
            for chat in chats:
                if chat['id'] == chat_id:
                    chat['name'] = new_name
                    break
            self.cache.update_chats(chats)
            return True
        # ====================================================
        def create_chat(self, chat_name, settings_data):
            # Имя может совпадать с другими; id = уникальная папка
            existing_chats = self.cache.get_chats(self)
            chat_id = self.generate_id()
            root = Path(self.get_chats_root())
            while (root / chat_id).exists(): chat_id = self.generate_id()
            chat_path = root / chat_id
            chat_path.mkdir(parents=True, exist_ok=True)
            (chat_path / "files").mkdir(exist_ok=True)
            new_chat = {"id": chat_id, "name": chat_name}
            # Newest chats at bottom of the list
            updated_chats = list(existing_chats) + [new_chat]
            self.cache.update_chats(updated_chats)
            default_mods = ModuleManager().get_default_modules()
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
            # model_config MUST win over chat_config: chat_config copies ALL settings_vars
            # including stale model_provider_params / model_type from global snapshot.
            all_settings = {**settings_data.get('chat_config', {}), **settings_data.get('model_config', {})}
            all_settings['chat_name'] = chat_name
            # версия релиза всегда штамп приложения (не пользовательское поле)
            try:
                from cross_gpt import RELEASE_VERSION as _app_rel
            except Exception:
                _app_rel = '2026-07'
            all_settings['release_version'] = _app_rel
            for key, value in all_settings.items(): self.sql_exec(settings_db, "INSERT INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
            default_mods = ModuleManager().get_default_modules()
            enabled_defaults = settings_data.get('default_mods_config', {})
            for mod in default_mods: self.sql_exec(settings_db, "INSERT INTO default_mods (id, name, adress, enabled) VALUES (?, ?, ?, ?)", (mod['id'], mod['name'], mod['adress'], 1 if enabled_defaults.get(mod['id'], False) else 0))
            for mod in settings_data.get('custom_mods_list', []): self.sql_exec(settings_db, "INSERT INTO custom_mods (name, description, adress) VALUES (?, ?, ?)", (mod['name'], mod['description'], mod['adress']))
            dialog_db = str(chat_path / "chatsettings.db")
            self.sql_exec(dialog_db, "CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, is_my INTEGER, attachments TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
            return new_chat
        def delete_chat(self, chat_id):
            chat_path = self.chat_folder(chat_id)
            if not chat_path.exists(): return False
            shutil.rmtree(chat_path)
            existing_chats = self.cache.get_chats(self)
            updated_chats = [chat for chat in existing_chats if chat['id'] != chat_id]
            self.cache.update_chats(updated_chats)
            try:
                self.cache.clear_messages_cache(chat_id)
            except Exception:
                pass
            return True
        def _load_messages_from_db(self, chat_id):
            db = self.chat_folder(chat_id) / "chatsettings.db"
            if not db.exists():
                return []
            rows = self.sql_exec(
                str(db),
                "SELECT text, is_my, attachments, timestamp FROM messages ORDER BY timestamp",
                fetchall=True,
            ) or []
            out = []
            for r in rows:
                try:
                    atts = json.loads(r[2]) if r[2] else []
                except Exception:
                    atts = []
                out.append({
                    "text": r[0],
                    "isMy": bool(r[1]),
                    "attachments": atts,
                    "timestamp": r[3],
                })
            return out
        def get_messages(self, chat_id, *, force_reload=False):
            """Messages from RAM cache (variant A); first access loads SQLite once per chat."""
            if not chat_id:
                return []
            cache = self.cache.messages_by_chat
            if force_reload or chat_id not in cache:
                cache[chat_id] = self._load_messages_from_db(chat_id)
            # return the list itself (UI may only read; mutators go through add_message)
            return cache[chat_id]
        def add_message(self, chat_id, text, is_my, attachments=None):
            db = self.chat_folder(chat_id) / "chatsettings.db"
            if not db.exists(): return False
            attachments_str = json.dumps([str(a) for a in attachments]) if attachments else None
            self.sql_exec(str(db), "INSERT INTO messages (text, is_my, attachments) VALUES (?, ?, ?)", (text, int(is_my), attachments_str))
            # keep RAM in sync (append or seed from DB)
            try:
                ts = None
                try:
                    row = self.sql_exec(
                        str(db),
                        "SELECT timestamp FROM messages ORDER BY id DESC LIMIT 1",
                        fetchone=True,
                    )
                    if row:
                        ts = row[0]
                except Exception:
                    ts = None
                msg = {
                    "text": text,
                    "isMy": bool(is_my),
                    "attachments": list(attachments) if attachments else [],
                    "timestamp": ts,
                }
                if chat_id in self.cache.messages_by_chat:
                    self.cache.messages_by_chat[chat_id].append(msg)
                else:
                    # not loaded yet — leave empty so next get_messages loads full history
                    pass
            except Exception as e:
                print(f"messages RAM cache update: {e}")
                try:
                    self.cache.clear_messages_cache(chat_id)
                except Exception:
                    pass
            return True
        def get_chat_settings(self, chat_id):
            db = self.chat_folder(chat_id) / "chatsettings.db"
            if not db.exists(): return {}
            rows = self.sql_exec(str(db), "SELECT key, value FROM settings", fetchall=True) or []
            return {k: v for k, v in rows}
        def get_chat_tool_paths(self, chat_id):
            """Пути модулей чата (enabled default_mods + custom) — та же форма, что load_chat_settings another_tools."""
            db = self.chat_folder(chat_id) / "chatsettings.db"
            if not db.exists(): return []
            paths = []
            default_mods = self.sql_exec(str(db), "SELECT adress FROM default_mods WHERE enabled=?", (1,), fetchall=True) or []
            paths.extend([row[0] for row in default_mods if row and row[0]])
            custom_mods = self.sql_exec(str(db), "SELECT adress FROM custom_mods", fetchall=True) or []
            paths.extend([row[0] for row in custom_mods if row and row[0]])
            return paths
        def set_chat_setting(self, chat_id, key, value):
            """Обновить один ключ settings в chatsettings.db."""
            db = self.chat_folder(chat_id) / "chatsettings.db"
            if not db.exists():
                return False
            self.sql_exec(
                str(db),
                "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value)))
            return True
        def get_default_mods(self):
            rows = self.sql_exec(self.db_path, "SELECT id, name, description, adress, enabled FROM default_mods", fetchall=True) or []
            return [{"id": r[0], "name": r[1], "description": r[2], "adress": r[3], "enabled": bool(r[4])} for r in rows]
        def get_custom_mods(self):
            rows = self.sql_exec(self.db_path, "SELECT id, name, description, adress, enabled FROM custom_mods", fetchall=True) or []
            return [{"id": r[0], "name": r[1], "description": r[2], "adress": r[3], "enabled": bool(r[4])} for r in rows]
        def update_default_mod_enabled(self, mod_id, enabled):
            self.sql_exec(self.db_path, "UPDATE default_mods SET enabled = ? WHERE id = ?", (1 if enabled else 0, mod_id))
            # keep RAM in sync without re-querying all mods
            try:
                ModuleManager().set_default_enabled(mod_id, enabled)
            except Exception:
                pass
            return True
        def remove_custom_mod(self, mod_id):
            self.sql_exec(self.db_path, "DELETE FROM custom_mods WHERE id = ?", (mod_id,))
            try:
                ModuleManager().update_custom_modules(self)
            except Exception:
                pass
            return True
        def add_custom_mod(self, file_path):
            valid, error_msg = ModuleValidator.validate_module(file_path)
            if not valid: raise ValueError(Lang.get("module_validation_error", error_msg=error_msg))
            name, description = self._get_localized_doc(Path(file_path), lang=Lang.current_language)
            self.sql_exec(self.db_path, "INSERT INTO custom_mods (name, description, adress, enabled) VALUES (?, ?, ?, ?)", (name, description, file_path, 1))
            try:
                ModuleManager().update_custom_modules(self)
            except Exception:
                pass
            return True
        def is_main_config_complete(self):
            settings = self.get_global_settings()
            if not settings: return False
            return bool(settings.get("model_type")) and bool(settings.get("model_provider_params"))
        def validate_model_settings(self, model_type, connection_string, plain_password=None):
            max_tokens = 8192
            try:
                if not model_type: return False, Lang.get("model_err_no_provider"), max_tokens
                # same aliases as worker (ollama→ollama_provider, xai→grok_provider, …)
                try:
                    from cross_gpt import _normalize_provider_module_name
                    model_type = _normalize_provider_module_name(model_type)
                except Exception:
                    pass
                provider_manager = ProviderManager()
                provider_data = provider_manager.providers.get(model_type)
                if not provider_data: return False, Lang.get("model_err_provider_missing", provider=model_type), max_tokens
                provider_module = importlib.import_module(f"model_providers.{model_type}")
                f = io.StringIO()
                with redirect_stdout(f):
                    try:
                        import inspect
                        sig = inspect.signature(provider_module.connect)
                        if '_decrypted_token' in sig.parameters and plain_password is not None:
                            valid, tokens, _, *rest = provider_module.connect(connection_string, _decrypted_token=plain_password)
                        elif '_decrypted_password' in sig.parameters:
                            valid, tokens, _, *rest = provider_module.connect(connection_string, _decrypted_password=plain_password)
                        else:
                            valid, tokens, _, *rest = provider_module.connect(connection_string)
                    except Exception as ex:
                        try: valid, tokens, _, error_text = provider_module.connect(connection_string); return False, Lang.get("model_err_validation_generic", e=error_text), max_tokens
                        except: return False, Lang.get("model_err_validation_generic", e=str(ex)), max_tokens
                if hasattr(provider_module, 'disconnect'): provider_module.disconnect()
                if valid: return True, Lang.get("model_validated_success", tokens=tokens), tokens
                else: err_msg = rest[0] if rest else Lang.get("custom_api_fail"); return False, err_msg, max_tokens
            except ImportError as e: print(e); return False, Lang.get("model_err_provider_missing", provider=model_type), max_tokens
            except Exception as e: return False, Lang.get("model_err_validation_generic", e=str(e)), max_tokens
    class BaseTopLevel(CTkToplevel):
        # modal=True: block parent (settings/create chat). Log uses modal=False.
        # preload=True: build UI hidden during splash; present() shows later.
        modal = True
        def __init__(self, master, *args, preload=False, **kwargs):
            super().__init__(master, *args, **kwargs)
            self._preload = bool(preload)
            self.configure(fg_color=DARK_BG)
            setup_window_geometry(self)
            self.protocol("WM_DELETE_WINDOW", self.on_close)
            try:
                self.grid_rowconfigure(0, weight=1)
                self.grid_columnconfigure(0, weight=1)
            except Exception:
                pass
            if self._preload:
                try:
                    self.withdraw()
                except Exception:
                    pass
            self.after_idle(self.setup_and_center)
            self.after(20, lambda: set_windows_dark_titlebar(self))
        def setup_and_center(self):
            setup_icon(self)
            if getattr(self, '_preload', False):
                try:
                    self.withdraw()
                except Exception:
                    pass
                return
            try:
                geo = getattr(self, '_default_geometry', None)
                if geo:
                    self.geometry(geo)
            except Exception:
                pass
            try:
                center_window(self)
            except Exception:
                pass
            self.lift()
            if getattr(self, 'modal', True):
                try:
                    self.grab_set()
                    self.transient(self.master)
                except Exception:
                    pass
        def present(self):
            """Show a preloaded (or reopened) window and take modal focus (no rebuild)."""
            self._preload = False
            try:
                self.configure(fg_color=DARK_BG)
            except Exception:
                pass
            # restore intended size before map (withdrawn winfo can be 1×1)
            try:
                geo = getattr(self, '_default_geometry', None)
                if geo:
                    self.geometry(geo)
            except Exception:
                pass
            try:
                self.deiconify()
            except Exception:
                pass
            try:
                set_windows_dark_titlebar(self)
            except Exception:
                pass
            try:
                # Single center on show — no delayed re-center (was causing a visible jump)
                center_window(self)
            except Exception:
                pass
            try:
                self.lift()
                self.focus_force()
            except Exception:
                pass
            if getattr(self, 'modal', True):
                try:
                    self.grab_set()
                    self.transient(self.master)
                except Exception:
                    pass
        def hide_to_preload(self):
            """Close visually but keep window for instant reopen."""
            try:
                self.grab_release()
            except Exception:
                pass
            self._preload = True
            try:
                self.withdraw()
            except Exception:
                pass
        def on_close(self):
            # Prefer hide-to-preload for settings/create (instant reopen)
            if getattr(self, 'keep_preloaded', False):
                self.hide_to_preload()
                return
            try:
                self.grab_release()
            except Exception:
                pass
            self.destroy()
    # ===== Общая функция для построения UI настроек чата =====
    # Semantic sections for chat settings (theme order). Unlisted keys → others.
    SETTINGS_THEME_ORDER = (
        'memory', 'gigo', 'agents', 'limits', 'translation',
        'files', 'tools', 'client', 'ui', 'others',
    )
    SETTINGS_THEME_MAP = {
        # memory / RAG / librarian
        'use_rag': 'memory',
        'use_librarian': 'memory',
        'librarian_use_models': 'memory',
        'librarian_use_web': 'memory',
        'save_emb_dialog': 'memory',
        # GIGO
        'use_gigo': 'gigo',
        'use_old_gigo': 'gigo',
        'gigo_idea_count': 'gigo',
        'gigo_plan_items': 'gigo',
        'gigo_use_entropy': 'gigo',
        'gigo_use_concepts': 'gigo',
        'gigo_use_filter': 'gigo',
        'gigo_use_librarian': 'gigo',
        'gigo_role_dreamer': 'gigo',
        'gigo_role_realist': 'gigo',
        'gigo_role_critic': 'gigo',
        'number_of_plan_items': 'gigo',
        # agents / critic / hierarchy
        'hierarchy_limit': 'agents',
        'recreate_agents': 'agents',
        'max_executor_recreates': 'agents',
        'max_messages_before_answer': 'agents',
        'max_critic_reactions': 'agents',
        'critic_reuse_dialog': 'agents',
        'use_psm': 'agents',
        'use_magical_prompt': 'agents',
        'module_hints_for_operator': 'agents',
        'give_all_tools': 'agents',
        'tools_no_examples': 'agents',
        'allow_command_not_at_start': 'agents',
        'give_operator_goal_to_executor': 'agents',
        'cut_wrong_command_history': 'agents',
        'filter_generations': 'agents',
        # numeric limits (non-agent)
        'text_cutter_token_limit': 'limits',
        'max_incoming_tokens': 'limits',
        # translation
        'do_translate': 'translation',
        'target_lang': 'translation',
        'local_and_tools_translate': 'translation',
        'use_local_cache': 'translation',
        'use_global_cache': 'translation',
        # files / OCR / FS
        'fs_use_git': 'files',
        'fs_copy_touched_on_end': 'files',
        'copy_user_attachments_to_files': 'files',
        'write_results': 'files',
        'write_log': 'files',
        'skip_nested_images': 'files',
        'allow_ocr': 'files',
        'one_shot_intention_permission': 'files',
        # shell / mcp
        'shell_skip_confirm': 'tools',
        'mcp_url': 'tools',
        # mid-dialog client
        'deliver_user_messages': 'client',
        # UI
        'show_message_datetime': 'ui',
        'ui_light_theme': 'ui',
        'ui_scale': 'ui',
    }

    def _settings_theme_for_key(key: str) -> str:
        return SETTINGS_THEME_MAP.get(key, 'others')

    def _group_settings_keys(keys):
        """Map theme_name → [keys] in SETTINGS_THEME_ORDER. Keys unchanged."""
        buckets = {}
        for k in keys:
            buckets.setdefault(_settings_theme_for_key(k), []).append(k)
        ordered = {}
        for theme in SETTINGS_THEME_ORDER:
            if theme in buckets and buckets[theme]:
                ordered[theme] = buckets[theme]
        for theme, items in buckets.items():
            if theme not in ordered and items:
                ordered[theme] = items
        return ordered

    def build_chat_settings_ui(parent, settings_vars, metadata):
        """
        Chat settings: grouped by key prefix (gigo_, fs_, …), collapsible sections.
        Ungrouped / single-prefix keys → section "others".
        DB keys stay without group rename. *_desc → spoiler under each row.
        """
        created_widgets = {}
        skip_keys = {
            'language', 'model_type', 'model_provider_params', 'token_limit', 'max_token_limit',
            'chat_name', 'chats_dir', 'release_version', 'provider_params_by_type',
            'use_small_model', 'small_model_type', 'small_model_provider_params',
            'small_token_limit', 'small_max_token_limit',
            'small_for_cutter_only', 'small_agent_until_protocol',
            'small_protocol_drop_error',
            # text_cutter_* live in chat settings tab (not model tab)
        }
        visible = [(k, v) for k, v in metadata.items() if k not in skip_keys and k in settings_vars]
        # entries before switches inside each group
        def sort_items(items):
            # entries/sliders first, then switches
            non_sw = [(k, w) for k, w in items if w != 'switch']
            switches = [(k, w) for k, w in items if w == 'switch']
            return non_sw + switches

        by_key = {k: w for k, w in visible}
        groups = _group_settings_keys([k for k, _ in visible])

        def _make_scale_slider(parent, key):
            """Discrete UI scale slider (no numeric label). Syncs settings_vars[key] percent string."""
            var = settings_vars[key]
            idx0 = ui_scale_pct_to_index(var.get() if hasattr(var, 'get') else '100')
            n_steps = max(1, len(UI_SCALE_STEPS) - 1)

            def _on_slide(v, sv=var):
                # CTkSlider passes current value as the command argument — use it
                try:
                    pct = ui_scale_index_to_pct(v)
                    sv.set(str(pct))
                except Exception:
                    pass

            slider = CTkSlider(
                parent,
                from_=0,
                to=n_steps,
                number_of_steps=n_steps,
                command=_on_slide,
                width=140,
                height=18,
                progress_color=PURPLE_ACCENT,
                button_color=WHITE,
                button_hover_color=PURPLE_ACCENT,
            )
            slider.set(float(idx0))
            try:
                var.set(str(ui_scale_index_to_pct(idx0)))
            except Exception:
                pass
            # stash for save-time re-read
            slider._milana_scale_key = key
            return slider

        def add_setting_row(container, key, wtype):
            desc_key = key + "_desc"
            has_desc = desc_key in Lang.texts
            frame = create_styled_frame(container)
            frame.pack(fill="x", pady=2)
            frame.grid_columnconfigure(0, weight=1)
            frame.grid_columnconfigure(1, weight=0)
            if has_desc:
                btn_text = "╰ " + Lang.get(key, default=key)
                btn = CTkButton(
                    frame, text=btn_text, anchor="w", fg_color="transparent",
                    hover_color=PURPLE_ACCENT, corner_radius=CORNER_RADIUS,
                    font=FONT_REGULAR, height=27, text_color=WHITE,
                )
                btn.grid(row=0, column=0, sticky="ew", padx=(0, 10))
                if wtype == 'switch':
                    switch = CTkSwitch(
                        frame, text="", variable=settings_vars[key],
                        onvalue="1", offvalue="0", switch_width=50, switch_height=25,
                        progress_color=PURPLE_ACCENT, font=FONT_REGULAR)
                    switch.grid(row=0, column=1, sticky="e")
                    if key == 'allow_ocr' and not bundled_image_models_available():
                        try:
                            settings_vars[key].set('0')
                            switch.configure(state='disabled')
                        except Exception:
                            pass
                    created_widgets[key] = switch
                elif wtype == 'slider' or key == 'ui_scale':
                    slider = _make_scale_slider(frame, key)
                    slider.grid(row=0, column=1, sticky="e", padx=(0, 4))
                    created_widgets[key] = slider
                else:
                    entry = create_styled_entry(frame, textvariable=settings_vars[key])
                    entry.grid(row=0, column=1, sticky="ew")
                    created_widgets[key] = entry
                desc_frame = create_styled_frame(frame, fg_color="transparent")
                desc_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))
                _desc = Lang.get(desc_key, default="")
                if key == 'allow_ocr' and not bundled_image_models_available():
                    _desc = (str(_desc) + "\n" + Lang.get(
                        "allow_ocr_no_models",
                        default="Image models not installed — reinstall with models to enable.",
                    )).strip()
                desc_label = create_styled_label(
                    desc_frame, text=_desc, wraplength=400, justify="left",
                    text_color=WHITE)
                desc_label.pack(fill="x", padx=5, pady=2)
                menu = tk.Menu(desc_label, tearoff=0, bg=DARK_SECONDARY, fg=WHITE, relief="flat", borderwidth=0, font=(FONT_FAMILY, 8))
                def copy_description(label=desc_label):
                    try:
                        text = label.cget("text")
                        if text:
                            label.clipboard_clear()
                            label.clipboard_append(text)
                    except tk.TclError:
                        pass
                menu.add_command(label=Lang.get("copy"), command=copy_description)
                def show_menu(event, m=menu):
                    try:
                        m.entryconfigure(0, label=Lang.get("copy"))
                    except tk.TclError:
                        pass
                    m.tk_popup(event.x_root, event.y_root)
                desc_label.bind("<Button-3>", show_menu)
                if sys.platform == "darwin":
                    desc_label.bind("<Button-2>", show_menu)
                def make_toggle(b=btn, df=desc_frame, k=key):
                    def toggle():
                        if df.winfo_ismapped():
                            df.grid_remove()
                            b.configure(text="╰ " + Lang.get(k, default=k), text_color=WHITE)
                        else:
                            df.grid()
                            b.configure(text="╭ " + Lang.get(k, default=k), text_color=WHITE)
                    return toggle
                toggle_cmd = make_toggle()
                btn.configure(command=toggle_cmd)
                desc_label.bind("<Button-1>", lambda e, t=toggle_cmd: t())
                desc_frame.grid_remove()
                def update_wraplength(event, label=desc_label, fr=frame):
                    width = fr.winfo_width() - 20
                    if width > 50:
                        label.configure(wraplength=width)
                frame.bind("<Configure>", update_wraplength)
                frame.after(10, lambda: update_wraplength(None))
            else:
                create_styled_label(frame, text=Lang.get(key, default=key), text_color=WHITE).grid(
                    row=0, column=0, sticky="w", padx=(0, 10))
                if wtype == 'switch':
                    switch = CTkSwitch(
                        frame, text="", variable=settings_vars[key],
                        onvalue="1", offvalue="0", switch_width=50, switch_height=25,
                        progress_color=PURPLE_ACCENT, font=FONT_REGULAR)
                    switch.grid(row=0, column=1, sticky="e")
                    if key == 'allow_ocr' and not bundled_image_models_available():
                        try:
                            settings_vars[key].set('0')
                            switch.configure(state='disabled')
                        except Exception:
                            pass
                    created_widgets[key] = switch
                elif wtype == 'slider' or key == 'ui_scale':
                    slider = _make_scale_slider(frame, key)
                    slider.grid(row=0, column=1, sticky="e", padx=(0, 4))
                    created_widgets[key] = slider
                else:
                    entry = create_styled_entry(frame, textvariable=settings_vars[key])
                    entry.grid(row=0, column=1, sticky="ew")
                    created_widgets[key] = entry

        for group_name, keys in groups.items():
            items = sort_items([(k, by_key[k]) for k in keys if k in by_key])
            if not items:
                continue
            section = create_styled_frame(parent)
            section.pack(fill="x", pady=(8, 2))
            section.grid_columnconfigure(0, weight=1)
            g_raw = Lang.get(f"settings_group_{group_name}", default=group_name)
            g_label = str(g_raw).upper()
            head_btn = CTkButton(
                section,
                text="╭ " + g_label,
                anchor="w",
                fg_color="transparent",
                hover_color=PURPLE_ACCENT,
                corner_radius=CORNER_RADIUS,
                font=(FONT_FAMILY, 13),
                height=28,
                text_color=WHITE,  # white / theme primary (not accent purple)
            )
            head_btn.grid(row=0, column=0, sticky="ew")
            body = create_styled_frame(section, fg_color="transparent")
            body.grid(row=1, column=0, sticky="ew", padx=(8, 0), pady=(2, 0))
            body.grid_columnconfigure(0, weight=1)
            for key, wtype in items:
                add_setting_row(body, key, wtype)
            # start expanded; click header to collapse
            def make_section_toggle(btn=head_btn, body_fr=body, label=g_label):
                def toggle():
                    if body_fr.winfo_ismapped():
                        body_fr.grid_remove()
                        btn.configure(text="╰ " + label)
                    else:
                        body_fr.grid()
                        btn.configure(text="╭ " + label)
                return toggle
            head_btn.configure(command=make_section_toggle())
        return created_widgets

    class DynamicModelUI:
        def __init__(self):
            self.provider_param_full_paths = {}
            self.model_frames = {}
            self.provider_manager = ProviderManager()
            self.providers = self.provider_manager.get_providers()
            self.real_token_values = {}
            self.real_pwd_status = {}
            # small model: separate widget namespace (same provider type as large is OK)
            self.small_provider_param_full_paths = {}
            self.small_model_frames = {}
            self.small_real_token_values = {}
            self.small_real_pwd_status = {}
            self.small_settings_vars = {}  # module_name -> {param: StringVar}
        def _ensure_dual_model_vars(self):
            """Defaults: large-only; small off."""
            defaults = {
                'use_small_model': '0',
                'small_model_type': '',
                'small_model_provider_params': '',
                'small_token_limit': self.settings_vars.get('token_limit', tk.StringVar(value='8192')).get() if self.settings_vars.get('token_limit') else '8192',
                'small_max_token_limit': self.settings_vars.get('max_token_limit', tk.StringVar(value='8192')).get() if self.settings_vars.get('max_token_limit') else '8192',
                'small_for_cutter_only': '1',
                'small_agent_until_protocol': '0',
                'small_protocol_drop_error': '0',
            }
            for k, v in defaults.items():
                if k not in self.settings_vars:
                    self.settings_vars[k] = tk.StringVar(value=str(v))
        def _create_model_ui(self, parent):
            self._ensure_dual_model_vars()
            # ----- LARGE (primary) -----
            model_type_frame = create_styled_frame(parent)
            model_type_frame.pack(fill="x", pady=10)
            create_styled_label(model_type_frame, text=Lang.get("model_type_large", default=Lang.get("model_type"))).pack(side="top")
            provider_items = list(self.providers.items())
            if not provider_items:
                create_styled_label(parent, text=Lang.get("no_providers_found")).pack()
            else:
                # dropdown instead of radio row (faster UI, less clutter)
                names = [m for m, _ in provider_items]
                labels = [self.providers[m]['name'] for m in names]
                self._provider_label_to_module = dict(zip(labels, names))
                self._provider_module_to_label = dict(zip(names, labels))
                cur = self.settings_vars['model_type'].get()
                if not cur or cur not in names:
                    cur = names[0]
                    self.settings_vars['model_type'].set(cur)
                self._large_provider_label_var = tk.StringVar(value=self._provider_module_to_label.get(cur, labels[0]))
                def _on_large_provider(choice):
                    mod = self._provider_label_to_module.get(choice, choice)
                    self.settings_vars['model_type'].set(mod)
                    self.toggle_model_frames()
                # ~1/3 narrower than full width
                CTkOptionMenu(
                    parent, variable=self._large_provider_label_var, values=labels,
                    command=_on_large_provider, width=220, **OPTIONMENU_THEME
                ).pack(anchor="w", pady=5)
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
            # text_cutter limits live on Chat settings tab (not here)
            if 'text_cutter_token_limit' not in self.settings_vars:
                self.settings_vars['text_cutter_token_limit'] = tk.StringVar(value='2000')
            if 'max_incoming_tokens' not in self.settings_vars:
                self.settings_vars['max_incoming_tokens'] = tk.StringVar(value='10000')
            self.toggle_model_frames()
            # SMALL UI is a separate tab (setup_small_model_tab) when available
        def _create_small_model_ui(self, parent):
            self._ensure_dual_model_vars()
            # no white border — flat section
            box = create_styled_frame(parent, fg_color="transparent", border_width=0)
            box.pack(fill="x", pady=(8, 5), padx=2)
            head = create_styled_frame(box, fg_color="transparent")
            head.pack(fill="x", padx=4, pady=4)
            create_styled_label(head, text=Lang.get("small_model_section", default="Small model (optional)")).pack(side="left")
            sw = CTkSwitch(
                head, text="", variable=self.settings_vars['use_small_model'],
                onvalue="1", offvalue="0", switch_width=50, switch_height=25,
                progress_color=PURPLE_ACCENT, font=FONT_REGULAR,
                command=self._toggle_small_model_section)
            sw.pack(side="right")
            # collapsible description
            desc_head = create_styled_frame(box, fg_color="transparent")
            desc_head.pack(fill="x", padx=4, pady=(0, 2))
            self._small_desc_btn = CTkButton(
                desc_head,
                text="╰ " + Lang.get("small_model_section_about", default="About small model"),
                anchor="w",
                fg_color="transparent",
                hover_color=PURPLE_ACCENT,
                corner_radius=CORNER_RADIUS,
                font=FONT_REGULAR,
                height=24,
                text_color=DARK_TEXT_SECONDARY,
            )
            self._small_desc_btn.pack(fill="x")
            self._small_desc_frame = create_styled_frame(box, fg_color="transparent")
            desc_label = create_styled_label(
                self._small_desc_frame,
                text=Lang.get("small_model_section_desc", default="Off by default. Large is primary; embeddings always from large."),
                wraplength=420, justify="left", text_color=DARK_TEXT_SECONDARY,
            )
            desc_label.pack(anchor="w", padx=8, pady=(0, 4))
            def _toggle_small_desc():
                if self._small_desc_frame.winfo_ismapped():
                    self._small_desc_frame.pack_forget()
                    self._small_desc_btn.configure(
                        text="╰ " + Lang.get("small_model_section_about", default="About small model"))
                else:
                    self._small_desc_frame.pack(fill="x", after=desc_head)
                    self._small_desc_btn.configure(
                        text="╭ " + Lang.get("small_model_section_about", default="About small model"))
            self._small_desc_btn.configure(command=_toggle_small_desc)
            # start collapsed
            # (desc_frame not packed yet)
            self.small_details = create_styled_frame(box, fg_color="transparent")
            self.small_details.pack(fill="x", padx=6, pady=4)
            # options
            opt = create_styled_frame(self.small_details, fg_color="transparent")
            opt.pack(fill="x", pady=2)
            create_styled_label(opt, text=Lang.get("small_for_cutter_only", default="Small only for cutter/summaries")).pack(side="left")
            CTkSwitch(opt, text="", variable=self.settings_vars['small_for_cutter_only'],
                      onvalue="1", offvalue="0", switch_width=50, switch_height=25,
                      progress_color=PURPLE_ACCENT).pack(side="right")
            opt2 = create_styled_frame(self.small_details, fg_color="transparent")
            opt2.pack(fill="x", pady=2)
            create_styled_label(opt2, text=Lang.get("small_agent_until_protocol", default="Agent on small; large after protocol fail")).pack(side="left")
            CTkSwitch(opt2, text="", variable=self.settings_vars['small_agent_until_protocol'],
                      onvalue="1", offvalue="0", switch_width=50, switch_height=25,
                      progress_color=PURPLE_ACCENT).pack(side="right")
            opt3 = create_styled_frame(self.small_details, fg_color="transparent")
            opt3.pack(fill="x", pady=2)
            create_styled_label(
                opt3,
                text=Lang.get(
                    "small_protocol_drop_error",
                    default="On protocol fail: drop error, retry on large",
                ),
            ).pack(side="left")
            CTkSwitch(
                opt3, text="", variable=self.settings_vars['small_protocol_drop_error'],
                onvalue="1", offvalue="0", switch_width=50, switch_height=25,
                progress_color=PURPLE_ACCENT,
            ).pack(side="right")
            create_styled_label(self.small_details, text=Lang.get("small_model_type", default="Small provider")).pack(anchor="w", pady=(6, 0))
            s_names = list(self.providers.keys())
            s_labels = [self.providers[m]['name'] for m in s_names]
            self._small_label_to_module = dict(zip(s_labels, s_names))
            self._small_module_to_label = dict(zip(s_names, s_labels))
            scur = self.settings_vars['small_model_type'].get()
            if not scur or scur not in s_names:
                scur = self.settings_vars['model_type'].get() if self.settings_vars['model_type'].get() in s_names else (s_names[0] if s_names else '')
                if scur:
                    self.settings_vars['small_model_type'].set(scur)
            self._small_provider_label_var = tk.StringVar(
                value=self._small_module_to_label.get(scur, s_labels[0] if s_labels else ''))
            def _on_small_provider(choice):
                mod = self._small_label_to_module.get(choice, choice)
                self.settings_vars['small_model_type'].set(mod)
                self._toggle_small_model_frames()
            if s_labels:
                CTkOptionMenu(
                    self.small_details, variable=self._small_provider_label_var, values=s_labels,
                    command=_on_small_provider, width=220, **OPTIONMENU_THEME
                ).pack(anchor="w", pady=4)
            self.small_frames_container = create_styled_frame(self.small_details)
            self.small_frames_container.pack(fill="x", pady=4)
            self._create_small_provider_frames(self.small_frames_container)
            st = create_styled_frame(self.small_details)
            st.pack(fill="x", pady=4)
            create_styled_label(st, text=Lang.get("small_token_limit", default="Small token limit")).pack(side="left", padx=(0, 8))
            create_styled_entry(st, textvariable=self.settings_vars['small_token_limit']).pack(side="left", fill="x", expand=True)
            btn_row = create_styled_frame(self.small_details, fg_color="transparent")
            btn_row.pack(fill="x", pady=4)
            create_styled_button(
                btn_row, text=Lang.get("small_copy_from_large", default="Copy large params (except model)"),
                command=self._copy_large_params_to_small).pack(side="left", padx=2)
            create_styled_button(
                btn_row, text=Lang.get("validate_small_model", default="Validate small"),
                command=self._validate_small_model).pack(side="left", padx=2)
            # seed small type from large if empty and enabled later
            if not self.settings_vars['small_model_type'].get() and self.settings_vars['model_type'].get():
                self.settings_vars['small_model_type'].set(self.settings_vars['model_type'].get())
            self._load_small_provider_params()
            self._toggle_small_model_frames()
            self._toggle_small_model_section()
        def _toggle_small_model_section(self):
            on = self.settings_vars.get('use_small_model') and self.settings_vars['use_small_model'].get() == '1'
            if not hasattr(self, 'small_details'):
                return
            try:
                if on:
                    self.small_details.pack(fill="x", padx=6, pady=4)
                else:
                    self.small_details.pack_forget()
            except Exception:
                pass
        def _create_small_provider_frames(self, container):
            try:
                for module_name, p_data in self.providers.items():
                    outer, content = create_braced_content(container, height=132, side='left')
                    self.small_model_frames[module_name] = outer
                    params = p_data.get('params', [])
                    scroll_frame = create_scrollable_frame(
                        content, fg_color="transparent", border_width=0, corner_radius=0)
                    scroll_frame.pack(fill="both", expand=True, padx=8, pady=1.5)
                    self.small_provider_param_full_paths.setdefault(module_name, {})
                    self.small_settings_vars.setdefault(module_name, {})
                    for param in params:
                        def make_callback(m_name=module_name):
                            return lambda p_name: self._on_small_param_change(p_name, m_name)
                        create_param_widget(
                            scroll_frame, param, self.small_settings_vars[module_name],
                            self.small_provider_param_full_paths[module_name], make_callback())
                    outer.pack(fill="x", padx=4, pady=4)
            except Exception as e:
                print(f"Error creating small model frames: {e}")
        def _on_small_param_change(self, param_name, provider_name):
            if param_name not in ["api_token", "token", "password"]:
                return
            val = self.small_settings_vars.get(provider_name, {}).get(param_name)
            if not val or val.get() == "•••":
                return
            if param_name in ["api_token", "token"]:
                pwd_var = self.small_settings_vars.get(provider_name, {}).get("password")
                if pwd_var and pwd_var.get() == "•••":
                    pwd_var.set("")
            if param_name == "password":
                tok_var = self.small_settings_vars.get(provider_name, {}).get("api_token") or self.small_settings_vars.get(provider_name, {}).get("token")
                if tok_var and tok_var.get() == "•••":
                    tok_var.set("")
        def _toggle_small_model_frames(self):
            selected = self.settings_vars['small_model_type'].get() if self.settings_vars.get('small_model_type') else ""
            for name, frame in self.small_model_frames.items():
                try:
                    if name == selected and frame.winfo_exists():
                        frame.pack(fill="x", padx=4, pady=4)
                    elif frame.winfo_exists():
                        frame.pack_forget()
                except Exception:
                    continue
            self._load_small_provider_params()
        def _apply_params_string_to_small_provider(self, provider_module: str, params_str: str):
            if not provider_module or not params_str:
                return
            try:
                params_map = dict(part.split('=', 1) for part in params_str.split(';') if '=' in part)
                ui_vars = self.small_settings_vars.get(provider_module, {})
                path_vars = self.small_provider_param_full_paths.get(provider_module, {})
                provider_info = self.providers.get(provider_module, {})
                for param_info in provider_info.get('params', []):
                    param_name = param_info['name']
                    value = params_map.get(param_name, '')
                    if param_info.get('is_file'):
                        if param_name in path_vars:
                            path_vars[param_name].set(value)
                    elif param_name in ui_vars:
                        if param_name in ["api_token", "token"]:
                            if value:
                                self.small_real_token_values[provider_module] = value
                                ui_vars[param_name].set("•••")
                            else:
                                ui_vars[param_name].set("")
                        elif param_name == "password":
                            self.small_real_pwd_status[provider_module] = value
                            ui_vars[param_name].set("•••" if value == "set" else "")
                        elif _param_is_bool(param_info):
                            ui_vars[param_name].set(
                                _normalize_bool_param_value(value, param_info.get('default')))
                        else:
                            ui_vars[param_name].set(value)
            except Exception as e:
                print(f"small params apply error: {e}")
        def _load_small_provider_params(self):
            prov = self.settings_vars['small_model_type'].get() if self.settings_vars.get('small_model_type') else ""
            pstr = self.settings_vars.get('small_model_provider_params')
            pstr = pstr.get() if pstr is not None else ""
            if prov and pstr:
                self._apply_params_string_to_small_provider(prov, pstr)
        def _build_small_connection_string(self) -> str:
            provider_module_name = self.settings_vars['small_model_type'].get() if self.settings_vars.get('small_model_type') else ""
            if not provider_module_name:
                return ""
            provider_data = self.providers.get(provider_module_name)
            if not provider_data:
                return ""
            parts = []
            self._last_small_plain_password = None
            ui_vars = self.small_settings_vars.get(provider_module_name, {})
            path_vars = self.small_provider_param_full_paths.get(provider_module_name, {})
            for param in provider_data.get('params', []):
                param_name = param['name']
                value = ""
                if param.get('is_file'):
                    if param_name in path_vars:
                        value = path_vars[param_name].get().strip()
                elif param_name in ui_vars:
                    value = ui_vars[param_name].get().strip()
                if param_name in ["api_token", "token"]:
                    if value == "•••":
                        value = self.small_real_token_values.get(provider_module_name, "")
                    else:
                        raw_token = value
                        raw_pwd = ui_vars.get("password", tk.StringVar()).get() if ui_vars.get("password") else ""
                        if raw_pwd == "•••":
                            raw_pwd = ""
                        if raw_pwd and raw_token:
                            value = encryption_utils.encrypt_token(raw_token, raw_pwd)
                            self._last_small_plain_password = raw_pwd
                        else:
                            value = raw_token
                elif param_name == "password":
                    if value == "•••":
                        value = self.small_real_pwd_status.get(provider_module_name, "")
                    else:
                        if value:
                            self._last_small_plain_password = value
                            value = "set"
                        else:
                            value = "empty"
                if not value:
                    default_val = param.get('default')
                    if default_val is not None:
                        if _param_is_bool(param):
                            value = _normalize_bool_param_value(default_val, default_val)
                        else:
                            value = str(default_val)
                if value:
                    # bool params always lowercase true/false in connect string
                    if _param_is_bool(param):
                        value = _normalize_bool_param_value(value, param.get('default'))
                    parts.append(f"{param_name}={value}")
            return ";".join(parts)
        def _copy_large_params_to_small(self):
            """Same provider as large → copy all params except model name into small widgets."""
            large_type = self.settings_vars['model_type'].get()
            if not large_type:
                return
            large_conn = self._build_connection_string()
            self.settings_vars['small_model_type'].set(large_type)
            # drop model= from copy so user must set small model (or keep if already set)
            small_map = {}
            for part in large_conn.split(';'):
                if '=' not in part:
                    continue
                k, v = part.split('=', 1)
                kl = k.strip().lower()
                if kl == 'model':
                    continue
                small_map[kl] = v.strip()
            # preserve existing small model name if any
            cur_small = self._build_small_connection_string()
            for part in cur_small.split(';'):
                if part.lower().startswith('model='):
                    small_map['model'] = part.split('=', 1)[1].strip()
            if 'model' not in small_map:
                small_map['model'] = ''  # user fills
            pstr = ";".join(f"{k}={v}" for k, v in small_map.items())
            self.settings_vars['small_model_provider_params'].set(pstr)
            self._toggle_small_model_frames()
            self._apply_params_string_to_small_provider(large_type, pstr)
            # token limit default from large if empty-ish
            try:
                self.settings_vars['small_token_limit'].set(self.settings_vars['token_limit'].get())
            except Exception:
                pass
        def _validate_small_model(self):
            if self.settings_vars.get('use_small_model') and self.settings_vars['use_small_model'].get() != '1':
                showinfo(self, Lang.get("info", default="Info"), Lang.get("small_model_disabled_hint", default="Enable small model first."))
                return
            model_type = self.settings_vars['small_model_type'].get()
            connection_string = self._build_small_connection_string()
            self.settings_vars['small_model_provider_params'].set(connection_string)
            plain = getattr(self, '_last_small_plain_password', None)
            valid, msg, max_tokens = self.backend.validate_model_settings(model_type, connection_string, plain)
            if valid:
                try:
                    self.settings_vars['small_max_token_limit'].set(str(max_tokens))
                    cur = int(self.settings_vars['small_token_limit'].get() or max_tokens)
                    if cur > int(max_tokens):
                        self.settings_vars['small_token_limit'].set(str(max_tokens))
                except Exception:
                    pass
                if plain:
                    encryption_utils.SESSION_PASSWORDS[f"small_{model_type}"] = plain
                showinfo(self, Lang.get("success"), msg)
            else:
                showerror(self, Lang.get("validation_error"), msg)
        def _clamp_cutter_to_model_limits(self):
            """If cutter/incoming caps exceed model ctx (large, or small when it runs cutter), auto-reduce."""
            try:
                large_tl = int(self.settings_vars['token_limit'].get() or 8192)
            except Exception:
                large_tl = 8192
            cutter_ctx = large_tl
            use_small = self.settings_vars.get('use_small_model') and self.settings_vars['use_small_model'].get() == '1'
            if use_small:
                # small used for cutter by default (small_for_cutter_only or always when small on for non-agent)
                try:
                    stl = int(self.settings_vars.get('small_token_limit', tk.StringVar(value=str(large_tl))).get() or large_tl)
                except Exception:
                    stl = large_tl
                cutter_ctx = min(large_tl, stl)
            room = max(500, cutter_ctx - 1000)
            for key, default in (('text_cutter_token_limit', 2000), ('max_incoming_tokens', 10000)):
                if key not in self.settings_vars:
                    continue
                try:
                    val = int(self.settings_vars[key].get() or default)
                except Exception:
                    val = default
                if val > room:
                    self.settings_vars[key].set(str(room))
                    let_log = print
                    try:
                        let_log(f"[ui] clamped {key} to {room} (model ctx {cutter_ctx})")
                    except Exception:
                        pass
        def collect_dual_model_settings(self) -> dict:
            """Settings keys for large + optional small (for save / create chat)."""
            self._ensure_dual_model_vars()
            self._clamp_cutter_to_model_limits()
            large_conn = self._build_connection_string()
            out = {
                'model_type': self.settings_vars['model_type'].get(),
                'model_provider_params': large_conn,
                'token_limit': self.settings_vars['token_limit'].get(),
                'max_token_limit': self.settings_vars.get('max_token_limit', tk.StringVar(value='8192')).get(),
                'use_small_model': self.settings_vars['use_small_model'].get(),
                'small_for_cutter_only': self.settings_vars['small_for_cutter_only'].get(),
                'small_agent_until_protocol': self.settings_vars['small_agent_until_protocol'].get(),
                'small_protocol_drop_error': self.settings_vars.get(
                    'small_protocol_drop_error', tk.StringVar(value='0')).get(),
                'text_cutter_token_limit': self.settings_vars.get('text_cutter_token_limit', tk.StringVar(value='2000')).get(),
                'max_incoming_tokens': self.settings_vars.get('max_incoming_tokens', tk.StringVar(value='10000')).get(),
            }
            if out['use_small_model'] == '1':
                s_conn = self._build_small_connection_string()
                out['small_model_type'] = self.settings_vars['small_model_type'].get()
                out['small_model_provider_params'] = s_conn
                out['small_token_limit'] = self.settings_vars['small_token_limit'].get()
                out['small_max_token_limit'] = self.settings_vars.get('small_max_token_limit', tk.StringVar(value='8192')).get()
                self.settings_vars['small_model_provider_params'].set(s_conn)
            else:
                out['small_model_type'] = self.settings_vars['small_model_type'].get() or ''
                out['small_model_provider_params'] = self.settings_vars['small_model_provider_params'].get() or ''
                out['small_token_limit'] = self.settings_vars['small_token_limit'].get() or ''
                out['small_max_token_limit'] = self.settings_vars.get('small_max_token_limit', tk.StringVar(value='')).get() or ''
            self.settings_vars['model_provider_params'].set(large_conn)
            return out
        def _create_specific_model_frames(self, container):
            try:
                for module_name, p_data in self.providers.items():
                    outer, content = create_braced_content(container, height=132, side='left')
                    self.model_frames[module_name] = outer
                    params = p_data.get('params', [])
                    scroll_frame = create_scrollable_frame(
                        content, fg_color="transparent", border_width=0, corner_radius=0)
                    scroll_frame.pack(fill="both", expand=True, padx=10, pady=1.5)
                    content_parent = scroll_frame
                    self.provider_param_full_paths.setdefault(module_name, {})
                    self.settings_vars.setdefault(module_name, {})
                    for param in params:
                        def make_callback(m_name=module_name): return lambda p_name: self.on_param_change(p_name, m_name)
                        create_param_widget(content_parent, param, self.settings_vars[module_name], self.provider_param_full_paths[module_name], make_callback())
                    outer.pack(fill="x", padx=5, pady=5)
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
        def _get_provider_params_map(self) -> dict:
            """JSON map provider_module → connection string (prevents cross-provider bleed)."""
            raw = ""
            try:
                var = self.settings_vars.get('provider_params_by_type')
                raw = var.get() if var is not None else ""
            except Exception:
                raw = ""
            if not raw:
                return {}
            try:
                data = json.loads(raw)
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
        def _set_provider_params_map(self, data: dict):
            if 'provider_params_by_type' not in self.settings_vars:
                self.settings_vars['provider_params_by_type'] = tk.StringVar(value="{}")
            try:
                self.settings_vars['provider_params_by_type'].set(json.dumps(data or {}, ensure_ascii=False))
            except Exception:
                self.settings_vars['provider_params_by_type'].set("{}")
        def _remember_provider_params(self, provider_module: str, params_str: str):
            if not provider_module:
                return
            m = self._get_provider_params_map()
            if params_str:
                m[provider_module] = params_str
            self._set_provider_params_map(m)
            if 'model_provider_params' in self.settings_vars and params_str is not None:
                # worker always reads model_provider_params for the *active* provider
                if self.settings_vars.get('model_type') and self.settings_vars['model_type'].get() == provider_module:
                    self.settings_vars['model_provider_params'].set(params_str)
        def _apply_params_string_to_provider(self, provider_module: str, params_str: str):
            """Fill only the given provider's widgets from its own connection string."""
            if not provider_module or not params_str:
                return
            try:
                params_map = dict(part.split('=', 1) for part in params_str.split(';') if '=' in part)
                provider_ui_vars = self.settings_vars.get(provider_module, {})
                provider_path_vars = self.provider_param_full_paths.get(provider_module, {})
                provider_info = self.providers.get(provider_module, {})
                if not provider_info:
                    return
                for param_info in provider_info.get('params', []):
                    param_name = param_info['name']
                    value = params_map.get(param_name, '')
                    if param_info.get('is_file'):
                        if param_name in provider_path_vars:
                            provider_path_vars[param_name].set(value)
                    else:
                        if param_name in provider_ui_vars:
                            if param_name in ["api_token", "token"]:
                                if value:
                                    self.real_token_values[provider_module] = value
                                    provider_ui_vars[param_name].set("•••")
                                else:
                                    provider_ui_vars[param_name].set("")
                            elif param_name == "password":
                                self.real_pwd_status[provider_module] = value
                                if value == "set":
                                    provider_ui_vars[param_name].set("•••")
                                else:
                                    provider_ui_vars[param_name].set("")
                            elif _param_is_bool(param_info):
                                # Switch onvalue/offvalue are lowercase true/false
                                provider_ui_vars[param_name].set(
                                    _normalize_bool_param_value(value, param_info.get('default')))
                            else:
                                provider_ui_vars[param_name].set(value)
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not parse provider params string for {provider_module}: {params_str}. Error: {e}")
        def _load_provider_params_for(self, provider_module: str):
            """Load params only for one provider from namespaced map (no cross-bleed)."""
            if not provider_module:
                return
            m = self._get_provider_params_map()
            params_str = m.get(provider_module, "")
            # Migration: if map empty for active provider, seed from model_provider_params once
            if not params_str:
                active = self.settings_vars.get('model_type')
                active_name = active.get() if active is not None else ""
                if provider_module == active_name:
                    mp = self.settings_vars.get('model_provider_params')
                    params_str = mp.get() if mp is not None else ""
                    if params_str:
                        self._remember_provider_params(provider_module, params_str)
            if params_str:
                self._apply_params_string_to_provider(provider_module, params_str)
        def _load_provider_params_from_string(self):
            """Load active provider only (compat name used across UI). Never fill other providers from active string."""
            current = self.settings_vars['model_type'].get() if self.settings_vars.get('model_type') else ""
            # Seed map from legacy single string if needed, then load every known entry into its own widgets
            m = self._get_provider_params_map()
            if not m and current:
                mp = self.settings_vars.get('model_provider_params')
                legacy = mp.get() if mp is not None else ""
                if legacy:
                    self._remember_provider_params(current, legacy)
                    m = self._get_provider_params_map()
            for prov, pstr in m.items():
                if prov in self.providers:
                    self._apply_params_string_to_provider(prov, pstr)
            # ensure active visible fields filled
            if current:
                self._load_provider_params_for(current)
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
                    if default_val is not None:
                        if _param_is_bool(param):
                            value = _normalize_bool_param_value(default_val, default_val)
                        else:
                            value = str(default_val)
                if value:
                    if _param_is_bool(param):
                        value = _normalize_bool_param_value(value, param.get('default'))
                    parts.append(f"{param_name}={value}")
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
            # only the selected provider's own cached params — never re-apply active string to another
            self._load_provider_params_for(selected_type)
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
        def __init__(self, master, backend, title_key="settings_title", geometry="510x390", preload=False):
            BaseTopLevel.__init__(self, master, preload=preload)
            DynamicModelUI.__init__(self)
            self.master = master
            self.backend = backend
            self.title(Lang.get(title_key))
            self._default_geometry = geometry
            self.geometry(geometry)
            # start size = geometry; min = ~2/3 so user can shrink with mouse but not to a speck
            try:
                gw, gh = geometry.lower().split("x", 1)
                gw, gh = int(gw), int(gh)
                self.minsize(max(320, int(gw * 2 / 3)), max(240, int(gh * 2 / 3)))
            except Exception:
                self.minsize(320, 240)
            self.configure(fg_color=DARK_BG)
            self.max_tokens = 8192
            self.validated = True
            self.grid_columnconfigure(0, weight=1)
            self.grid_rowconfigure(0, weight=1)
        def refresh_from_global(self):
            """Reload StringVars from backend global settings (keys already present)."""
            try:
                settings = self.backend.get_global_settings()
            except Exception:
                return
            if not getattr(self, 'settings_vars', None):
                return
            for key, val in settings.items():
                if key == 'chat_name':
                    continue
                var = self.settings_vars.get(key)
                if var is None or not hasattr(var, 'set'):
                    continue
                try:
                    var.set(str(val) if val is not None else '')
                except Exception:
                    pass
            try:
                if hasattr(self, '_load_provider_params_from_string'):
                    self._load_provider_params_from_string()
            except Exception:
                pass
            try:
                if hasattr(self, 'toggle_model_frames'):
                    self.toggle_model_frames()
            except Exception:
                pass
            try:
                if hasattr(self, '_toggle_small_model_frames'):
                    self._toggle_small_model_frames()
            except Exception:
                pass
            try:
                if hasattr(self, '_toggle_small_model_section'):
                    self._toggle_small_model_section()
            except Exception:
                pass
            # Settings mods tab: pending flags from RAM (no DB re-read)
            try:
                if hasattr(self, 'pending_default_mods'):
                    self.pending_default_mods = {}
                if hasattr(self, 'scrollable_frame') and self.scrollable_frame.winfo_exists():
                    if hasattr(self, 'rebuild_mods_list'):
                        self.rebuild_mods_list()
            except Exception:
                pass
        def setup_model_tab(self, parent):
            parent.grid_columnconfigure(0, weight=1)
            parent.grid_rowconfigure(0, weight=1)
            scroll = create_scrollable_frame(parent, fg_color="transparent", corner_radius=0, border_width=0)
            scroll.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
            main_frame = create_styled_frame(scroll)
            main_frame.pack(fill="both", expand=True, padx=2, pady=2)
            main_frame.grid_columnconfigure(0, weight=1)
            self._create_model_ui(main_frame)
        def setup_small_model_tab(self, parent):
            parent.grid_columnconfigure(0, weight=1)
            parent.grid_rowconfigure(0, weight=1)
            scroll = create_scrollable_frame(parent, fg_color="transparent", corner_radius=0, border_width=0)
            scroll.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
            if not hasattr(self, 'settings_vars') or not self.settings_vars:
                self.settings_vars = self._get_default_settings()
            self._create_small_model_ui(scroll)
            try:
                if hasattr(self, 'small_details'):
                    self.small_details.pack(fill="x", padx=6, pady=4)
            except Exception:
                pass
        def setup_chat_settings_tab(self, parent):
            parent.grid_columnconfigure(0, weight=1)
            parent.grid_rowconfigure(0, weight=1)
            # no outer margin — settings content flush to tab edges
            scrollable_frame = create_scrollable_frame(parent, fg_color="transparent", corner_radius=0, border_width=0)
            scrollable_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
            scrollable_frame.grid_columnconfigure(0, weight=1)
            # Выбор папки чатов — только в главных настройках (SettingsWindow), не при создании чата
            if getattr(self, '_show_chats_dir_picker', False):
                self._build_chats_dir_picker(scrollable_frame)
            # read-only: версия приложения (release)
            try:
                from cross_gpt import RELEASE_VERSION as _app_rel
            except Exception:
                _app_rel = "2026-07"
            rel_frame = create_styled_frame(scrollable_frame)
            rel_frame.pack(fill="x", pady=4)
            create_styled_label(
                rel_frame,
                text=f"{Lang.get('release_version')}: {_app_rel}",
                font=FONT_REGULAR).pack(anchor="w", padx=2)
            # MCP: явно в шапке вкладки (также в metadata как mcp_url)
            if 'mcp_url' not in self.settings_vars:
                self.settings_vars['mcp_url'] = tk.StringVar(
                    value=self.backend.get_global_settings().get('mcp_url', ''))
            metadata = self.backend.get_settings_metadata()
            # гарантируем widget_type для mcp_url в UI даже на старых БД
            if 'mcp_url' not in metadata:
                metadata = dict(metadata)
                metadata['mcp_url'] = 'entry'
            # ui_scale as slider even if old DB has entry
            if metadata.get('ui_scale') != 'slider':
                metadata = dict(metadata)
                metadata['ui_scale'] = 'slider'
            self.created_widgets = build_chat_settings_ui(scrollable_frame, self.settings_vars, metadata)
        def _build_chats_dir_picker(self, parent):
            if 'chats_dir' not in self.settings_vars:
                self.settings_vars['chats_dir'] = tk.StringVar(value=self.backend.get_chats_root())
            frame = create_styled_frame(parent)
            frame.pack(fill="x", pady=4)
            frame.grid_columnconfigure(0, weight=1)
            btn_text = "╰ " + Lang.get("chats_dir", default="chats_dir")
            title_btn = CTkButton(
                frame, text=btn_text, anchor="w", fg_color="transparent",
                hover_color=PURPLE_ACCENT, corner_radius=CORNER_RADIUS, font=FONT_REGULAR, height=27)
            title_btn.grid(row=0, column=0, sticky="ew", padx=(0, 10))
            path_row = create_styled_frame(frame)
            path_row.grid(row=0, column=1, sticky="ew")
            path_row.grid_columnconfigure(0, weight=1)
            entry = create_styled_entry(path_row, textvariable=self.settings_vars['chats_dir'])
            entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
            def browse():
                chosen = filedialog.askdirectory(initialdir=self.settings_vars['chats_dir'].get() or suggest_default_chats_dir())
                if chosen:
                    self.settings_vars['chats_dir'].set(chosen)
            create_styled_button(path_row, text=Lang.get("browse"), width=80, command=browse).grid(row=0, column=1)
            desc_frame = create_styled_frame(frame, fg_color="transparent")
            desc_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))
            desc_label = create_styled_label(desc_frame, text=Lang.get("chats_dir_desc", default=""), wraplength=400, justify="left")
            desc_label.pack(fill="x", padx=5, pady=2)
            def toggle():
                if desc_frame.winfo_ismapped():
                    desc_frame.grid_remove()
                    title_btn.configure(text="╰ " + Lang.get("chats_dir", default="chats_dir"))
                else:
                    desc_frame.grid()
                    title_btn.configure(text="╭ " + Lang.get("chats_dir", default="chats_dir"))
            title_btn.configure(command=toggle)
            desc_label.bind("<Button-1>", lambda e: toggle())
            desc_frame.grid_remove()
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
            if has_token_param and not has_pwd_param:
                showerror(self, Lang.get("error", default="Error"), "Провайдер не валиден: отсутствует параметр password при наличии api_token."); return
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
                self._remember_provider_params(model_type, connection_string)
            else: self.validated = False; showerror(self, Lang.get("validation_error"), msg)
        def _get_default_settings(self):
            settings = self.backend.get_global_settings()
            if 'max_token_limit' not in settings: settings['max_token_limit'] = '8192'
            if 'provider_params_by_type' not in settings: settings['provider_params_by_type'] = '{}'
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
            self.minsize(400, 250)
            try:
                self.resizable(True, True)
            except Exception:
                pass
            self.protocol("WM_DELETE_WINDOW", self.on_close)
            self._messages_scroll_gen = 0
            self.after(0, lambda: set_windows_dark_titlebar(self))
            self._needs_initial_setup = not self.backend.is_main_config_complete()
            if self._needs_initial_setup:
                # main stays hidden; InitialSettings is the only window (do not deiconify main later)
                self.after(0, self.show_initial_settings)
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
            # after map/deiconify, message scroll metrics need a second pass
            for delay in (150, 400, 900):
                try:
                    self.after(delay, self._startup_messages_scroll_fix)
                except Exception:
                    pass
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
            """Hide UI first (no lag), then stop workers / destroy on next idle tick."""
            active_chats = list(self.chat_processes.keys())
            if active_chats:
                if not askyesno(
                    self,
                    Lang.get("active_chats_on_close_title"),
                    Lang.get("active_chats_on_close_message", count=len(active_chats)),
                ):
                    return
            # Instant: window disappears before terminate/join work
            try:
                self.withdraw()
            except Exception:
                pass
            def _finish_quit(chats=list(active_chats)):
                for chat_id in chats:
                    try:
                        self.terminate_chat_process(chat_id)
                    except Exception:
                        pass
                try:
                    self.destroy()
                except Exception:
                    pass
            try:
                self.after(1, _finish_quit)
            except Exception:
                _finish_quit()
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
                is_blinking = self.chat_blink_states.get(chat_id, False)
                is_hover = bool(getattr(chat_button, '_hover', False))
                if is_hover:
                    color = PURPLE_ACCENT
                elif self.current_chat_id == chat_id:
                    color = PURPLE_ACCENT
                elif chat_id in self.chat_blink_states:
                    # мигание: чёрный ↔ фиолетовый
                    color = ACTIVE_CHAT_COLOR if is_blinking else BLINK_CHAT_COLOR_OFF
                else:
                    color = "transparent"
                chat_button.configure(fg_color=color)
        def update_chat_list_name(self, chat_id, new_name):
            """Update one row label only (no full list rebuild)."""
            if not hasattr(self, 'chats_list_frame') or not self.chats_list_frame.winfo_exists():
                return False
            for row_frame in self.chats_list_frame.winfo_children():
                try:
                    if not isinstance(row_frame, CTkFrame) or not row_frame.winfo_children():
                        continue
                    chat_button = row_frame.winfo_children()[0]
                    if getattr(chat_button, 'chat_id', None) != chat_id:
                        continue
                    chat_button.configure(text=new_name)
                    return True
                except Exception:
                    continue
            return False
        def show_initial_settings(self):
            """First-run wizard. Keep main withdrawn so Linux deiconify race cannot hide the wizard."""
            try:
                self.withdraw()
            except Exception:
                pass
            self._needs_initial_setup = True
            try:
                win = InitialSettingsWindow(self, self.backend)
                try:
                    win.deiconify()
                    win.lift()
                    win.focus_force()
                    win.grab_set()
                except Exception:
                    pass
            except Exception as e:
                import traceback
                print(f"InitialSettingsWindow failed: {e}\n{traceback.format_exc()}")
                try:
                    showerror(self, Lang.get("error", default="Error"), f"Initial setup failed:\n{e}")
                except Exception:
                    pass
                try:
                    self.deiconify()
                except Exception:
                    pass
        def refresh_ui_language(self):
            """Update main-window strings after Lang.load_language (without full UI teardown)."""
            try:
                self.title(Lang.get("app_title"))
            except Exception:
                pass
            for attr, key, default in (
                ('log_btn', 'log', 'log'),
                ('send_btn', 'send', 'Send'),
                ('settings_btn', 'settings', 'Settings'),
            ):
                try:
                    w = getattr(self, attr, None)
                    if w is not None and w.winfo_exists():
                        w.configure(text=Lang.get(key, default=default))
                except Exception:
                    pass
            try:
                if hasattr(self, 'input_text') and self.input_text.winfo_exists():
                    # placeholder if supported
                    ph = Lang.get("message_placeholder", default="")
                    if ph:
                        try:
                            self.input_text.configure(placeholder_text=ph)
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                self.update_chat_controls()
            except Exception:
                pass
        def setup_main_ui(self):
            for widget in self.winfo_children():
                try:
                    widget.destroy()
                except Exception:
                    pass
            self.grid_rowconfigure(0, weight=1)
            self.grid_columnconfigure(0, weight=0)
            self.grid_columnconfigure(1, weight=1)
            self.bind("<Control-o>", self.add_attachment)
            if sys.platform == "darwin": self.bind("<Command-o>", self.add_attachment)
            left_panel_container = create_styled_frame(self, width=200)
            left_panel_container.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
            left_panel_container.grid_rowconfigure(0, weight=1)
            left_panel_container.grid_rowconfigure(1, weight=0)
            self.chats_list_frame = create_scrollable_frame(
                left_panel_container, fg_color="transparent", border_width=0, corner_radius=0)
            self.chats_list_frame.grid(row=0, column=0, sticky="nsew", pady=1.5)
            # Скоба справа от списка чатов (пустой список → цвет = фон)
            left_panel_container.grid_columnconfigure(0, weight=1)
            left_panel_container.grid_columnconfigure(1, weight=0)
            self.chat_list_right_wall = tk.Canvas(left_panel_container, width=6, bg=DARK_BG, highlightthickness=0)
            self.chat_list_right_wall.grid(row=0, column=1, sticky="ns")
            def draw_right_wall(event=None):
                try:
                    self.chat_list_right_wall.configure(bg=DARK_BG)
                except Exception:
                    pass
                has_chats = False
                try:
                    has_chats = bool(self.backend.get_chats())
                except Exception:
                    has_chats = False
                # empty list: brace blends with bg
                color = WHITE if has_chats else DARK_BG
                draw_brace_on_canvas(self.chat_list_right_wall, side='right', color=color)
            self._draw_chat_list_brace = draw_right_wall
            self.chat_list_right_wall.bind("<Configure>", draw_right_wall)
            bottom_buttons_frame = create_styled_frame(left_panel_container)
            bottom_buttons_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(3,0))
            self.settings_btn = create_styled_button(bottom_buttons_frame, text="☰", command=self.open_settings, width=20, height=20)
            self.settings_btn.pack(side=tk.LEFT, padx=(0, 2))
            self.new_chat_btn = create_styled_button(bottom_buttons_frame, text="+", command=self.create_chat_window_show, width=20, height=20)
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
            right_panel_container.grid(row=0, column=1, sticky="nsew", padx=(0, 5), pady=(0, 5))
            right_panel_container.grid_columnconfigure(0, weight=1)
            right_panel_container.grid_rowconfigure(0, weight=1)
            right_panel_container.grid_rowconfigure(1, weight=0)
            # Messages flush to top of window (no top gap)
            self.messages_bordered_frame = create_styled_frame(right_panel_container, fg_color=DARK_BG, border_width=0, corner_radius=0)
            self.messages_bordered_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 0))
            self.messages_bordered_frame.grid_rowconfigure(0, weight=1)
            self.messages_bordered_frame.grid_columnconfigure(0, weight=1)
            self.messages_frame = create_scrollable_frame(
                self.messages_bordered_frame, fg_color="transparent", border_width=0, corner_radius=0)
            self.messages_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
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
                try:
                    _style_scrollbar_widget(self.input_text._scrollbar, idle=False)
                except Exception:
                    pass
            if hasattr(self.input_text, '_scrollbar_horizontal'):
                try:
                    _style_scrollbar_widget(self.input_text._scrollbar_horizontal, idle=True)
                except Exception:
                    pass
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
            # Startup layout is flaky: re-sync message scrollbar after first paints
            for delay in (80, 250, 600, 1200):
                try:
                    self.after(delay, self._startup_messages_scroll_fix)
                except Exception:
                    pass
        def _startup_messages_scroll_fix(self):
            """Fix sticky/wrong thumb after app restart (canvas size unknown at first load)."""
            try:
                if not hasattr(self, 'messages_frame') or not self.messages_frame.winfo_exists():
                    return
                if not self.current_chat_id:
                    self._force_no_scroll_state()
                    return
                msgs = self.backend.get_messages(self.current_chat_id)
                if not msgs:
                    self._force_no_scroll_state()
                else:
                    self._scroll_messages_to_bottom()
            except Exception:
                pass
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
        def _pack_chat_row(self, chat):
            """Build one chat list row (used by load_chats and append_chat_row)."""
            row_frame = create_styled_frame(self.chats_list_frame)
            row_frame.pack(fill="x", pady=1)
            row_frame.grid_columnconfigure(0, weight=1)
            chat_button = create_styled_button(
                row_frame, text=chat['name'], anchor="center", fg_color="transparent",
                border_width=0, command=lambda c_id=chat["id"]: self.on_chat_select(c_id))
            # icon/chat buttons: padx left=1, right=0
            _btn_pad = (1, 0)
            chat_button.grid(row=0, column=0, sticky="ew", padx=_btn_pad)
            setattr(chat_button, "chat_id", chat["id"])
            chat_button._hover = False
            def _on_chat_enter(e, btn=chat_button):
                btn._hover = True
                try: btn.configure(fg_color=PURPLE_ACCENT)
                except tk.TclError: pass
            def _on_chat_leave(e, btn=chat_button):
                btn._hover = False
                self.update_chat_list_colors()
            chat_button.bind("<Enter>", _on_chat_enter)
            chat_button.bind("<Leave>", _on_chat_leave)
            chat_root = self.backend.chat_folder(chat["id"])
            reports_path = chat_root / "reports"
            results_path = chat_root / "results"
            has_reports = reports_path.exists() and reports_path.is_dir()
            has_results = results_path.exists() and results_path.is_dir()
            column_offset = 1
            # Round icon chips (r / ƒ / ✔ / ✘) — corner_radius 50 → full pill/circle at 18×18
            _icon_kw = dict(
                width=18, height=18, fg_color="transparent", hover_color=PURPLE_ACCENT,
                corner_radius=50, text_color=WHITE, border_width=0,
                round_width_to_even_numbers=False, round_height_to_even_numbers=False)
            files_button = CTkButton(row_frame, text="ƒ", command=lambda c_id=chat["id"]: self.open_folder(c_id, "files"), **_icon_kw)
            files_button.grid(row=0, column=column_offset, padx=_btn_pad)
            column_offset += 1
            if has_reports:
                reports_button = CTkButton(row_frame, text="r", command=lambda c_id=chat["id"]: self.open_folder(c_id, "reports"), **_icon_kw)
                reports_button.grid(row=0, column=column_offset, padx=_btn_pad)
                column_offset += 1
            if has_results:
                results_button = CTkButton(row_frame, text="✔", command=lambda c_id=chat["id"]: self.open_folder(c_id, "results"), **_icon_kw)
                results_button.grid(row=0, column=column_offset, padx=_btn_pad)
                column_offset += 1
            delete_button = CTkButton(row_frame, text="✘", command=lambda c_id=chat["id"]: self.delete_selected_chat(c_id), **_icon_kw)
            delete_button.grid(row=0, column=column_offset, padx=_btn_pad)
            return row_frame

        def append_chat_row(self, chat):
            """Add one chat at the bottom of the list without rebuilding all rows."""
            if not hasattr(self, 'chats_list_frame') or not self.chats_list_frame.winfo_exists():
                self.load_chats()
                return
            # default pack order = bottom (newest last)
            self._pack_chat_row(chat)
            try:
                self.chats_list_frame.update_idletasks()
                if hasattr(self.chats_list_frame, '_parent_canvas'):
                    canvas = self.chats_list_frame._parent_canvas
                    canvas.configure(scrollregion=canvas.bbox("all") or (0, 0, 0, 0))
                    canvas.yview_moveto(1.0)
            except Exception:
                pass
            self.update_chat_list_colors()
            try:
                if hasattr(self, '_draw_chat_list_brace'):
                    self._draw_chat_list_brace()
            except Exception:
                pass

        def _scroll_chat_list_to_bottom(self):
            """Newest chats are at the bottom — pin list viewport there after load/layout."""
            if not hasattr(self, 'chats_list_frame') or not self.chats_list_frame.winfo_exists():
                return
            if not hasattr(self.chats_list_frame, '_parent_canvas'):
                return
            try:
                self.chats_list_frame.update_idletasks()
                canvas = self.chats_list_frame._parent_canvas
                bbox = canvas.bbox("all")
                canvas.configure(scrollregion=bbox or (0, 0, 0, 0))
                canvas.yview_moveto(1.0)
            except Exception:
                pass

        def load_chats(self):
            current_selection = self.current_chat_id
            for widget in self.chats_list_frame.winfo_children(): widget.destroy()
            # always re-read order from disk (oldest→newest)
            try:
                chats = self.backend._load_chats_from_db()
                self.backend.cache.update_chats(chats)
            except Exception:
                chats = self.backend.get_chats()
            # last chat = newest (bottom); prefer keep selection, else select last
            first_chat_id = chats[-1]["id"] if chats else None
            for chat in chats:
                self._pack_chat_row(chat)
            self.chats_list_frame.update_idletasks()
            self._scroll_chat_list_to_bottom()
            # layout after first paint often resets yview — re-pin bottom
            for delay in (40, 120, 300, 700):
                try:
                    self.after(delay, self._scroll_chat_list_to_bottom)
                except Exception:
                    pass
            chat_ids = [c["id"] for c in chats]
            if current_selection not in chat_ids:
                self.on_chat_select(first_chat_id) if first_chat_id else self.clear_chat_view()
            self.update_chat_list_colors()
            try:
                if hasattr(self, '_draw_chat_list_brace'):
                    self._draw_chat_list_brace()
            except Exception:
                pass
        def open_folder(self, chat_id, folder_name):
            folder_path = self.backend.chat_folder(chat_id) / folder_name
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
            # invalidate delayed scroll jobs from previous chat (they re-broke the thumb)
            self._messages_scroll_gen = getattr(self, '_messages_scroll_gen', 0) + 1
            self.clear_messages()
            self.load_chat_messages()
            self.update_chat_controls()
            self.update_chat_list_colors()
        def _force_no_scroll_state(self):
            """Empty / short chat: scrollregion = viewport, full thumb, hide accent bar."""
            if not hasattr(self, 'messages_frame') or not self.messages_frame.winfo_exists():
                return
            canvas = getattr(self.messages_frame, '_parent_canvas', None)
            sb = getattr(self.messages_frame, '_scrollbar', None)
            if canvas is None:
                return
            try:
                self.messages_frame.update_idletasks()
            except Exception:
                pass
            try:
                cw = max(int(canvas.winfo_width() or 1), 1)
                ch = max(int(canvas.winfo_height() or 1), 1)
                # viewport-sized region → yview is always (0, 1), no tiny leftover thumb
                canvas.configure(scrollregion=(0, 0, cw, ch))
                canvas.yview_moveto(0.0)
                canvas.xview_moveto(0.0)
            except Exception:
                pass
            if sb is not None:
                try:
                    sb.set(0.0, 1.0)
                except Exception:
                    pass
                _style_scrollbar_widget(sb, idle=True)
                # hard-hide thumb track when nothing to scroll (CTk still draws min-pixel knob)
                try:
                    sb.grid_remove()
                except Exception:
                    pass
            try:
                hsb = getattr(self.messages_frame, '_scrollbar_horizontal', None)
                if hsb is not None:
                    hsb.set(0.0, 1.0)
                    _style_scrollbar_widget(hsb, idle=True)
            except Exception:
                pass

        def _sync_messages_scrollbar(self):
            """Recompute scrollregion + thumb; empty chat must not keep long-chat knob."""
            if not hasattr(self, 'messages_frame') or not self.messages_frame.winfo_exists():
                return
            try:
                canvas = getattr(self.messages_frame, '_parent_canvas', None)
                sb = getattr(self.messages_frame, '_scrollbar', None)
                if canvas is None:
                    return
                try:
                    self.messages_frame.update_idletasks()
                except Exception:
                    pass
                kids = [w for w in self.messages_frame.winfo_children() if w.winfo_exists()]
                vw = max(int(canvas.winfo_width() or 1), 1)
                vh = max(int(canvas.winfo_height() or 1), 1)
                if not kids:
                    self._force_no_scroll_state()
                    return
                bbox = canvas.bbox("all")
                content_h = (bbox[3] - bbox[1]) if bbox else 0
                content_w = (bbox[2] - bbox[0]) if bbox else 0
                need_v = content_h > vh + 2
                need_h = content_w > vw + 2
                if not need_v:
                    # short chat: same as empty — no sticky mini-thumb
                    try:
                        canvas.configure(scrollregion=(0, 0, max(vw, content_w, 1), vh))
                        canvas.yview_moveto(0.0)
                    except Exception:
                        pass
                    if sb is not None:
                        try:
                            sb.set(0.0, 1.0)
                        except Exception:
                            pass
                        _style_scrollbar_widget(sb, idle=True)
                        try:
                            sb.grid_remove()
                        except Exception:
                            pass
                else:
                    try:
                        canvas.configure(scrollregion=bbox)
                    except Exception:
                        pass
                    first, last = canvas.yview()
                    if sb is not None:
                        try:
                            # restore grid if we hid it on previous empty chat
                            try:
                                sb.grid()
                            except Exception:
                                pass
                            sb.set(first, last)
                        except Exception:
                            pass
                        _style_scrollbar_widget(sb, idle=False)
                if hasattr(self.messages_frame, '_refresh_scrollbar_fade'):
                    try:
                        self.messages_frame._refresh_scrollbar_fade()
                    except Exception:
                        pass
            except Exception:
                pass

        def clear_messages(self):
            for widget in self.messages_frame.winfo_children():
                try:
                    widget.destroy()
                except Exception:
                    pass
            # drop forced size so next messages can expand; then force no-scroll metrics
            try:
                self.messages_frame.configure(height=0)
            except Exception:
                pass
            try:
                # let packer recompute empty req height
                self.messages_frame.update_idletasks()
            except Exception:
                pass
            self._force_no_scroll_state()

        def _reset_messages_scroll_metrics(self, gen=None):
            if gen is not None and gen != getattr(self, '_messages_scroll_gen', 0):
                return
            self._sync_messages_scrollbar()

        def _scroll_messages_to_bottom(self, gen=None):
            if gen is not None and gen != getattr(self, '_messages_scroll_gen', 0):
                return
            if not hasattr(self, 'messages_frame') or not self.messages_frame.winfo_exists():
                return
            if not hasattr(self.messages_frame, '_parent_canvas'):
                return
            try:
                kids = [w for w in self.messages_frame.winfo_children() if w.winfo_exists()]
                if not kids:
                    self._force_no_scroll_state()
                    return
                self.messages_frame.update_idletasks()
                canvas = self.messages_frame._parent_canvas
                bbox = canvas.bbox("all")
                vh = max(int(canvas.winfo_height() or 0), 1)
                if not bbox:
                    self._force_no_scroll_state()
                    return
                content_h = max(bbox[3] - bbox[1], 1)
                if content_h <= vh + 2:
                    self._sync_messages_scrollbar()
                    return
                canvas.configure(scrollregion=bbox)
                canvas.yview_moveto(1.0)
                self._sync_messages_scrollbar()
            except tk.TclError:
                pass

        def _scroll_messages_to_bottom_retry(self):
            """After chat switch/layout — recompute metrics; ignore if chat switched again."""
            gen = getattr(self, '_messages_scroll_gen', 0)
            self._scroll_messages_to_bottom(gen=gen)
            for delay in (10, 40, 100, 200, 400, 700):
                try:
                    self.after(delay, lambda g=gen: self._scroll_messages_to_bottom(gen=g))
                except Exception:
                    pass

        def load_chat_messages(self):
            for widget in list(self.messages_frame.winfo_children()):
                try:
                    widget.destroy()
                except Exception:
                    pass
            if not self.current_chat_id:
                self._force_no_scroll_state()
                return
            messages = self.backend.get_messages(self.current_chat_id)
            if not messages:
                # empty chat: hard-reset now + delayed (CTk Configure may re-apply old region)
                self._force_no_scroll_state()
                gen = getattr(self, '_messages_scroll_gen', 0)

                def _keep_empty(g=gen):
                    if g != getattr(self, '_messages_scroll_gen', 0):
                        return
                    self._force_no_scroll_state()

                try:
                    self.after_idle(_keep_empty)
                    for delay in (30, 80, 160, 350, 600):
                        self.after(delay, _keep_empty)
                except Exception:
                    pass
                return
            show_dt = self.backend.get_global_settings().get("show_message_datetime", "0") == "1"
            for msg in messages:
                self.add_message_to_ui(
                    msg["text"], msg["isMy"],
                    attachments=msg.get("attachments", []),
                    timestamp=msg.get("timestamp"),
                    show_datetime=show_dt,
                    scroll=False)
            # ensure scrollbar visible again if content overflows
            try:
                sb = getattr(self.messages_frame, '_scrollbar', None)
                if sb is not None:
                    sb.grid()
            except Exception:
                pass
            self.after_idle(self._scroll_messages_to_bottom_retry)
            self._update_message_wraplengths(force=True)
        def copy_text_to_clipboard(self, text): self.clipboard_clear(); self.clipboard_append(text)
        def add_message_to_ui(self, text, is_my, is_question=False, attachments=None, timestamp=None, show_datetime=False, scroll=True):
            bubble, msg_text_widget = create_chat_message_bubble(
                self.messages_frame, text, is_my, attachments, is_question,
                timestamp=timestamp, show_datetime=show_datetime)
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
                    hover_color = PURPLE_ACCENT if is_my else DARK_BG
                    # open button first so it stays visible on narrow widths
                    CTkButton(
                        att_frame, text="↗", font=FONT_REGULAR, width=28, height=25,
                        fg_color="transparent", hover_color=hover_color,
                        command=lambda a=att: self.open_attachment(a)).pack(side=tk.LEFT, padx=(0, 4))
                    create_styled_label(att_frame, text=Path(att).name).pack(side=tk.LEFT, fill=tk.X, expand=True)
            if scroll:
                self.after_idle(self._scroll_messages_to_bottom)
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
            if not self.current_chat_id:
                self.create_chat_window_show()
                return

            attachments_paths = [str(a.resolve()) for a in self.attachments] if self.attachments else []
            # Optional: copy user attachments into chat/files (workspace)
            if attachments_paths:
                try:
                    chat_s = self.backend.get_chat_settings(self.current_chat_id) or {}
                    glob_s = self.backend.get_global_settings() or {}
                    copy_on = str(chat_s.get('copy_user_attachments_to_files',
                                             glob_s.get('copy_user_attachments_to_files', '0'))) == '1'
                except Exception:
                    copy_on = False
                if copy_on:
                    try:
                        dest_dir = self.backend.chat_folder(self.current_chat_id) / 'files'
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        copied = []
                        for p in attachments_paths:
                            src = Path(p)
                            if not src.is_file():
                                continue
                            dest = dest_dir / src.name
                            if dest.exists():
                                stem, suf = src.stem, src.suffix
                                n = 1
                                while dest.exists():
                                    dest = dest_dir / f"{stem}_{n}{suf}"
                                    n += 1
                            shutil.copy2(str(src), str(dest))
                            copied.append(str(dest.resolve()))
                        if copied:
                            attachments_paths = copied
                    except Exception as e:
                        print(f"copy_user_attachments_to_files: {e}")
            # Флаг ждём answer_user: снять ПОСЛЕ формирования message_data (раньше сбрасывали до проверки)
            awaiting_user_answer = bool(self.waiting_for_answer.get(self.current_chat_id))
            if awaiting_user_answer:
                self.waiting_for_answer[self.current_chat_id] = False

            if self.backend.add_message(self.current_chat_id, text, True, attachments_paths):
                # ====== НОВЫЙ БЛОК: Автоматическое переименование чата ======
                if self.current_chat_id:
                    chats = self.backend.get_chats()
                    chat = next((c for c in chats if c['id'] == self.current_chat_id), None)
                    if chat and chat['name'] == "—":
                        new_name = text.strip()[:9] if text.strip() else None
                        if new_name:
                            self.backend.update_chat_name(self.current_chat_id, new_name)
                            # only rename the one row — do not rebuild the whole chat list
                            try:
                                if not self.update_chat_list_name(self.current_chat_id, new_name):
                                    self.load_chats()
                            except Exception:
                                try:
                                    self.load_chats()
                                except Exception:
                                    pass
                # ========================================================

                self.attachments.clear()
                self.show_attachments()
                self.add_message_to_ui(text, True, attachments=attachments_paths)
                self.input_text.delete("1.0", "end")
                self.adjust_input_height()

                cmd = None
                if awaiting_user_answer:
                    cmd = 'answer_user'
                else:
                    # mid-dialog inject when process already running and option on
                    try:
                        deliver = str(self.backend.get_chat_settings(self.current_chat_id).get('deliver_user_messages', '0')) == '1'
                    except Exception:
                        deliver = False
                    if deliver and self.current_chat_id in self.chat_processes and self.chat_processes[self.current_chat_id].is_alive():
                        cmd = 'user_inject'
                message_data = {'text': text, 'attachments': attachments_paths or None, 'command': cmd}
                if self.current_chat_id in self.input_queues:
                    self.input_queues[self.current_chat_id].put(message_data)
                if self.current_chat_id not in self.chat_processes:
                    self.resume_chat()
                else:
                    self.update_chat_controls()
        def start_chat_process(self, chat_id):
            chat_settings = self.backend.get_chat_settings(chat_id)
            # Совместимость версии релиза: спросить, если чат новее/старее/без версии
            try:
                from cross_gpt import RELEASE_VERSION as APP_RELEASE
            except Exception:
                APP_RELEASE = "2026-07"
            chat_rel = str(chat_settings.get("release_version", "") or "").strip()
            if chat_rel != APP_RELEASE:
                if not chat_rel:
                    msg = Lang.get("release_version_missing", app=APP_RELEASE)
                elif chat_rel > APP_RELEASE:
                    msg = Lang.get("release_version_newer", chat=chat_rel, app=APP_RELEASE)
                else:
                    msg = Lang.get("release_version_older", chat=chat_rel, app=APP_RELEASE)
                if not askyesno(self, Lang.get("release_version_title"), msg):
                    return
                # пользователь продолжил — фиксируем текущую app-версию в чате
                try:
                    self.backend.set_chat_setting(chat_id, "release_version", APP_RELEASE)
                    chat_settings["release_version"] = APP_RELEASE
                except Exception as e:
                    print(f"stamp release_version: {e}")
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
                        except Exception: showerror(self, Lang.get("error", default="Error"), "Неверный пароль!")
            input_queue = multiprocessing.Queue()
            output_queue = multiprocessing.Queue()
            log_queue = multiprocessing.Queue()
            current_passwords = encryption_utils.SESSION_PASSWORDS.copy()
            from cross_gpt import initialize_work
            # Worker ALWAYS reads model_* from chatsettings.db.
            # Override only carries another_tools (same on first start and resume).
            settings_override = {
                "another_tools": self.backend.get_chat_tool_paths(chat_id),
            }
            # Log what UI thinks model is (must match DB)
            try:
                _ps = chat_settings.get("model_provider_params") or ""
                _m = ""
                for _part in _ps.split(";"):
                    if _part.strip().lower().startswith("model="):
                        _m = _part.split("=", 1)[1].strip()
                print(
                    f"[start_chat] chat={chat_id} type={chat_settings.get('model_type')} "
                    f"large_model={_m!r} use_small={chat_settings.get('use_small_model')!r} "
                    f"params_len={len(_ps)}"
                )
            except Exception:
                pass
            p = multiprocessing.Process(
                target=initialize_work,
                args=(get_base_dir(), chat_id, input_queue, output_queue, log_queue, current_passwords, settings_override))
            p.start()
            if not hasattr(self, '_chat_process_started'):
                self._chat_process_started = set()
            self._chat_process_started.add(chat_id)
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
                        cmd = response.get('command')
                        is_question = cmd in ('ask_user', 'answer_user', 'wait_user')
                    else:
                        message_text = str(response)
                        attachments = None
                        is_question = False
                        cmd = None
                    if not message_text and not is_question: continue
                    # Система ждёт ответ пользователя (ask_user / end wait)
                    if cmd in ('ask_user', 'wait_user') or is_question:
                        self.waiting_for_answer[chat_id] = True
                    # Убираем из UI сырые маркеры команд (ask_user и т.п.), command остаётся для is_question
                    import re as _re
                    display_text = _re.sub(r'!{2,4}\s*[\w\-]+\s*!{2,4}', '', message_text or '')
                    display_text = _re.sub(r'\n{3,}', '\n\n', display_text).strip() or message_text
                    if display_text:
                        self.backend.add_message(chat_id, display_text, False, attachments)
                        if chat_id == self.current_chat_id: self.add_message_to_ui(display_text, False, is_question=is_question, attachments=attachments)
                        else: self.chat_blink_states[chat_id] = True
                    if not self.focus_get(): self.flash_window()
            except queue.Empty: pass
            if chat_id in self.chat_processes and self.chat_processes[chat_id].is_alive(): self.after(500, lambda c=chat_id: self.check_chat_responses(c))
            else:
                if chat_id in self.active_chats:
                    self._cleanup_chat_process_data(chat_id)
                    if chat_id == self.current_chat_id: self.update_chat_controls()
        def preload_settings_and_create_chat(self):
            """Build Settings + CreateChat hidden (during splash) for instant open."""
            if not self.backend.is_main_config_complete():
                return
            try:
                if not (self.settings_window and self.settings_window.winfo_exists()):
                    self.settings_window = SettingsWindow(self, self.backend, preload=True)
                    self.settings_window.keep_preloaded = True
            except Exception as e:
                print(f"preload settings: {e}")
                self.settings_window = None
            try:
                if not (self.create_chat_window and self.create_chat_window.winfo_exists()):
                    self.create_chat_window = CreateChatWindow(self, self.backend, preload=True)
                    self.create_chat_window.keep_preloaded = True
            except Exception as e:
                print(f"preload create chat: {e}")
                self.create_chat_window = None
            # Force full layout of hidden dialogs so first present() only deiconifies
            for w in (self.settings_window, self.create_chat_window):
                if not w:
                    continue
                try:
                    if w.winfo_exists():
                        w.update_idletasks()
                        try:
                            w.update()
                        except Exception:
                            pass
                        # stay withdrawn (update can briefly map on some WMs)
                        try:
                            w.withdraw()
                        except Exception:
                            pass
                except Exception:
                    pass
            try:
                self.update_idletasks()
            except Exception:
                pass
        def sync_create_chat_from_global(self):
            """After saving global settings, refresh open/preloaded CreateChat fields."""
            w = self.create_chat_window
            if not w or not w.winfo_exists():
                return
            try:
                if hasattr(w, 'refresh_from_global'):
                    w.refresh_from_global()
            except Exception as e:
                print(f"sync create chat: {e}")
        def create_chat_window_show(self):
            if self.create_chat_window and self.create_chat_window.winfo_exists():
                try:
                    if hasattr(self.create_chat_window, 'refresh_from_global'):
                        self.create_chat_window.refresh_from_global()
                    self.create_chat_window.present()
                    return
                except Exception:
                    try:
                        self.create_chat_window.destroy()
                    except Exception:
                        pass
            self.create_chat_window = CreateChatWindow(self, self.backend, preload=False)
            self.create_chat_window.keep_preloaded = True
        def open_settings(self):
            if self.settings_window and self.settings_window.winfo_exists():
                try:
                    if hasattr(self.settings_window, 'refresh_from_global'):
                        self.settings_window.refresh_from_global()
                    self.settings_window.present()
                    return
                except Exception:
                    try:
                        self.settings_window.destroy()
                    except Exception:
                        pass
            self.settings_window = SettingsWindow(self, self.backend, preload=False)
            self.settings_window.keep_preloaded = True
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
            # Leave room for scrollbar + padding; scale may inflate chrome
            frame_w = self.messages_bordered_frame.winfo_width()
            pad = max(80, int(28 * (UI_SCALE_PCT / 100.0)) + SCROLLBAR_WIDTH * 3)
            available_width = max(80, frame_w - pad)
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
            # Compact size fitted to text (not huge empty dialog)
            msg = str(message or "")
            lines = msg.split("\n") if msg else [""]
            max_line = max((len(l) for l in lines), default=12)
            # ~7px per char, clamp width
            width = min(max(260, max_line * 7 + 48), 420)
            wrap = max(width - 48, 180)
            chars_per = max(wrap // 7, 16)
            wrapped = 0
            for l in lines:
                wrapped += max(1, (len(l) + chars_per - 1) // chars_per) if l else 1
            # text block + title bar + buttons + padding
            height = min(max(96 + wrapped * 17 + 44, 110), 360)
            super().__init__(parent)
            self.title(title)
            self.geometry(f"{width}x{height}")
            self.minsize(min(width, 240), 96)
            self.maxsize(480, 420)
            self.configure(fg_color=DARK_BG)
            self.result = None
            self.grid_columnconfigure(0, weight=1)
            self.grid_rowconfigure(0, weight=1)
            main_frame = create_styled_frame(self)
            main_frame.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 6))
            main_frame.grid_columnconfigure(0, weight=1)
            main_frame.grid_rowconfigure(0, weight=1)
            message_label = create_styled_label(
                main_frame, text=msg, wraplength=wrap, justify="left", font=FONT_REGULAR, text_color=WHITE)
            message_label.grid(row=0, column=0, sticky="nsew")
            btn_frame = create_styled_frame(self)
            btn_frame.grid(row=1, column=0, sticky="se", padx=12, pady=(0, 10))
            for text_key, value in buttons:
                btn_text = Lang.get(text_key.lower())
                btn = create_styled_button(btn_frame, text=btn_text, command=lambda v=value: self.set_result(v))
                btn.pack(side="left", padx=(8, 0))
                if text_key.lower() in ["ok", "yes"]: self.bind("<Return>", lambda e, v=value: self.set_result(v))
            self.bind("<Escape>", lambda e: self.on_close())
            self.transient(parent)
            self.protocol("WM_DELETE_WINDOW", self.on_close)
            self.after(10, self.setup_and_center)
            # after layout, shrink height if content smaller than estimate
            def _fit():
                try:
                    self.update_idletasks()
                    need_h = message_label.winfo_reqheight() + btn_frame.winfo_reqheight() + 36
                    need_w = max(message_label.winfo_reqwidth() + 36, 240)
                    need_h = min(max(need_h, 100), 400)
                    need_w = min(max(need_w, 240), 480)
                    self.geometry(f"{need_w}x{need_h}")
                except Exception:
                    pass
            self.after(30, _fit)
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
        def _clear_step_widgets(self):
            for widget in self.winfo_children():
                try:
                    widget.destroy()
                except Exception:
                    pass

        def show_step1_language(self):
            self._clear_step_widgets()
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
            self._clear_step_widgets()
            self.settings_vars = self._get_default_settings()
            btn_frame = create_styled_frame(self)
            btn_frame.pack(side="bottom", fill="x", pady=(0, 20), padx=20)
            self.validate_btn = create_styled_button(btn_frame, text=Lang.get("validate_model"), command=self.validate_model)
            self.validate_btn.pack(side="left")
            back_btn = create_styled_button(btn_frame, text="←", command=self.show_step1_language)
            back_btn.pack(side="left", padx=(10, 10))
            next_btn = create_styled_button(btn_frame, text="→", command=self.show_step3_chats_dir)
            next_btn.pack(side="right")
            self._init_next_to_chats_btn = next_btn
            main_frame = create_styled_frame(self)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            self._create_model_ui(main_frame)
            self._load_provider_params_from_string()
        def show_step3_chats_dir(self):
            """Separate page: chats folder (after model)."""
            if not getattr(self, 'validated', False):
                if not askyesno(self, Lang.get("warning"), Lang.get("model_not_validated_continue", default="Model not validated. Continue?")):
                    return
            self._clear_step_widgets()
            if not getattr(self, 'settings_vars', None):
                self.settings_vars = self._get_default_settings()
            if 'chats_dir' not in self.settings_vars:
                self.settings_vars['chats_dir'] = tk.StringVar(value=suggest_default_chats_dir())
            self.title(Lang.get("chats_dir_step_title", default=Lang.get("chats_dir")))
            btn_frame = create_styled_frame(self)
            btn_frame.pack(side="bottom", fill="x", pady=(0, 20), padx=20)
            create_styled_button(btn_frame, text="←", command=self.show_step2_model).pack(side="left")
            self.save_btn = create_styled_button(
                btn_frame, text=Lang.get("save_and_continue"), command=self.save_settings)
            self.save_btn.pack(side="right")
            main = create_styled_frame(self)
            main.pack(fill="both", expand=True, padx=24, pady=24)
            create_styled_label(
                main, text=Lang.get("chats_dir", default="chats_dir"), text_color=WHITE
            ).pack(anchor="w", pady=(0, 8))
            path_row = create_styled_frame(main, fg_color="transparent")
            path_row.pack(fill="x", pady=4)
            path_row.grid_columnconfigure(0, weight=1)
            create_styled_entry(path_row, textvariable=self.settings_vars['chats_dir']).grid(
                row=0, column=0, sticky="ew", padx=(0, 5))
            def browse_chats():
                chosen = filedialog.askdirectory(
                    initialdir=self.settings_vars['chats_dir'].get() or suggest_default_chats_dir())
                if chosen:
                    self.settings_vars['chats_dir'].set(chosen)
            create_styled_button(path_row, text=Lang.get("browse"), width=80, command=browse_chats).grid(row=0, column=1)
            create_styled_label(
                main, text=Lang.get("chats_dir_desc", default=""), wraplength=420, justify="left",
                text_color=WHITE,
            ).pack(anchor="w", pady=(12, 0))
        def _get_default_settings(self):
            settings = self.backend.get_global_settings()
            settings['language'] = self.lang_var.get()
            if 'max_token_limit' not in settings: settings['max_token_limit'] = '8192'
            if 'chats_dir' not in settings or not settings.get('chats_dir'):
                settings['chats_dir'] = suggest_default_chats_dir()
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
            if has_token_param and not has_pwd_param:
                showerror(self, Lang.get("error", default="Error"), "Провайдер не валиден: отсутствует параметр password при наличии api_token."); return
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
                self._remember_provider_params(model_type, connection_string)
            else:
                self.validated = False
                showerror(self, Lang.get("validation_error"), msg)
        def save_settings(self):
            if not self.validated:
                if not askyesno(self, Lang.get("warning"), Lang.get("model_not_validated_continue", default="Model not validated. Continue?")):
                    return
            self.enforce_token_limit()
            try:
                token_limit = int(self.settings_vars['token_limit'].get())
                max_limit = int(self.settings_vars['max_token_limit'].get())
                if not (1 <= token_limit <= max_limit): raise ValueError
            except (ValueError, TypeError): showerror(self, Lang.get("error"), Lang.get("token_limit_info", max_tokens=self.max_tokens)); return
            chats_dir = normalize_chats_dir(self.settings_vars.get('chats_dir', tk.StringVar(value=suggest_default_chats_dir())).get())
            try: os.makedirs(chats_dir, exist_ok=True)
            except OSError as e:
                showerror(self, Lang.get("error"), str(e)); return
            dual = self.collect_dual_model_settings()
            mt = dual['model_type']
            conn = dual['model_provider_params']
            self._remember_provider_params(mt, conn)
            settings_to_save = {
                'language': self.lang_var.get(),
                'model_type': mt,
                'token_limit': dual['token_limit'],
                'max_token_limit': dual['max_token_limit'],
                'model_provider_params': conn,
                'provider_params_by_type': self.settings_vars.get('provider_params_by_type', tk.StringVar(value='{}')).get(),
                'chats_dir': chats_dir}
            settings_to_save.update({k: dual[k] for k in dual if k.startswith('small_') or k == 'use_small_model'})
            self.backend.update_global_settings(settings_to_save)
            self.on_close()
        def on_close(self):
            super().on_close()
            if self.master.winfo_exists():
                if self.backend.is_main_config_complete():
                    self.master._needs_initial_setup = False
                    try:
                        self.master.deiconify()
                    except Exception:
                        pass
                    self.master.setup_main_ui()
                    try:
                        self.master.bring_to_front()
                    except Exception:
                        pass
                else:
                    self.master.destroy()
    class SettingsWindow(BaseSettingsWindow):
        def __init__(self, master, backend, preload=False):
            super().__init__(master, backend, "settings_title", "510x390", preload=preload)
            self.keep_preloaded = True
            self._show_chats_dir_picker = True  # только главные настройки
            self.bind("<Control-o>", self.add_custom_mod)
            if sys.platform == "darwin": self.bind("<Command-o>", self.add_custom_mod)
            tabview = CTkTabview(self, **TAB_VIEW_THEME)
            tabview.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
            main_tab = tabview.add(Lang.get("tab_main"))
            small_tab = tabview.add(Lang.get("tab_small_model", default="Small model"))
            chat_settings_tab = tabview.add(Lang.get("tab_chat_settings"))
            mods_tab = tabview.add(Lang.get("tab_modules"))
            self.setup_main_tab(main_tab)
            self.setup_small_model_tab(small_tab)
            self.setup_chat_settings_tab(chat_settings_tab)
            self.setup_mods_tab(mods_tab)
            flush_tabview_content(tabview)
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
            # scrollable model tab body — flush to tab edges
            scroll = create_scrollable_frame(parent, fg_color="transparent", corner_radius=0, border_width=0)
            scroll.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
            main_frame = create_styled_frame(scroll)
            main_frame.pack(fill="both", expand=True, padx=2, pady=2)
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
            # small top gap so Save/Validate don't stick to scroll content
            btn_frame.grid(row=1, column=0, sticky="ew", padx=6, pady=(10, 8))
            create_styled_button(btn_frame, text=Lang.get("validate_model"), command=self.validate_model).pack(side="left", padx=5)
            self.save_btn_settings = create_styled_button(btn_frame, text=Lang.get("save"), command=self.save_settings)
            self.save_btn_settings.pack(side="left", padx=5)
            create_styled_button(btn_frame, text=Lang.get("reset_settings_button"), command=self.reset_settings).pack(side="right", padx=5)
            self._load_provider_params_from_string()
        def setup_small_model_tab(self, parent):
            parent.grid_columnconfigure(0, weight=1)
            parent.grid_rowconfigure(0, weight=1)
            scroll = create_scrollable_frame(parent, fg_color="transparent", corner_radius=0, border_width=0)
            scroll.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
            if not hasattr(self, 'settings_vars') or not self.settings_vars:
                self.settings_vars = self._get_default_settings()
            self._create_small_model_ui(scroll)
            # always show details on dedicated tab
            try:
                if hasattr(self, 'small_details'):
                    self.small_details.pack(fill="x", padx=6, pady=4)
            except Exception:
                pass
        def save_settings(self):
            # 1) Валидация и диалоги — пока окно открыто
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
            except (ValueError, TypeError):
                showerror(self, Lang.get("error"), Lang.get("token_limit_info", max_tokens=self.max_tokens)); return
            dual = self.collect_dual_model_settings()
            conn = dual['model_provider_params']
            self._remember_provider_params(current_model_type, conn)
            settings_to_save = {
                'language': self.settings_vars['language'].get(),
                'model_type': dual['model_type'],
                'token_limit': dual['token_limit'],
                'max_token_limit': dual['max_token_limit'],
                'model_provider_params': conn,
                'provider_params_by_type': self.settings_vars.get('provider_params_by_type', tk.StringVar(value='{}')).get(),
            }
            settings_to_save.update({k: dual[k] for k in (
                'use_small_model', 'small_model_type', 'small_model_provider_params',
                'small_token_limit', 'small_max_token_limit',
                'small_for_cutter_only', 'small_agent_until_protocol',
                'small_protocol_drop_error') if k in dual})
            metadata = self.backend.get_settings_metadata()
            for key in metadata:
                if key in self.settings_vars: settings_to_save[key] = self.settings_vars[key].get()
            # dual model keys win over metadata pass
            settings_to_save.update({k: dual[k] for k in dual if k.startswith('small_') or k in ('use_small_model', 'model_type', 'model_provider_params', 'token_limit', 'max_token_limit')})
            # re-sync ui_scale from slider position (StringVar can lag if only drag-end missed)
            if 'ui_scale' in self.settings_vars:
                try:
                    raw_scale = self.settings_vars['ui_scale'].get()
                    # also peek slider widget if still mapped
                    w = getattr(self, 'created_widgets', None) or {}
                    sl = w.get('ui_scale') if isinstance(w, dict) else None
                    if sl is not None and hasattr(sl, 'get'):
                        raw_scale = str(ui_scale_index_to_pct(sl.get()))
                    settings_to_save['ui_scale'] = str(clamp_ui_scale_pct(raw_scale))
                except Exception:
                    settings_to_save['ui_scale'] = str(clamp_ui_scale_pct(settings_to_save.get('ui_scale', '100')))
            old_chats_root = self.backend.get_chats_root()
            new_chats_root = old_chats_root
            do_migrate = False
            if 'chats_dir' in self.settings_vars:
                new_chats_root = normalize_chats_dir(self.settings_vars['chats_dir'].get())
                settings_to_save['chats_dir'] = new_chats_root
                if os.path.normpath(old_chats_root) != os.path.normpath(new_chats_root):
                    has_old = os.path.isdir(old_chats_root) and any(
                        os.path.isdir(os.path.join(old_chats_root, n)) for n in os.listdir(old_chats_root) if not n.startswith('.'))
                    if has_old:
                        do_migrate = askyesno(
                            self, Lang.get("chats_dir_migrate_title"),
                            Lang.get("chats_dir_migrate_message", old=old_chats_root, new=new_chats_root))
            pending_mods = {}
            if hasattr(self, 'pending_default_mods'):
                pending_mods = {mid: bool(var.get()) for mid, var in self.pending_default_mods.items()}
            new_language = self.settings_vars['language'].get()
            original_language = self.original_language
            theme_on = str(settings_to_save.get('ui_light_theme', '0') or '0') == '1'
            try:
                prev_theme = str(self.backend.get_global_settings().get('ui_light_theme', '0') or '0') == '1'
            except Exception:
                prev_theme = False
            theme_changed = theme_on != prev_theme
            scale_pct = clamp_ui_scale_pct(settings_to_save.get('ui_scale', '100'))
            settings_to_save['ui_scale'] = str(scale_pct)
            try:
                prev_scale = clamp_ui_scale_pct(self.backend.get_global_settings().get('ui_scale', '100'))
            except Exception:
                prev_scale = 100
            scale_changed = scale_pct != prev_scale
            lang_changed = new_language != original_language
            chats_dir_changed = bool(do_migrate) or (
                str(settings_to_save.get('chats_dir', '') or '') != str(old_chats_root or '')
            )
            visual_rebuild = theme_changed or scale_changed
            master = self.master
            backend = self.backend

            # 1) RAM only — instant, so UI never waits on SQLite while dialog is open
            try:
                backend.update_global_settings(settings_to_save, persist=False)
            except Exception as e:
                showerror(self, Lang.get("error"), str(e))
                return
            for mod_id, enabled in pending_mods.items():
                try:
                    backend.update_default_mod_enabled(mod_id, enabled)
                except Exception as e:
                    print(f"mod enable: {e}")

            def _set_settings_btn(state):
                if not master.winfo_exists():
                    return
                if hasattr(master, 'settings_btn'):
                    try:
                        master.settings_btn.configure(state=state)
                    except tk.TclError:
                        pass

            # Hide (keep preloaded) — never destroy so reopen is instant
            try:
                self.hide_to_preload()
            except Exception:
                try:
                    self.grab_release()
                except Exception:
                    pass
                try:
                    self.withdraw()
                except Exception:
                    pass

            # 2) After hide: paint one frame, then DB + optional visual work (never block open dialog)
            def _after_settings_closed():
                _set_settings_btn("disabled")
                try:
                    # batch disk write (one transaction)
                    try:
                        backend.persist_global_settings(settings_to_save)
                    except Exception as e:
                        print(f"persist settings: {e}")
                    if do_migrate:
                        ok, msg = backend.migrate_chats_dir(old_chats_root, new_chats_root)
                        if not ok and master.winfo_exists():
                            showerror(master, Lang.get("error"), Lang.get("chats_dir_migrate_failed", e=msg))
                    try:
                        os.makedirs(new_chats_root, exist_ok=True)
                    except OSError as e:
                        if master.winfo_exists():
                            showerror(master, Lang.get("error"), str(e))
                    if lang_changed:
                        Lang.load_language(new_language)
                        try:
                            backend.rescan_and_localize_modules()
                        except Exception as e:
                            print(f"rescan modules on lang change: {e}")
                        if master.winfo_exists() and hasattr(master, 'refresh_ui_language'):
                            try:
                                master.refresh_ui_language()
                            except Exception as e:
                                print(f"refresh_ui_language: {e}")
                    if master.winfo_exists() and hasattr(master, 'sync_create_chat_from_global'):
                        try:
                            master.sync_create_chat_from_global()
                        except Exception as e:
                            print(f"sync create after settings: {e}")
                    if visual_rebuild and master.winfo_exists():
                        try:
                            if theme_changed:
                                apply_ui_theme(theme_on)
                            apply_ui_scale(scale_pct)
                            master.configure(fg_color=DARK_BG)
                            for attr in ('settings_window', 'create_chat_window'):
                                w = getattr(master, attr, None)
                                if w is not None:
                                    try:
                                        if w.winfo_exists():
                                            w.destroy()
                                    except Exception:
                                        pass
                                    setattr(master, attr, None)
                            if hasattr(master, 'setup_main_ui') and backend.is_main_config_complete():
                                master.setup_main_ui()
                            # rebuild preloaded dialogs after scale/theme so next open is instant
                            if hasattr(master, 'preload_settings_and_create_chat'):
                                try:
                                    master.preload_settings_and_create_chat()
                                except Exception as e:
                                    print(f"re-preload after visual rebuild: {e}")
                        except Exception as e:
                            print(f"theme/scale apply: {e}")
                    else:
                        # Keep preloaded settings vars in sync with what we just saved
                        sw = getattr(master, 'settings_window', None)
                        if sw is not None and sw.winfo_exists() and hasattr(sw, 'refresh_from_global'):
                            try:
                                sw.refresh_from_global()
                            except Exception as e:
                                print(f"settings refresh after save: {e}")
                        if chats_dir_changed and master.winfo_exists() and hasattr(master, 'load_chats'):
                            try:
                                backend.cache.update_chats(backend._load_chats_from_db())
                                master.load_chats()
                            except Exception as e:
                                print(f"reload chats after dir change: {e}")
                finally:
                    _set_settings_btn("normal")

            if master.winfo_exists():
                # after(1): let hide paint first so user never stares at frozen open dialog
                master.after(1, _after_settings_closed)
            else:
                try:
                    backend.persist_global_settings(settings_to_save)
                except Exception:
                    pass
        def reset_settings(self):
            if askyesno(self, Lang.get("reset_settings_confirm_title"), Lang.get("reset_settings_confirm_message")):
                db_path = Path(self.backend.db_path)
                if db_path.exists(): db_path.unlink()
                showinfo(self, Lang.get("info"), Lang.get("restart_required"))
                self.master.destroy()
        def setup_mods_tab(self, parent):
            parent.grid_rowconfigure(0, weight=1)
            parent.grid_columnconfigure(0, weight=1)
            self.scrollable_frame = create_scrollable_frame(
                parent, fg_color="transparent", label_text="", corner_radius=0, border_width=0)
            self.scrollable_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
            self.rebuild_mods_list()
            create_styled_button(parent, text=Lang.get("add_module"), command=self.add_custom_mod).grid(row=1, column=0, pady=5, padx=5)
        def rebuild_mods_list(self):
            for widget in self.scrollable_frame.winfo_children(): widget.destroy()
            self.scrollable_frame.grid_columnconfigure(0, weight=1)
            mm = ModuleManager()
            if not mm.loaded:
                mm.load_modules(self.backend)
            default_mods = mm.get_default_modules()
            custom_mods = mm.get_custom_modules()
            if default_mods:
                create_styled_label(self.scrollable_frame, text=Lang.get("system_modules"), font=FONT_REGULAR).pack(anchor="w", padx=5, pady=(5,2))
                for mod in default_mods: self.create_mod_ui(self.scrollable_frame, mod, is_default=True)
            create_styled_label(self.scrollable_frame, text=Lang.get("global_custom_modules"), font=FONT_REGULAR).pack(anchor="w", padx=5, pady=(10,2))
            for mod in custom_mods: self.create_mod_ui(self.scrollable_frame, mod, is_default=False)
        def create_mod_ui(self, parent, mod_data, is_default):
            # Только локальное состояние; запись в БД — по кнопке Save
            if not hasattr(self, 'pending_default_mods'):
                self.pending_default_mods = {}
            if is_default:
                if mod_data["id"] not in self.pending_default_mods:
                    self.pending_default_mods[mod_data["id"]] = tk.BooleanVar(value=mod_data["enabled"])
                enabled_var = self.pending_default_mods[mod_data["id"]]
            else:
                enabled_var = None
            def remove_callback():
                if askyesno(self, Lang.get("warning"), Lang.get("remove_module_confirm")):
                    self.backend.remove_custom_mod(mod_data["id"])
                    ModuleManager().update_custom_modules(self.backend)
                    self.rebuild_mods_list()
            create_module_ui_item(
                parent, mod_data, "default" if is_default else "custom",
                enabled_var=enabled_var if is_default else None,
                on_toggle=None,
                on_remove=None if is_default else remove_callback,
                show_checkbox=is_default)
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
        modal = False  # can stay open alongside main window
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
            self.messages_frame = create_scrollable_frame(
                self, fg_color="transparent", border_width=0, corner_radius=0)
            self.messages_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
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
        def __init__(self, master, backend, preload=False):
            super().__init__(master, backend, "create_chat_title", "510x415", preload=preload)
            self.keep_preloaded = True
            self.bind("<Control-o>", self.add_new_local_mod)
            if sys.platform == "darwin": self.bind("<Command-o>", self.add_new_local_mod)

            module_manager = ModuleManager()
            self.custom_mods_for_chat = module_manager.get_custom_modules().copy()
            self.newly_added_mods = []
            self.max_tokens = int(self.backend.get_global_settings().get("token_limit", 8192))
            self.validated = True
            self.settings_vars = self._get_default_settings()
            # Устанавливаем имя по умолчанию — длинное тире
            self.settings_vars['chat_name'].set("—")

            self.original_model_type = self.settings_vars['model_type'].get()
            self.original_connection_string = self.settings_vars['model_provider_params'].get()

            self.grid_columnconfigure(0, weight=1)
            self.grid_rowconfigure(0, weight=1)
            self.grid_rowconfigure(1, weight=0)

            tabview = CTkTabview(self, **TAB_VIEW_THEME)
            tabview.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

            model_tab = tabview.add(Lang.get("tab_model"))
            small_tab = tabview.add(Lang.get("tab_small_model", default="Small model"))
            chat_tab = tabview.add(Lang.get("tab_chat_settings"))
            mods_tab = tabview.add(Lang.get("tab_modules"))

            self.setup_model_tab(model_tab)
            self.setup_small_model_tab(small_tab)
            self.setup_chat_settings_tab(chat_tab)
            self.setup_mods_tab(mods_tab)
            flush_tabview_content(tabview)

            bottom_frame = create_styled_frame(self)
            bottom_frame.grid(row=1, column=0, sticky="ew", padx=6, pady=(10, 8))

            self.create_btn = create_styled_button(bottom_frame, text=Lang.get("create"), command=self.create_chat_finalize)
            self.create_btn.pack(side='left')
            create_styled_button(bottom_frame, text=Lang.get("validate_model"), command=self.validate_model).pack(side='left', padx=5)
            create_styled_button(bottom_frame, text=Lang.get("cancel"), command=self.hide_to_preload).pack(side='left', padx=5)

            self._load_provider_params_from_string()

        def refresh_from_global(self):
            """Pull global defaults + modules from RAM (AppCache settings + ModuleManager)."""
            BaseSettingsWindow.refresh_from_global(self)
            try:
                self.max_tokens = int(self.backend.get_global_settings().get("token_limit", 8192))
            except Exception:
                pass
            try:
                mm = ModuleManager()
                if not mm.loaded:
                    mm.load_modules(self.backend)  # only if never loaded
                default_mods = mm.get_default_modules()
                self.settings_vars['default_mods'] = {
                    mod['id']: tk.BooleanVar(value=mod['enabled']) for mod in default_mods
                }
                self.custom_mods_for_chat = list(mm.get_custom_modules())
                self.newly_added_mods = []
                if hasattr(self, 'mods_scrollable_frame') and self.mods_scrollable_frame.winfo_exists():
                    self.rebuild_mods_list()
            except Exception as e:
                print(f"create chat refresh modules: {e}")
            try:
                self.original_model_type = self.settings_vars['model_type'].get()
                self.original_connection_string = self.settings_vars['model_provider_params'].get()
            except Exception:
                pass

        def _get_default_settings(self):
            settings = self.backend.get_global_settings()
            s_vars = {key: tk.StringVar(value=val) for key, val in settings.items()}
            s_vars['chat_name'] = tk.StringVar(value="—")
            mm = ModuleManager()
            if not mm.loaded:
                mm.load_modules(self.backend)
            default_mods = mm.get_default_modules()
            s_vars['default_mods'] = {mod['id']: tk.BooleanVar(value=mod['enabled']) for mod in default_mods}
            return s_vars

        def setup_mods_tab(self, parent):
            parent.grid_rowconfigure(0, weight=1)
            parent.grid_columnconfigure(0, weight=1)
            self.mods_scrollable_frame = create_scrollable_frame(
                parent, fg_color="transparent", corner_radius=0, border_width=0)
            self.mods_scrollable_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
            self.rebuild_mods_list()

        def rebuild_mods_list(self):
            for widget in self.mods_scrollable_frame.winfo_children(): widget.destroy()
            self.mods_scrollable_frame.grid_columnconfigure(0, weight=1)
            mm = ModuleManager()
            if not mm.loaded:
                mm.load_modules(self.backend)
            default_mods = mm.get_default_modules()
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
            dual = self.collect_dual_model_settings()
            conn = dual['model_provider_params']
            self._remember_provider_params(current_model_type, conn)
            # Dual-model fields only here — never let stale settings_vars overwrite them in merge
            model_config = {
                'model_type': dual['model_type'],
                'model_provider_params': conn,
                'token_limit': dual['token_limit'],
                'max_token_limit': dual.get('max_token_limit', ''),
                'provider_params_by_type': self.settings_vars.get('provider_params_by_type', tk.StringVar(value='{}')).get(),
                'use_small_model': dual.get('use_small_model', '0'),
                'small_model_type': dual.get('small_model_type', ''),
                'small_model_provider_params': dual.get('small_model_provider_params', ''),
                'small_token_limit': dual.get('small_token_limit', ''),
                'small_max_token_limit': dual.get('small_max_token_limit', ''),
                'small_for_cutter_only': dual.get('small_for_cutter_only', '1'),
                'small_agent_until_protocol': dual.get('small_agent_until_protocol', '0'),
                'small_protocol_drop_error': dual.get('small_protocol_drop_error', '0'),
                'text_cutter_token_limit': dual.get('text_cutter_token_limit', self.settings_vars.get('text_cutter_token_limit', tk.StringVar(value='2000')).get()),
                'max_incoming_tokens': dual.get('max_incoming_tokens', self.settings_vars.get('max_incoming_tokens', tk.StringVar(value='10000')).get()),
            }
            metadata = self.backend.get_settings_metadata()
            # Keys that must come from model_config only (not metadata dump of settings_vars)
            _model_owned = set(model_config.keys())
            chat_config = {"language": Lang.current_language}
            for key in metadata:
                if key in _model_owned:
                    continue
                if key in self.settings_vars:
                    chat_config[key] = self.settings_vars[key].get()
            # mcp_url always from chat-settings tab if present
            if 'mcp_url' in self.settings_vars:
                chat_config['mcp_url'] = self.settings_vars['mcp_url'].get()
            default_mods_config = {mid: var.get() for mid, var in self.settings_vars['default_mods'].items()}
            final_custom_mods = list(self.custom_mods_for_chat) + list(self.newly_added_mods)
            settings_bundle = {"model_config": model_config, "chat_config": chat_config, "default_mods_config": default_mods_config, "custom_mods_list": final_custom_mods}
            valid_password = getattr(self, 'valid_password', None)
            master = self.master
            backend = self.backend
            # Do not freeze main UI — only hide create-chat dialog
            try:
                self.grab_release()
            except Exception:
                pass
            try:
                self.hide_to_preload()
            except Exception:
                try:
                    self.destroy()
                except Exception:
                    pass
            try:
                chat_data = backend.create_chat(chat_name, settings_bundle)
                if not chat_data:
                    if master.winfo_exists():
                        showerror(master, Lang.get("error"), Lang.get("chat_name_exists"))
                        if getattr(master, 'create_chat_window', None) and master.create_chat_window.winfo_exists():
                            try:
                                master.create_chat_window.present()
                            except Exception:
                                pass
                    return
                if valid_password:
                    encryption_utils.SESSION_PASSWORDS[chat_data["id"]] = valid_password
                elif model_config['model_type'] in encryption_utils.SESSION_PASSWORDS:
                    encryption_utils.SESSION_PASSWORDS[chat_data["id"]] = encryption_utils.SESSION_PASSWORDS[model_config['model_type']]
                if master.winfo_exists():
                    # append row at bottom — no full list rebuild
                    if hasattr(master, 'append_chat_row'):
                        master.append_chat_row(chat_data)
                    else:
                        master.load_chats()
                    master.on_chat_select(chat_data["id"])
                try:
                    w = getattr(master, 'create_chat_window', None)
                    if w and w.winfo_exists() and 'chat_name' in getattr(w, 'settings_vars', {}):
                        w.settings_vars['chat_name'].set("—")
                        w.newly_added_mods = []
                except Exception:
                    pass
            except Exception as e:
                if master.winfo_exists():
                    showerror(master, Lang.get("error"), str(e))
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
        try:
            _gs = backend.get_global_settings()
            _light = str(_gs.get("ui_light_theme", "0") or "0") == "1"
            apply_ui_theme(_light)
            apply_ui_scale(_gs.get("ui_scale", "100"))
        except Exception:
            apply_ui_theme(False)
            apply_ui_scale(100)
        app = ChatApp(backend)
        setup_icon(app)
        config_ok = backend.is_main_config_complete() and not getattr(app, '_needs_initial_setup', False)
        # During splash: build Settings + CreateChat hidden so open is instant
        # Skip when first-run wizard is required (main must stay withdrawn).
        try:
            if config_ok and hasattr(app, 'preload_settings_and_create_chat'):
                app.preload_settings_and_create_chat()
        except Exception as e:
            print(f"preload dialogs: {e}")
        if config_ok:
            app.after(0, app.bring_to_front)
            if sys.platform.startswith("linux"):
                app.deiconify()
        else:
            # wizard owns the UI; never deiconify empty main (was: splash blink → blank window)
            try:
                app.withdraw()
            except Exception:
                pass
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