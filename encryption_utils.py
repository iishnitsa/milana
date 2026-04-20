import base64
import os
import sys
import platform
import tkinter as tk
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import customtkinter as ctk

# Глобальный кэш паролей для сессии (хранится в ОЗУ)
# Формат: { 'model_type': 'password', 'chat_id': 'password' }
SESSION_PASSWORDS = {}

def _get_fernet(password: str, salt: bytes) -> Fernet:
    kdf = PBKDF2HMAC( algorithm=hashes.SHA256(), length=32, salt=salt, iterations=390000)
    return Fernet(base64.urlsafe_b64encode(kdf.derive(password.encode())))

def encrypt_token(token: str, password: str) -> str:
    if not password: return token
    salt = os.urandom(16)
    f = _get_fernet(password, salt)
    encrypted = f.encrypt(token.encode())
    return base64.b64encode(salt + encrypted).decode('utf-8')

def decrypt_token(encrypted_token: str, password: str) -> str: # Дешифрует токен с использованием пароля
    if not password: return encrypted_token
    try:
        data = base64.b64decode(encrypted_token.encode('utf-8'))
        salt, encrypted = data[:16], data[16:]
        f = _get_fernet(password, salt)
        return f.decrypt(encrypted).decode('utf-8')
    except Exception: raise ValueError("Wrong password or broken token")

def set_windows_dark_titlebar(window):
    """Устанавливает тёмный заголовок окна на Windows (скопировано из ui.py)"""
    if sys.platform != "win32": return
    try:
        import ctypes
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        value = ctypes.c_int(2)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(value), ctypes.sizeof(value))
        DWMWA_CAPTION_COLOR = 35
        color = ctypes.c_int(0x00000000)  # чёрный цвет
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_CAPTION_COLOR, ctypes.byref(color), ctypes.sizeof(color))
        ctypes.windll.user32.SetWindowPos(hwnd, None, 0, 0, 0, 0, 0x0027)
    except: pass

class PasswordDialog(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("")
        self.geometry("300x65")
        self.minsize(300, 65)
        self.maxsize(300, 65)
        self.configure(fg_color="#000000") # Черный фон
        self.result = None
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        # Устанавливаем тёмный заголовок окна (только Windows)
        self.after(10, lambda: set_windows_dark_titlebar(self))
        if master: self.transient(master)
        self.grid_columnconfigure(0, weight=1)
        input_frame = ctk.CTkFrame(self, fg_color="transparent")
        input_frame.grid(row=1, column=0, pady=5, padx=5, sticky="ew")
        input_frame.grid_columnconfigure(1, weight=1)
        lock_lbl = ctk.CTkLabel(input_frame, text="🔒", text_color="#ffffff", font=("Georgia", 18))
        lock_lbl.grid(row=0, column=0, padx=(0, 5))
        self.entry = ctk.CTkEntry(input_frame, show="•", fg_color="#5200ff", border_width=0, text_color="#ffffff", height=27, corner_radius=50, font=("Georgia", 12))
        self.entry.grid(row=0, column=1, sticky="ew")
        self.entry.bind("<Return>", lambda e: self.submit())
        btn = ctk.CTkButton(self, text="✔", fg_color="transparent", hover_color="#5200ff", command=self.submit, width=20, height=20, corner_radius=50)
        btn.grid(row=2, column=0)
        self.entry.focus_set()
    def submit(self):
        self.result = self.entry.get()
        self.grab_release()
        self.destroy()
    def on_close(self):
        self.result = None
        self.grab_release()
        self.destroy()

def ask_for_password(master=None):
    """Вызывает стилизованное окно запроса пароля"""
    own_root = False
    if master is None:
        try:
            master = ctk.CTk.get_default_root()
            if master is None: raise RuntimeError
        except:
            master = ctk.CTk()
            master.withdraw()
            own_root = True
    else: pass
    dialog = PasswordDialog(master)
    dialog.grab_set()
    master.wait_window(dialog)
    result = dialog.result
    if own_root: master.destroy()
    return result