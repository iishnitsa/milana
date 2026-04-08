import base64
import os
import tkinter as tk
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import customtkinter as ctk

# Глобальный кэш паролей для сессии (хранится в ОЗУ)
# Формат: { 'model_type': 'password', 'chat_id': 'password' }
SESSION_PASSWORDS = {}

def _get_fernet(password: str, salt: bytes) -> Fernet:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=390000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return Fernet(key)

def encrypt_token(token: str, password: str) -> str:
    """Шифрует токен с использованием пароля"""
    if not password:
        return token
    salt = os.urandom(16)
    f = _get_fernet(password, salt)
    encrypted = f.encrypt(token.encode())
    return base64.b64encode(salt + encrypted).decode('utf-8')

def decrypt_token(encrypted_token: str, password: str) -> str:
    """Дешифрует токен с использованием пароля"""
    if not password:
        return encrypted_token
    try:
        data = base64.b64decode(encrypted_token.encode('utf-8'))
        salt, encrypted = data[:16], data[16:]
        f = _get_fernet(password, salt)
        return f.decrypt(encrypted).decode('utf-8')
    except Exception:
        raise ValueError("Неверный пароль или поврежденный токен")

class PasswordDialog(ctk.CTkToplevel):
    def __init__(self, master=None, title="Авторизация токена"):
        super().__init__(master)
        self.title(title)
        self.geometry("350x180")
        self.configure(fg_color="#000000") # Черный фон
        self.result = None
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        if master:
            self.transient(master)
            
        self.grid_columnconfigure(0, weight=1)
        
        # Стилизованный заголовок
        lbl = ctk.CTkLabel(self, text="Пожалуйста, введите пароль для\nрасшифровки API токена:", text_color="#ffffff", font=("Georgia", 14))
        lbl.grid(row=0, column=0, pady=(20, 10), padx=20)
        
        # Поле ввода 5200ff со значком замка
        input_frame = ctk.CTkFrame(self, fg_color="transparent")
        input_frame.grid(row=1, column=0, pady=5, padx=20, sticky="ew")
        input_frame.grid_columnconfigure(1, weight=1)
        
        lock_lbl = ctk.CTkLabel(input_frame, text="🔒", text_color="#ffffff", font=("Georgia", 18))
        lock_lbl.grid(row=0, column=0, padx=(0, 5))
        
        self.entry = ctk.CTkEntry(input_frame, show="•", fg_color="#5200ff", border_width=0, text_color="#ffffff", height=30, font=("Georgia", 14))
        self.entry.grid(row=0, column=1, sticky="ew")
        self.entry.bind("<Return>", lambda e: self.submit())
        
        btn = ctk.CTkButton(self, text="Подтвердить", fg_color="transparent", hover_color="#5200ff", command=self.submit, corner_radius=50)
        btn.grid(row=2, column=0, pady=(15, 20))
        
        self.entry.focus_set()
        
    def submit(self):
        self.result = self.entry.get()
        if hasattr(self, 'grab_status') and self.grab_status() == "grab":
            self.grab_release()
        self.destroy()
        
    def on_close(self):
        self.result = None
        if hasattr(self, 'grab_status') and self.grab_status() == "grab":
            self.grab_release()
        self.destroy()

def ask_for_password(master=None):
    """Вызывает стилизованное окно запроса пароля"""
    dialog = PasswordDialog(master)
    if master:
        # Для модальности, если вызвано из UI
        dialog.grab_set()
        master.wait_window(dialog)
    else:
        dialog.wait_window(dialog)
    return dialog.result