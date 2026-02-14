import os
import sys
import importlib.util
import multiprocessing
import subprocess
import tkinter as tk
from tkinter import messagebox
import ctypes
import traceback

# Пытаемся сразу скрыть консоль (для Windows)
def hide_console():
    """Скрывает консольное окно текущего процесса"""
    try:
        if sys.platform == "win32":
            # Получаем дескриптор консоли
            console_window = ctypes.windll.kernel32.GetConsoleWindow()
            if console_window:
                # Скрываем окно
                ctypes.windll.user32.ShowWindow(console_window, 0)  # SW_HIDE
    except:
        pass

# Вызываем сразу при импорте
hide_console()

def show_error(title, message):
    """Показывает сообщение об ошибке в диалоговом окне"""
    try:
        if sys.platform == "win32":
            ctypes.windll.user32.MessageBoxW(0, message, title, 0x10)  # MB_ICONERROR
        else:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(title, message)
            root.destroy()
    except:
        print(f"ERROR: {title} - {message}", file=sys.stderr)

def show_info(title, message):
    """Показывает информационное сообщение"""
    try:
        if sys.platform == "win32":
            ctypes.windll.user32.MessageBoxW(0, message, title, 0x40)  # MB_ICONINFORMATION
        else:
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo(title, message)
            root.destroy()
    except:
        print(f"INFO: {title} - {message}", file=sys.stderr)

# Эту часть PyInstaller УВИДИТ и положит библиотеки в сборку
if False:
    import os
    import sys
    import zipfile
    import re
    import traceback
    import time
    import difflib
    import pickle
    import gzip
    import io
    import base64
    import json
    import tkinter
    from tkinter import simpledialog, Toplevel, Menu, filedialog
    import gc
    from io import BytesIO
    from multiprocessing import queues
    import chardet
    import numpy as np
    import pandas as pd
    import cv2
    from PIL import Image, ImageTk
    import pymupdf
    from docx import Document
    import chromadb
    import easyocr
    from chromadb.config import Settings
    from sklearn.decomposition import PCA
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import sqlite3
    from sqlite3 import connect, OperationalError
    import importlib.util
    import importlib
    from pathlib import Path
    import inspect
    import ast
    import multiprocessing
    import customtkinter
    import platform
    import random
    import string
    import shutil
    from contextlib import redirect_stdout
    import uuid
    import requests
    from urllib.parse import quote_plus, urlparse, urljoin
    from ddgs import DDGS
    from bs4 import BeautifulSoup

def run_as_interpreter():
    """Режим исполнения ui.py внутри среды лаунчера"""
    if len(sys.argv) < 2:
        return
    
    script_path = sys.argv[1]
    base_dir = os.path.dirname(os.path.abspath(script_path))
    
    # Добавляем пути, чтобы ui.py видел библиотеки
    sys.path.insert(0, base_dir)

    try:
        # Подготавливаем загрузку ui.py как главного модуля
        spec = importlib.util.spec_from_file_location("__main__", script_path)
        ui_module = importlib.util.module_from_spec(spec)
        
        # КРИТИЧЕСКИЙ ФИКС: подменяем __main__ до выполнения кода
        sys.modules["__main__"] = ui_module
        
        spec.loader.exec_module(ui_module)
    except Exception as e:
        error_msg = traceback.format_exc()
        show_error("Ошибка в ui.py", f"{str(e)}\n\n{error_msg}")
        sys.exit(1)

def get_pythonw_path():
    """Находит путь к pythonw.exe (Windows) или возвращает обычный python"""
    if sys.platform != "win32":
        return sys.executable
    
    # Пытаемся найти pythonw.exe в разных местах
    possible_paths = [
        # Рядом с текущим интерпретатором
        os.path.join(os.path.dirname(sys.executable), "pythonw.exe"),
        # В стандартных путях
        r"C:\Python39\pythonw.exe",
        r"C:\Python310\pythonw.exe",
        r"C:\Python311\pythonw.exe",
        r"C:\Python312\pythonw.exe",
        r"C:\Program Files\Python39\pythonw.exe",
        r"C:\Program Files\Python310\pythonw.exe",
        r"C:\Program Files\Python311\pythonw.exe",
        r"C:\Program Files\Python312\pythonw.exe",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Если не нашли pythonw.exe, используем обычный python.exe
    return sys.executable

def launch_external_ui():
    """Запуск ui.py без консольного окна"""
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(os.path.abspath(sys.executable))
        # Для замороженного приложения используем сам exe
        executable = sys.executable
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Для скрипта ищем pythonw.exe
        executable = get_pythonw_path()

    ui_path = os.path.join(base_dir, "ui.py")

    if not os.path.exists(ui_path):
        show_error("Ошибка", f"Файл ui.py не найден:\n{ui_path}\n\n"
                   "Убедитесь, что файл ui.py находится в той же папке, что и программа.")
        return

    try:
        # Для Windows используем специальные флаги
        if sys.platform == "win32":
            # Скрываем окно процесса
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0  # SW_HIDE
            
            # Флаги создания процесса:
            # CREATE_NO_WINDOW = 0x08000000 - не создавать окно консоли
            # DETACHED_PROCESS = 0x00000008 - отсоединить от консоли родителя
            # CREATE_NEW_CONSOLE = 0x00000010 - НЕ ИСПОЛЬЗУЕМ (создаст новую консоль)
            creationflags = 0x08000000 | 0x00000008  # CREATE_NO_WINDOW | DETACHED_PROCESS
            
            # Перенаправляем все потоки в никуда и закрываем дескрипторы
            process = subprocess.Popen(
                [executable, ui_path],
                cwd=base_dir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                startupinfo=startupinfo,
                creationflags=creationflags,
                close_fds=True,  # Закрываем все унаследованные дескрипторы
                shell=False       # Не использовать shell (он может создать консоль)
            )
        else:
            # Для Linux/macOS
            process = subprocess.Popen(
                [executable, ui_path],
                cwd=base_dir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,  # Отделяем от текущей сессии
                close_fds=True
            )
        
        # Небольшая задержка для проверки, что процесс запустился
        try:
            process.wait(timeout=0.1)
            # Если процесс завершился сразу - возможно, была ошибка
            if process.returncode is not None and process.returncode != 0:
                show_error("Ошибка запуска", 
                          f"Процесс завершился с кодом {process.returncode}.\n"
                          "Проверьте наличие всех необходимых файлов.")
        except subprocess.TimeoutExpired:
            # Процесс запустился и работает - это нормально
            pass
            
    except Exception as e:
        show_error("Ошибка запуска", f"Не удалось запустить ui.py:\n{str(e)}")

def check_environment():
    """Проверка окружения перед запуском"""
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(os.path.abspath(sys.executable))
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Проверяем наличие критически важных папок
    required_dirs = ['data', 'lang', 'default_tools', 'model_providers']
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        show_error("Отсутствуют папки", 
                  f"Не найдены следующие папки:\n{', '.join(missing_dirs)}\n\n"
                  "Убедитесь, что программа установлена правильно.")
        return False
    
    return True

def main():
    """Основная функция"""
    # Ещё раз пытаемся скрыть консоль (на всякий случай)
    hide_console()
    
    # Фикс для работы multiprocessing в упакованном EXE
    multiprocessing.freeze_support()

    # Проверка: если это дочерний процесс multiprocessing
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == '--multiprocessing-fork' or arg.startswith('--multiprocessing'):
            return
    
    # Если передан .py файл — запускаем его
    if len(sys.argv) > 1 and sys.argv[1].endswith(".py"):
        run_as_interpreter()
    else:
        # Проверяем окружение перед запуском
        if not check_environment():
            # Если проверка не пройдена, показываем сообщение и ждем
            if sys.platform == "win32":
                ctypes.windll.user32.MessageBoxW(0, 
                    "Программа будет закрыта.", 
                    "Ошибка проверки окружения", 0x10)
            sys.exit(1)
        
        # Обычный запуск пользователем
        launch_external_ui()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Ловим все необработанные исключения
        error_msg = traceback.format_exc()
        show_error("Критическая ошибка", f"Необработанное исключение:\n{str(e)}\n\n{error_msg}")
        sys.exit(1)