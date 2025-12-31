import os
import sys
import importlib.util
import multiprocessing
import subprocess

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
    from bs4 import BeautifulSoup

def run_as_interpreter():
    """Режим исполнения ui.py внутри среды лаунчера"""
    if len(sys.argv) < 2: return
    
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
        import traceback
        print("\n" + "="*40)
        print("ОШИБКА ВНУТРИ UI.PY:")
        traceback.print_exc()
        print("="*40)
        input("Нажмите Enter для выхода...")

def launch_external_ui():
    """Запуск процесса-интерпретатора"""
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(os.path.abspath(sys.executable))
        executable = sys.executable
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        executable = sys.executable

    ui_path = os.path.join(base_dir, "ui.py")

    if not os.path.exists(ui_path):
        print(f"Файл не найден: {ui_path}")
        input("Нажмите Enter...")
        return

    # Запускаем сам EXE, передавая ему путь к ui.py
    subprocess.Popen([executable, ui_path], cwd=base_dir)

if __name__ == "__main__":
    # Фикс для работы multiprocessing в упакованном EXE
    multiprocessing.freeze_support()

    # Проверка: если это дочерний процесс multiprocessing
    if len(sys.argv) > 1 and sys.argv[1] == '--multiprocessing-fork':
        pass 
    
    # Если передан .py файл — запускаем его
    elif len(sys.argv) > 1 and sys.argv[1].endswith(".py"):
        run_as_interpreter()
        
    else:
        # Обычный запуск пользователем
        launch_external_ui()