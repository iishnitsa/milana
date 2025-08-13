import os
import re
import tkinter as tk
from tkinter import filedialog

def extract_module_doc(content):
    """Извлекает документацию модуля (первые 2-4 строки в тройных кавычках)"""
    doc_match = re.search(r'^[\'"]{3}(.*?)[\'"]{3}', content, re.DOTALL)
    if not doc_match:
        return None
    
    doc_lines = [line.strip() for line in doc_match.group(1).split('\n') if line.strip()]
    return doc_lines[:4]

def extract_main_attributes(content):
    """Извлекает атрибуты функции main с их значениями, включая многострочные"""
    attr_pattern = re.compile(
        r'^main\.(\w+)\s*=\s*([\'"]{3}.*?[\'"]{3}|[\'"].*?[\'"])',
        re.DOTALL | re.MULTILINE
    )
    
    attributes = {}
    for match in attr_pattern.finditer(content):
        attr_name = match.group(1)
        attr_value = match.group(2)
        
        # Проверяем, не закомментирована ли строка
        line_start = content.rfind('\n', 0, match.start()) + 1
        line = content[line_start:match.start()].strip()
        if line.startswith('#'):
            continue
            
        # Обработка многострочных строк
        if attr_value.startswith("'''") or attr_value.startswith('"""'):
            # Удаляем кавычки и сохраняем как есть
            attr_value = attr_value[3:-3]
        else:
            # Удаляем одинарные кавычки
            attr_value = attr_value[1:-1]
        
        attributes[attr_name] = attr_value
    
    return attributes

def create_localization_template(py_file_path):
    """Создает шаблон файла локализации"""
    with open(py_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    localization_data = {
        'module_doc': extract_module_doc(content) or [],
        'attributes': extract_main_attributes(content)
    }
    
    if not localization_data['module_doc'] and not localization_data['attributes']:
        return False
    
    file_dir = os.path.dirname(py_file_path)
    file_name = os.path.basename(py_file_path)
    lang_file = os.path.join(file_dir, file_name.replace('.py', '_lang.py'))
    
    template = [
        "locales = {",
        "    'ru': {",
        "        'module_doc': ["
    ]
    
    # Добавляем документацию модуля
    for line in localization_data['module_doc']:
        if not line:  # Для пустых строк
            template.append("            '',")
        else:
            template.append(f"            {repr(line)},")
    
    template.append("        ],")
    
    # Добавляем атрибуты
    if localization_data['attributes']:
        template.append("")
        for attr, value in localization_data['attributes'].items():
            if '\n' in value:  # Многострочные строки
                template.append(f"        'main.{attr}': '''{value}''',")
            else:
                template.append(f"        'main.{attr}': {repr(value)},")
    
    template.append("    }")
    template.append("}")
    
    with open(lang_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(template))
    
    return True

def select_and_process_files():
    """Открывает диалог выбора файлов и обрабатывает их"""
    root = tk.Tk()
    root.withdraw()
    
    files = filedialog.askopenfilenames(
        title="Выберите Python файлы для локализации",
        filetypes=[("Python files", "*.py")]
    )
    
    if not files:
        print("Файлы не выбраны")
        return
    
    for file_path in files:
        try:
            success = create_localization_template(file_path)
            if success:
                print(f"Файл локализации создан для: {file_path}")
            else:
                print(f"Не найдены данные для локализации в: {file_path}")
        except Exception as e:
            print(f"Ошибка при обработке {file_path}: {str(e)}")
    
    print("\nГотово! Теперь вы можете:")
    print("1. Открыть созданные *_lang.py файлы")
    print("2. Перевести значения на русский язык")
    print("3. Сохранить файлы")

if __name__ == "__main__":
    select_and_process_files()