import os
import sys
import zipfile
from docx import Document
import pymupdf
from io import BytesIO
from PIL import Image
import chardet
from cross_gpt import let_log
import gc
import numpy as np
import pandas as pd
import cv2
from cross_gpt import (
    err_image_process_text_infoloaders,
    text_on_image_prompt_infoloaders,
    err_image_process_pdf_infoloaders,
    page_pdf_prompt_infoloaders,
    attachment_prompt_infoloaders,
    unprocessable_file_infoloaders,
    file_processing_error_infoloaders,
    image_processing_error_infoloaders,
    corrupted_zip_infoloaders,
    zip_processing_error_infoloaders,
    unsupported_format_infoloaders,
    file_open_error_infoloaders,
    zip_archive_name_infoloaders,
    model_early_loading_error_text,
    excel_cheet_text,
    excel_cheet_size_text,
    excel_cheet_strings_text,
    excel_cheet_columns_text,
    excel_cheet_data_text,
    excel_cheet_error_text,
    excel_empty_text,
    excel_error_text,
)
def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def get_external_path(relative_path):
    """Получает путь к файлу/папке рядом с .exe или скриптом"""
    if hasattr(sys, '_MEIPASS'):
        # Если запущено как exe, берем директорию, где лежит сам exe
        return os.path.join(os.path.dirname(sys.executable), relative_path)
    # Если запущен просто .py, берем текущую папку
    return os.path.join(os.path.abspath("."), relative_path)

# Глобальные переменные для ленивой загрузки моделей
_BLIP_PROCESSOR = None
_BLIP_MODEL = None
_BLIP_TOKENIZER = None
_CLIP_PROCESSOR = None
_CLIP_MODEL = None
_OCR_INSTANCE = None
_IMAGE_MODELS_LOADED = False
_IMAGE_MODELS_LOAD_FAILED = False  # Маркер ошибки загрузки моделей

def _load_image_models():
    """Ленивая загрузка моделей обработки изображений из внешней папки data/models/"""
    global _BLIP_PROCESSOR, _BLIP_MODEL, _OCR_INSTANCE
    global _IMAGE_MODELS_LOADED, _IMAGE_MODELS_LOAD_FAILED
    
    if _IMAGE_MODELS_LOADED:
        return True
    
    if _IMAGE_MODELS_LOAD_FAILED:
        raise RuntimeError(model_early_loading_error_text)
    
    let_log("Загрузка моделей обработки изображений из локальных директорий...")
    
    try:
        import torch # попытка решения проблемы dll
        import os
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import easyocr

        # Определяем пути к внешним папкам
        # Ожидаемая структура: [папка с exe]/data/models/blip и easyocr
        blip_path = get_external_path(os.path.join("data", "models", "blip"))
        easyocr_path = get_external_path(os.path.join("data", "models", "easyocr"))

        # 1. Загрузка BLIP
        if os.path.exists(blip_path):
            _BLIP_PROCESSOR = BlipProcessor.from_pretrained(blip_path, use_fast=True)
            _BLIP_MODEL = BlipForConditionalGeneration.from_pretrained(blip_path)
            let_log("BLIP загружен локально")
        else:
            let_log(f"ВНИМАНИЕ: Локальная модель BLIP не найдена в {blip_path}. Попытка загрузки из сети...")
            _BLIP_PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
            _BLIP_MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        # 2. Загрузка EasyOCR
        try:
            # Создаем папку, если её нет, чтобы Reader не ругался
            os.makedirs(easyocr_path, exist_ok=True)
            
            _OCR_INSTANCE = easyocr.Reader(
                ['en', 'ru'], 
                gpu=False,
                model_storage_directory=easyocr_path,
                download_enabled=False # Строго берем только из папки
            )
            let_log("EasyOCR загружен локально")
        except Exception as ocr_error:
            let_log(f"Ошибка загрузки EasyOCR из {easyocr_path}: {ocr_error}")
            _OCR_INSTANCE = None
        
        _IMAGE_MODELS_LOADED = True
        _IMAGE_MODELS_LOAD_FAILED = False
        return True
        
    except Exception as e:
        let_log(f"Критическая ошибка загрузки моделей: {e}")
        _IMAGE_MODELS_LOADED = False
        _IMAGE_MODELS_LOAD_FAILED = True
        raise
'''
def _load_image_models():
    """Ленивая загрузка моделей обработки изображений"""
    global _BLIP_PROCESSOR, _BLIP_MODEL, _OCR_INSTANCE
    global _IMAGE_MODELS_LOADED, _IMAGE_MODELS_LOAD_FAILED
    
    if _IMAGE_MODELS_LOADED:
        return True
    
    if _IMAGE_MODELS_LOAD_FAILED:
        raise RuntimeError(model_early_loading_error_text)
    
    let_log("Загрузка моделей обработки изображений...")
    
    try:
        import os
        # Переменные для Paddle можно оставить, если используете его в других местах
        os.environ['PADDLE_SKIP_CHECK_INSTALL'] = '1'
        
        # --- Настройка пути для EasyOCR ---
        # Папка будет находиться по пути: [папка с exe]\data\models\easyocr
        easyocr_path = get_external_path(os.path.join("data", "models", "easyocr"))
        
        # Если папки еще нет, создадим её (чтобы не было ошибки)
        if not os.path.exists(easyocr_path):
            os.makedirs(easyocr_path, exist_ok=True)

        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        # BLIP (пока качается автоматически)
        _BLIP_PROCESSOR = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base", 
            use_fast=True
        )
        _BLIP_MODEL = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        try:
            let_log('ТЕПЕРЬ OCR')
            import easyocr
            # Инициализируем Reader с указанием внешней папки
            _OCR_INSTANCE = easyocr.Reader(
                ['en', 'ru'], 
                gpu=False,
                model_storage_directory=easyocr_path, # Указываем нашу папку
                download_enabled=False               # Запрещаем качать (берем только из папки)
            )
        except Exception as ocr_error:
            # Исправляем текст лога, так как теперь у нас EasyOCR
            let_log(f"EasyOCR не загружен (проверьте наличие моделей в {easyocr_path}): {ocr_error}")
            _OCR_INSTANCE = None
        
        _IMAGE_MODELS_LOADED = True
        _IMAGE_MODELS_LOAD_FAILED = False
        let_log("Модели обработки изображений загружены")
        return True
        
    except Exception as e:
        let_log(f"Ошибка загрузки моделей: {e}")
        _IMAGE_MODELS_LOADED = False
        _IMAGE_MODELS_LOAD_FAILED = True
        raise
'''
def process_image(image_content, input_file_handlers):
    """Обработка изображения с использованием BLIP и EasyOCR"""
    let_log('Обработка изображения')
    
    try:
        _load_image_models()
    except Exception as e:
        return f"Ошибка загрузки моделей: {e}"
    
    try:
        if isinstance(image_content, bytes):
            image = Image.open(BytesIO(image_content)).convert('RGB')
        else:
            image = Image.open(image_content).convert('RGB')
    except Exception as e:
        return f"{err_image_process_text_infoloaders}{e}"

    # BLIP анализ (описание картинки)
    try:
        inputs = _BLIP_PROCESSOR(image, return_tensors="pt")
        output_ids = _BLIP_MODEL.generate(**inputs, max_length=50, num_beams=4)
        caption = _BLIP_PROCESSOR.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        let_log(f"BLIP ошибка: {e}")
        caption = "Не удалось создать описание"
    
    # EasyOCR обработка (извлечение текста)
    extracted_text = ""
    if _OCR_INSTANCE is not None:
        try:
            # Преобразуем PIL Image в numpy array для EasyOCR
            img_np = np.array(image)
            # Вызываем специфичный для EasyOCR метод .readtext()
            results = _OCR_INSTANCE.readtext(img_np)
            # Извлекаем только текст (он находится во втором элементе кортежа: (bbox, text, prob))
            extracted_text = "\n".join([res[1] for res in results])
        except Exception as ocr_error:
            let_log(f"EasyOCR не удался: {ocr_error}")
            extracted_text = ""
    
    if extracted_text:
        return f"{caption}\n{text_on_image_prompt_infoloaders}\n{extracted_text}"
    else:
        return f"{caption}"

def cleanup_image_models():
    """
    Очищает глобальные переменные с моделями и вызывает сборщик мусора
    Вызывается после обработки всех пользовательских файлов
    """
    global _BLIP_PROCESSOR, _BLIP_MODEL, _BLIP_TOKENIZER
    global _CLIP_PROCESSOR, _CLIP_MODEL, _OCR_INSTANCE
    global _IMAGE_MODELS_LOADED, _IMAGE_MODELS_LOAD_FAILED
    
    let_log("Очистка моделей обработки изображений...")
    
    # Очищаем переменные
    _BLIP_PROCESSOR = None
    _BLIP_MODEL = None
    _BLIP_TOKENIZER = None
    _CLIP_PROCESSOR = None
    _CLIP_MODEL = None
    _OCR_INSTANCE = None
    _IMAGE_MODELS_LOADED = False
    _IMAGE_MODELS_LOAD_FAILED = False  # Сбрасываем маркер ошибки при очистке
    
    # Принудительный вызов сборщика мусора
    gc.collect()
    let_log("Модели очищены, сборщик мусора вызван")


def process_pdf(file_path, input_file_handlers):
    let_log('пдф')
    try: pdf = pymupdf.open(file_path)
    except Exception as ex: let_log(ex); return f"{err_image_process_pdf_infoloaders}{ex}"
    full_text = []
    attachment_count = 0
    # Проход по страницам документа
    for page_num, page in enumerate(pdf, start=1):
        # Извлечение текста со страницы
        text = page.get_text()
        if text.strip():
            full_text.append(f"--- {page_pdf_prompt_infoloaders} {page_num} ---")
            full_text.append(text.strip())
        # Извлечение изображений
        for image_index, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]  # Индекс изображения
            base_image = pdf.extract_image(xref)
            image_bytes = base_image["image"]
            extension = base_image["ext"]
            attachment_count += 1
            file_name = f"{attachment_prompt_infoloaders}{attachment_count}"
            try:
                # Теперь передаем изображение как байты
                result = input_file_handlers.get(extension, lambda x: unprocessable_file_infoloaders)(image_bytes, input_file_handlers)
                full_text.append(f"[{file_name}: {result}]")
            except Exception as ex: let_log(ex); full_text.append(f"[{file_name}: {image_processing_error_infoloaders} ({ex})]")
    pdf.close()
    return "\n".join(full_text)

def process_docx(file_path, input_file_handlers):
    let_log('док икс')
    try:
        # Открываем документ
        doc = Document(file_path)
    except Exception as e: return f"{file_open_error_infoloaders}{e}"
    full_text = []
    attachment_count = 0
    # Проход по всех элементов документа
    for paragraph in doc.paragraphs:
        # Добавляем текст из параграфа
        if paragraph.text.strip():
            full_text.append(paragraph.text.strip())
        # Проверяем наличие изображений в параграфе
        for run in paragraph.runs:
            if run.element.xpath('.//a:blip/@r:embed'):
                attachment_count += 1
                # Получаем ID изображения
                embed_id = run.element.xpath('.//a:blip/@r:embed')[0]
                # Извлекаем часть документа, соответствующую медиафайлу
                media_part = doc.part.related_parts[embed_id]
                # Определяем расширение файла
                extension = os.path.splitext(media_part.partname)[-1]
                file_name_with_ext = f"{attachment_prompt_infoloaders}{attachment_count}{extension}"
                try:
                    # Передаём содержимое изображения в обработчик
                    result = input_file_handlers.get(extension[1:], lambda x: unprocessable_file_infoloaders)(media_part.blob, input_file_handlers)
                    full_text.append(f"[{file_name_with_ext}: {result}]")
                except Exception as e: full_text.append(f"[{file_name_with_ext}: {file_processing_error_infoloaders} ({e})]")
    return "\n".join(full_text)

def process_zip(file_path_or_data, input_file_handlers):
    let_log('ZIP-архив')
    results = []  # Теперь возвращаем список результатов
    
    try:
        # Определяем, является ли вход данными или путём к файлу
        if isinstance(file_path_or_data, bytes):
            zip_file = zipfile.ZipFile(BytesIO(file_path_or_data))
        else:
            zip_file = zipfile.ZipFile(file_path_or_data)
            
        with zip_file:
            for file_info in zip_file.infolist():
                # Пропускаем директории
                if file_info.is_dir():
                    continue
                    
                # Получаем расширение файла
                _, ext = os.path.splitext(file_info.filename)
                ext = ext[1:].lower()  # Убираем точку и приводим к нижнему регистру
                
                # Читаем содержимое файла
                file_content = zip_file.read(file_info)
                
                # Обрабатываем файл соответствующим обработчиком
                if ext in input_file_handlers:
                    try:
                        result = input_file_handlers[ext](file_content, input_file_handlers)
                        results.append({
                            'filename': file_info.filename,
                            'content': result,
                            'type': 'file'
                        })
                    except Exception as e:
                        results.append({
                            'filename': file_info.filename,
                            'content': f"{file_processing_error_infoloaders}{e}",
                            'type': 'error'
                        })
                else:
                    # Если формат не поддерживается, просто добавляем информацию о файле
                    results.append({
                        'filename': file_info.filename,
                        'content': unsupported_format_infoloaders,
                        'type': 'unsupported'
                    })
                    
    except zipfile.BadZipFile:
        return [{'filename': zip_archive_name_infoloaders, 'content': corrupted_zip_infoloaders, 'type': 'error'}]
    except Exception as e:
        return [{'filename': zip_archive_name_infoloaders, 'content': f'{zip_processing_error_infoloaders}: {e}', 'type': 'error'}]
    
    return results


def process_text(file_path_or_data, input_file_handlers):
    let_log('text')
    if isinstance(file_path_or_data, str):
        try:
            with open(file_path_or_data, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'utf-8'  # Защита от None
                return raw_data.decode(encoding, errors='replace')  # Обработка ошибок
        except (FileNotFoundError, IOError):
            return file_path_or_data
    else:
        raw_data = file_path_or_data.encode('utf-8', errors='ignore')
        result = chardet.detect(raw_data)
        encoding = result['encoding'] or 'utf-8'  # Защита от None
        return raw_data.decode(encoding, errors='replace')  # Обработка ошибок

def process_excel(file_path_or_data, input_file_handlers):
    """
    Обрабатывает Excel файлы (xlsx, xls) - извлекает данные из всех листов
    """
    let_log('Excel файл')
    
    try:
        # Определяем, является ли вход данными или путём к файлу
        if isinstance(file_path_or_data, bytes):
            # Используем BytesIO для чтения из памяти
            import io
            excel_file = pd.ExcelFile(io.BytesIO(file_path_or_data))
        else:
            excel_file = pd.ExcelFile(file_path_or_data)
        
        results = []
        
        # Проходим по всем листам
        for sheet_name in excel_file.sheet_names:
            try:
                # Читаем лист в DataFrame
                df = excel_file.parse(sheet_name)
                
                # Преобразуем DataFrame в текстовое представление
                # Убираем NaN значения и приводим к строке
                df_text = df.fillna('').astype(str).to_string(index=False)
                
                results.append(f"--- {excel_cheet_text}{sheet_name} ---")
                results.append(f"{excel_cheet_size_text}{df.shape[0]} {excel_cheet_strings_text} {df.shape[1]} {excel_cheet_columns_text}")
                results.append(excel_cheet_data_text)
                results.append(df_text)
                results.append("") # Пустая строка для разделения
                
            except Exception as e:
                results.append(f"{excel_cheet_error_text} '{sheet_name}' {e}")
                continue
        
        excel_file.close()
        
        if results:
            return "\n".join(results)
        else:
            return excel_empty_text
            
    except Exception as e:
        error_msg = f"{excel_error_text} {e}"
        let_log(error_msg)
        return error_msg
