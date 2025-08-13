import os
from docx import Document
import pymupdf as fitz # PyMuPDF
from io import BytesIO
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from paddleocr import PaddleOCR
import chardet
from cross_gpt import let_log
import gc

def process_image(image_content, input_file_handlers):
    let_log('изображение')
    try:
        # Пытаемся открыть данные как изображение
        temp_image = Image.open(BytesIO(image_content))
        temp_image.verify()  # Проверяем валидность изображения
        # Заново открываем изображение для работы
        image = Image.open(BytesIO(image_content))
        image = image.convert('RGB')  # Конвертируем в RGB
    except Exception:
        try:
            with open(image_content, 'rb') as file:
                image = Image.open(file)
                image.verify()  # Проверяем валидность изображения
                image = image.convert('RGB')  # Конвертируем в RGB
        except Exception as e: return f"Ошибка при обработке изображения: {e}"
    img_model_name = "blip"  # Выберите модель: "blip" или "clip"
    if img_model_name == "blip":
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        inputs = processor(image, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=1024)
        caption = processor.decode(output[0], skip_special_tokens=True)
    elif img_model_name == "clip":
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        inputs = processor(images=image, return_tensors="pt")
        features = model.get_image_features(**inputs)
        caption = "Описание изображения: Эмбеддинги изображения сгенерированы."
    else: raise ValueError(f"Неизвестная модель: {img_model_name}")
    ocr = PaddleOCR(lang='en')
    ocr_result = ocr.ocr(image_content, cls=True)
    del ocr
    gc.collect()
    def extract_text(data):
        """Извлечение текста из вложенных структур."""
        extracted = []
        for item in data:
            if isinstance(item, list):  # Если элемент — список, обрабатываем рекурсивно
                extracted.extend(extract_text(item))
            elif isinstance(item, tuple) and isinstance(item[0], str):  # Если это кортеж с текстом
                extracted.append(item[0])
        return extracted
    extracted_texts = extract_text(ocr_result)
    extracted_text = "\n".join(extracted_texts)
    let_log(f"{caption}\nТекст на изображении:\n{extracted_text}")
    return f"{caption}\nТекст на изображении:\n{extracted_text}"

def process_pdf(file_path, input_file_handlers):
    let_log('пдф')
    try: pdf = fitz.open(file_path)
    except Exception as ex: let_log(ex); return f"Ошибка открытия PDF: {ex}"
    full_text = []
    attachment_count = 0
    # Проход по страницам документа
    for page_num, page in enumerate(pdf, start=1):
        # Извлечение текста со страницы
        text = page.get_text()
        if text.strip():
            full_text.append(f"--- Страница {page_num} ---")
            full_text.append(text.strip())
        # Извлечение изображений
        for image_index, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]  # Индекс изображения
            base_image = pdf.extract_image(xref)
            image_bytes = base_image["image"]
            extension = base_image["ext"]
            attachment_count += 1
            file_name = f"Attachment_{attachment_count}"
            try:
                # Теперь передаем изображение как байты
                result = input_file_handlers.get(extension, lambda x: "Необрабатываемый файл")(image_bytes, input_file_handlers)
                full_text.append(f"[{file_name}: {result}]")
            except Exception as ex: let_log(ex); full_text.append(f"[{file_name}: Ошибка обработки изображения ({ex})]")
    pdf.close()
    return "\n".join(full_text)

def process_docx(file_path, input_file_handlers):
    let_log('док икс')
    try:
        # Открываем документ
        doc = Document(file_path)
    except Exception as e: return f"Ошибка открытия файла: {e}"
    full_text = []
    attachment_count = 0
    # Проход по всем элементам документа
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
                file_name_with_ext = f"Attachment_{attachment_count}{extension}"
                try:
                    # Передаём содержимое изображения в обработчик
                    result = input_file_handlers.get(extension[1:], lambda x: "Необрабатываемый файл")(media_part.blob, input_file_handlers)
                    full_text.append(f"[{file_name_with_ext}: {result}]")
                except Exception as e: full_text.append(f"[{file_name_with_ext}: Ошибка обработки файла ({e})]")
    return "\n".join(full_text)

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