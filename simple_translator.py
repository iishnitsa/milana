import sqlite3
import os
import sys
import re
import time
import random
import translators as ts
from cross_gpt import cacher, sql_exec, global_trans_cache_exec, use_local_cache, use_global_cache

# Единый список переводчиков: (имя, макс_символов, таймаут)
TRANSLATORS = [
    ('yandex',   10000, 10),
    ('baidu',    6000, 10),
    ('deepl',    5000, 30),
    ('google',   2000, 15),
    ('bing',     5000, 10),
    ('caiyun',   5000, 10),
    ('niutrans', 5000, 10)]

def get_translation_from_cache(text, to_lang):
    """Проверяет локальный кэш (chatsettings.db). Глобальный НЕ проверяется."""
    if use_local_cache:
        result = sql_exec("SELECT translation FROM translation_cache WHERE src_text=? AND to_lang=?", (text, to_lang), fetchone=True)
        if result: return result[0]
    return None

def save_translation_to_cache(text, translation, to_lang):
    """Сохраняет перевод в локальный и глобальный кэш (без from_lang)."""
    if use_local_cache: sql_exec("INSERT INTO translation_cache (src_text, translation, to_lang) VALUES (?, ?, ?)", (text, translation, to_lang))
    if use_global_cache: global_trans_cache_exec("INSERT INTO translation_cache (src_text, translation, to_lang) VALUES (?, ?, ?)", (text, translation, to_lang))

def _get_max_chars(translator_name):
    """Возвращает максимальное количество символов для указанного переводчика."""
    for name, max_chars, _ in TRANSLATORS:
        if name == translator_name: return max_chars
    raise ValueError(f"Неизвестный переводчик: {translator_name}")

def _split_text(text, max_chars):
    """Разбивает текст на части, стараясь сохранить целостность предложений."""
    if not text.strip(): return [text]
    if len(text) <= max_chars: return [text]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for sent in sentences:
        if not sent.strip(): continue
        if len(current) + len(sent) + 1 <= max_chars: current += (sent + " ") if current else sent
        else:
            if current: chunks.append(current.strip())
            if len(sent) > max_chars:
                words = sent.split()
                temp = ""
                for w in words:
                    if len(temp) + len(w) + 1 <= max_chars:
                        temp += (w + " ") if temp else w
                    else:
                        if temp: chunks.append(temp.strip())
                        temp = w
                if temp: current = temp
                else: current = ""
            else: current = sent
    if current: chunks.append(current.strip())
    if not chunks: return [text]
    return chunks

def translate_command_name(name, from_lang, to_lang, translator=None):
    """
    Переводит имя команды, разбивая по '_' или пробелам.
    Возвращает строку в нижнем регистре с подчёркиваниями.
    """
    if not name or not name.strip():
        return name
    # Определяем разделитель: если есть '_', используем его, иначе пробел
    if '_' in name:
        parts = name.split('_')
    else:
        parts = name.split()
    translated_parts = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Переводим часть
        if translator:
            trans = ts.translate_text(part, from_language=from_lang, to_language=to_lang, translator=translator)
        else:
            trans = ts.translate_text(part, from_language=from_lang, to_language=to_lang)
        # Приводим к нижнему регистру и убираем лишние пробелы
        trans = trans.strip().lower()
        # Удаляем знаки препинания (можно оставить только буквы, цифры и подчёркивания)
        trans = re.sub(r'[^\w]', '', trans)
        if trans:
            translated_parts.append(trans)
    # Если после перевода ничего не осталось, возвращаем оригинал в нижнем регистре
    if not translated_parts:
        return name.lower().replace(' ', '_')
    return '_'.join(translated_parts)

def translate_with_preserve(text, to_lang, from_lang='auto', translator=None):
    """
    Переводит текст, сохраняя:
      - блоки кода (```...```)
      - инлайн-код (`...`)
      - маркеры команд (!!!имя!!! и вариации)
    Имена команд переводятся с сохранением структуры (разбиваются по '_' или пробелам).
    """
    if not text or not text.strip():
        return text

    # Паттерны
    pattern_code = r'```.*?```'          # блоки кода
    pattern_inline = r'`[^`]*`'          # инлайн-код
    pattern_command = r'[!¡]{1,3}\s*([\w\s]+?)\s*[!¡]{1,3}'  # команды

    # 1. Заменяем кодовые блоки
    code_blocks = []
    def repl_code(m):
        placeholder = f'__SPECIAL_CODE_{len(code_blocks)}__'
        code_blocks.append((placeholder, m.group(0)))
        return placeholder
    processed = re.sub(pattern_code, repl_code, text, flags=re.DOTALL)

    # 2. Заменяем инлайн-код
    inline_blocks = []
    def repl_inline(m):
        placeholder = f'__SPECIAL_INLINE_{len(inline_blocks)}__'
        inline_blocks.append((placeholder, m.group(0)))
        return placeholder
    processed = re.sub(pattern_inline, repl_inline, processed)

    # 3. Заменяем команды
    command_blocks = []
    def repl_command(m):
        full = m.group(0)
        name = m.group(1).strip()
        placeholder = f'__SPECIAL_CMD_{len(command_blocks)}__'
        command_blocks.append((placeholder, full, name))
        return placeholder
    processed = re.sub(pattern_command, repl_command, processed)

    # 4. Переводим оставшийся текст
    if processed.strip():
        if translator:
            translated = ts.translate_text(processed, from_language=from_lang, to_language=to_lang, translator=translator)
        else:
            translated = ts.translate_text(processed, from_language=from_lang, to_language=to_lang)
    else:
        translated = ''

    # 5. Восстанавливаем команды (с переводом имени)
    for placeholder, full, name in command_blocks:
        # Переводим имя команды
        translated_name = translate_command_name(name, from_lang, to_lang, translator)
        # Ищем в full позиции имени, чтобы сохранить форматирование (восклицательные знаки, пробелы)
        match = re.search(pattern_command, full)
        if match:
            start_name = match.start(1)
            end_name = match.end(1)
            new_full = full[:start_name] + translated_name + full[end_name:]
        else:
            # fallback
            new_full = f"!!!{translated_name}!!!"
        translated = translated.replace(placeholder, new_full)

    # 6. Восстанавливаем инлайн-код
    for placeholder, original in inline_blocks:
        translated = translated.replace(placeholder, original)

    # 7. Восстанавливаем кодовые блоки
    for placeholder, original in code_blocks:
        translated = translated.replace(placeholder, original)

    return translated

@cacher
def _translate_single_chunk(chunk, from_lang, to_lang, translator, timeout):
    """
    Переводит один чанк через указанного переводчика с сохранением специальных блоков.
    Возвращает строку или выбрасывает исключение.
    """
    time.sleep(random.uniform(0.1, 0.5))
    # Используем функцию с сохранением специальных блоков
    return translate_with_preserve(chunk, to_lang, from_lang, translator)

def _translate_one_text(text, from_lang, to_lang, translator, timeout):
    """
    Переводит один текст (возможно, длинный) через указанного переводчика.
    Разбивает на чанки, если нужно, и объединяет результат.
    Если какой-либо чанк не удаётся – выбрасывает исключение.
    """
    max_chars = _get_max_chars(translator)
    if len(text) <= max_chars:
        return _translate_single_chunk(text, from_lang, to_lang, translator, timeout)
    chunks = _split_text(text, max_chars)
    translated_chunks = []
    for chunk in chunks:
        translated_chunks.append(_translate_single_chunk(chunk, from_lang, to_lang, translator, timeout))
    return " ".join(translated_chunks)

def _try_translate_batch(texts, from_lang, to_lang, translator, timeout):
    """
    Пытается перевести весь список текстов одним переводчиком.
    Возвращает список переводов, если все успешно, иначе выбрасывает исключение.
    """
    results = []
    for text in texts:
        results.append(_translate_one_text(text, from_lang, to_lang, translator, timeout))
    return results

def _translate_batch_with_fallback(texts, from_lang, to_lang):
    """
    Переводит список текстов с использованием кругов и fallback между переводчиками.
    Возвращает список переводов или выбрасывает исключение, если ни один переводчик не сработал.
    """
    if not texts: return []
    max_rounds = 2
    for round_num in range(1, max_rounds + 1):
        for translator, _, timeout in TRANSLATORS:
            try:
                return _try_translate_batch(texts, from_lang, to_lang, translator, timeout)
            except Exception:
                continue
    raise RuntimeError("Не удалось перевести тексты ни одним переводчиком после двух кругов попыток.")

@cacher
def translate_texts(texts, to_lang, from_lang='auto'):
    """
    Переводит список текстов.
    Сначала проверяет кэш для каждого текста, непереведённые отправляет на пакетный перевод.
    Возвращает список переводов в том же порядке.
    """
    if not texts: return []
    result = []
    to_translate = []
    indices = []
    for i, text in enumerate(texts):
        if not text or not text.strip():
            result.append("")
            continue
        cached = get_translation_from_cache(text, to_lang)
        if cached is not None:
            result.append(cached)
        else:
            result.append(None)
            to_translate.append(text)
            indices.append(i)
    if to_translate:
        translations = _translate_batch_with_fallback(to_translate, from_lang, to_lang)
        for idx, trans in zip(indices, translations):
            result[idx] = trans
        for text, trans in zip(to_translate, translations):
            save_translation_to_cache(text, trans, to_lang)
    return result

def translate_text(text, to_lang, from_lang='auto'): return translate_texts([text], to_lang, from_lang)[0]