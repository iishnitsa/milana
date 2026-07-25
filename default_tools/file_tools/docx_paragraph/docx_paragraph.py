'''
docx_paragraph
smart .docx paragraph read/edit by natural language request with path detection and fuzzy matching; Send a request like: "read paragraph about 'old text' in document.docx" or "replace 'old text' with 'new text' in report.docx". Mode and file are auto-detected
Docx paragraph
Reads or edits a paragraph in .docx via filesystem pipeline and intention checks
'''

import os
import re
import json
import difflib
import zipfile
import xml.etree.ElementTree as et

from cross_gpt import ask_model, filesystem_project_path, get_embs
from filesystem import pipeline, status_success, status_failed, status_forbidden


def _scan_docx_files(root_path, max_files=400):
    out = []
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', '.venv', 'venv')]
        for name in files:
            if not name.lower().endswith('.docx'):
                continue
            abs_path = os.path.join(root, name)
            rel_path = os.path.relpath(abs_path, root_path).replace('\\', '/')
            out.append({'abs': abs_path, 'rel': rel_path, 'name': name})
            if len(out) >= max_files:
                return out
    return out


def _cosine(a, b):
    if not a or not b:
        return 0.0
    size = min(len(a), len(b))
    if size <= 0:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(size):
        x = float(a[i])
        y = float(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def _text_score(query_text, paragraph_text):
    if not query_text or not paragraph_text:
        return 0.0
    ratio = difflib.SequenceMatcher(None, query_text.lower(), paragraph_text.lower()).ratio()
    try:
        v1 = get_embs(query_text)
        v2 = get_embs(paragraph_text)
        sim = _cosine(v1, v2)
    except Exception:
        sim = 0.0
    return ratio * 0.45 + sim * 0.55


def _read_docx_paragraphs(docx_path):
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with zipfile.ZipFile(docx_path, 'r') as zf:
        xml_bytes = zf.read('word/document.xml')
    root = et.fromstring(xml_bytes)
    out = []
    idx = 0
    for p in root.findall('.//w:p', ns):
        texts = []
        for t_node in p.findall('.//w:t', ns):
            texts.append(t_node.text or '')
        text = ''.join(texts).strip()
        if text:
            out.append({'index': idx, 'text': text})
            idx += 1
    return out


def _replace_docx_paragraph(docx_path, paragraph_index, new_text):
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    et.register_namespace('w', ns['w'])

    with zipfile.ZipFile(docx_path, 'r') as zin:
        infos = zin.infolist()
        payload = {}
        for info in infos:
            payload[info.filename] = zin.read(info.filename)

    root = et.fromstring(payload['word/document.xml'])
    hit = -1
    changed = False
    for p in root.findall('.//w:p', ns):
        t_nodes = p.findall('.//w:t', ns)
        paragraph_text = ''.join((n.text or '') for n in t_nodes).strip()
        if not paragraph_text:
            continue
        hit += 1
        if hit != int(paragraph_index):
            continue
        if not t_nodes:
            return False, 'В найденном абзаце нет текстовых нод'
        t_nodes[0].text = str(new_text)
        for node in t_nodes[1:]:
            node.text = ''
        changed = True
        break

    if not changed:
        return False, 'Абзац не найден'

    payload['word/document.xml'] = et.tostring(root, encoding='utf-8', xml_declaration=True)

    tmp_out = docx_path + '.tmp'
    with zipfile.ZipFile(tmp_out, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
        for info in infos:
            data = payload.get(info.filename, b'')
            zout.writestr(info, data)

    os.replace(tmp_out, docx_path)
    return True, 'ok'


def _extract_json(text):
    if not text:
        return None
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    match = re.search(r'\{.*\}', text, flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _extract_request_struct(request_text, candidates):
    # Локализуемые ключевые слова
    edit_keywords = main.edit_keywords
    read_keywords = main.read_keywords

    mode = 'read'
    low = request_text.lower()
    if any(k in low for k in edit_keywords):
        mode = 'edit'
    elif any(k in low for k in read_keywords):
        mode = 'read'

    quoted = re.findall(r'["“”](.+?)["“”]', request_text)
    target_query = quoted[0].strip() if quoted else ''
    replacement = quoted[1].strip() if len(quoted) > 1 else ''

    candidates_view = '\n'.join([f"{i+1}. {c['rel']}" for i, c in enumerate(candidates[:120])])

    # Собираем промпт
    prompt = main.prompt_extract_part1 + candidates_view + main.prompt_extract_part2

    try:
        raw = ask_model(prompt, all_user=True)
        parsed = _extract_json(raw)
    except Exception:
        parsed = None

    if not parsed:
        parsed = {}
    parsed.setdefault('mode', mode)
    parsed.setdefault('file_hint', '')
    parsed.setdefault('target_query', target_query)
    parsed.setdefault('replacement', replacement)
    return parsed


def _pick_docx(request_struct, request_text, candidates):
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    hint = str(request_struct.get('file_hint') or '').strip().lower()
    if hint:
        for c in candidates:
            if hint in c['rel'].lower() or hint in c['name'].lower():
                return c

    # Собираем промпт
    file_list = '\n'.join([c['rel'] for c in candidates[:120]])
    prompt = main.prompt_pick_file + "\n" + file_list
    try:
        chosen = ask_model(prompt, all_user=True).strip()
    except Exception:
        chosen = ''
    chosen_low = chosen.lower()
    for c in candidates:
        if chosen_low == c['rel'].lower() or chosen_low in c['rel'].lower():
            return c
    return candidates[0]


def _best_paragraph(paragraphs, query_text):
    if not paragraphs:
        return None
    if not query_text:
        return {'index': paragraphs[0]['index'], 'text': paragraphs[0]['text'], 'score': 0.0}
    best = None
    best_score = -1.0
    for p in paragraphs:
        score = _text_score(query_text, p['text'])
        if score > best_score:
            best_score = score
            best = p
    return {'index': best['index'], 'text': best['text'], 'score': best_score}


def main(text):
    if not hasattr(main, 'attr_names'):
        main.attr_names = (
            'no_docx_text',
            'cannot_parse_text',
            'edit_need_replacement_text',
            'paragraph_not_found_text',
            'done_text',
            'forbidden_text',
            'failed_text',
            'prompt_extract_part1',
            'prompt_extract_part2',
            'prompt_pick_file',
            'edit_keywords',
            'read_keywords',
        )
        main.no_docx_text = 'Docx файлы не найдены'
        main.cannot_parse_text = 'Не удалось разобрать запрос'
        main.edit_need_replacement_text = 'Для edit нужен новый текст абзаца (replacement)'
        main.paragraph_not_found_text = 'Абзац не найден'
        main.done_text = 'Готово'
        main.forbidden_text = 'Запрещено'
        main.failed_text = 'Ошибка'
        main.prompt_extract_part1 = """
Извлеки структуру запроса для работы с docx.
Верни ТОЛЬКО JSON без пояснений с ключами:
{"mode":"read|edit","file_hint":"...","target_query":"...","replacement":"..."}
Запрос:
"""
        main.prompt_extract_part2 = """

Кандидаты файлов:
"""
        main.prompt_pick_file = """
Выбери один файл docx для запроса.
Верни только относительный путь как в списке, без комментариев.
Запрос: """ + text + """
Список:"""
        main.edit_keywords = ('замени', 'поменя', 'измени', 'replace', 'edit')
        main.read_keywords = ('прочита', 'покажи', 'read', 'show')
        return

    workspace = filesystem_project_path
    candidates = _scan_docx_files(workspace)
    if not candidates:
        return main.no_docx_text

    request_struct = _extract_request_struct(text, candidates)
    mode = str(request_struct.get('mode') or '').strip().lower()
    if mode not in ('read', 'edit'):
        mode = 'read'

    target_doc = _pick_docx(request_struct, text, candidates)
    if not target_doc:
        return main.no_docx_text

    query_text = str(request_struct.get('target_query') or '').strip()
    replacement_text = str(request_struct.get('replacement') or '').strip()
    if mode == 'edit' and not replacement_text:
        return main.edit_need_replacement_text

    if mode == 'read':
        def read_actor(payload):
            paragraphs = _read_docx_paragraphs(payload['temp_file'])
            best = _best_paragraph(paragraphs, payload.get('arg', {}).get('target_query', ''))
            if not best:
                return {'status': status_failed, 'reason': main.paragraph_not_found_text}
            report = {
                'file': target_doc['rel'],
                'paragraph_index': best['index'],
                'score': round(best['score'], 4),
                'text': best['text'],
            }
            return {'status': status_success, 'data': report}

        result = pipeline(
            read_actor,
            target_doc['abs'],
            action='read',
            handler_arg={'target_query': query_text},
            repo_path=workspace,
            use_git=False,
        )
        if result.get('status') == status_forbidden:
            return f"{main.forbidden_text}: {result.get('reason')}"
        if result.get('status') != status_success:
            return f"{main.failed_text}: {result.get('reason')}"
        report = result.get('data')
        return json.dumps(report, ensure_ascii=False, indent=2)

    def edit_actor(payload):
        paragraphs = _read_docx_paragraphs(payload['temp_file'])
        best = _best_paragraph(paragraphs, payload.get('arg', {}).get('target_query', ''))
        if not best:
            return {'status': status_failed, 'reason': main.paragraph_not_found_text}
        ok, message = _replace_docx_paragraph(
            payload['temp_file'],
            best['index'],
            payload.get('arg', {}).get('replacement', ''),
        )
        if not ok:
            return {'status': status_failed, 'reason': message}
        report = {
            'file': target_doc['rel'],
            'paragraph_index': best['index'],
            'old_text': best['text'],
            'new_text': payload.get('arg', {}).get('replacement', ''),
        }
        return {'status': status_success, 'data': report}

    result = pipeline(
        edit_actor,
        target_doc['abs'],
        action='edit',
        handler_arg={'target_query': query_text, 'replacement': replacement_text},
        repo_path=workspace,
        use_git=False,
    )
    if result.get('status') == status_forbidden:
        return f"{main.forbidden_text}: {result.get('reason')}"
    if result.get('status') != status_success:
        return f"{main.failed_text}: {result.get('reason')}"
    return f"{main.done_text}: {json.dumps(result.get('data'), ensure_ascii=False)}"