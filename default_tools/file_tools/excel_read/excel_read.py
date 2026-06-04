'''
excel_read
read spreadsheet as execution-like operation: inspect csv/tsv/xlsx data and formulas through filesystem read pipeline; Send a request with the file name, e.g. "show contents of data.xlsx". Returns JSON with headers, sample rows, formulas for XLSX
Excel read
Reads spreadsheet and returns compact structured summary for agent
'''

import os
import re
import json
import csv
import zipfile
import difflib
import xml.etree.ElementTree as et

from cross_gpt import ask_model, filesystem_project_path, pipeline, status_success, status_failed, status_forbidden


def _scan_table_files(root_path, max_files=500):
    exts = ('.xlsx', '.csv', '.tsv')
    out = []
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', '.venv', 'venv')]
        for name in files:
            if not name.lower().endswith(exts):
                continue
            abs_path = os.path.join(root, name)
            rel_path = os.path.relpath(abs_path, root_path).replace('\\', '/')
            out.append({'abs': abs_path, 'rel': rel_path, 'name': name})
            if len(out) >= max_files:
                return out
    return out


def _pick_file_from_request(request_text, files):
    if not files:
        return None
    if len(files) == 1:
        return files[0]

    low = request_text.lower()
    best = None
    best_score = -1.0
    for item in files:
        rel_low = item['rel'].lower()
        name_low = item['name'].lower()
        score = max(
            difflib.SequenceMatcher(None, low, rel_low).ratio(),
            difflib.SequenceMatcher(None, low, name_low).ratio(),
        )
        if name_low in low or rel_low in low:
            score += 0.25
        if score > best_score:
            best = item
            best_score = score

    if best_score > 0.42:
        return best

    # Собираем промпт
    file_list = '\n'.join([f['rel'] for f in files[:150]])
    prompt = main.prompt_pick_file + "\n" + file_list
    try:
        picked = ask_model(prompt, all_user=True).strip().lower()
    except Exception:
        picked = ''
    for item in files:
        if picked == item['rel'].lower() or picked in item['rel'].lower():
            return item
    return best if best else files[0]


def _read_csv_like(path_text, delimiter):
    rows = []
    with open(path_text, 'r', encoding='utf-8', errors='replace', newline='') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for i, row in enumerate(reader):
            rows.append(row)
            if i >= 200:
                break
    header = rows[0] if rows else []
    body = rows[1:] if len(rows) > 1 else []
    return {
        'format': 'csv' if delimiter == ',' else 'tsv',
        'rows_loaded': len(rows),
        'columns': len(header),
        'header': header[:50],
        'sample_rows': body[:10],
    }


def _xlsx_shared_strings(zf):
    path = 'xl/sharedStrings.xml'
    if path not in zf.namelist():
        return []
    ns = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    root = et.fromstring(zf.read(path))
    out = []
    for si in root.findall('.//x:si', ns):
        parts = []
        for t in si.findall('.//x:t', ns):
            parts.append(t.text or '')
        out.append(''.join(parts))
    return out


def _xlsx_sheets(zf):
    ns = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    root = et.fromstring(zf.read('xl/workbook.xml'))
    out = []
    for sheet in root.findall('.//x:sheets/x:sheet', ns):
        out.append({
            'name': sheet.attrib.get('name', ''),
            'sheet_id': sheet.attrib.get('sheetId', ''),
            'rid': sheet.attrib.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id', ''),
        })
    return out


def _xlsx_sheet_paths(zf):
    rel_path = 'xl/_rels/workbook.xml.rels'
    if rel_path not in zf.namelist():
        return {}
    ns = {'r': 'http://schemas.openxmlformats.org/package/2006/relationships'}
    root = et.fromstring(zf.read(rel_path))
    out = {}
    for rel in root.findall('.//r:Relationship', ns):
        rid = rel.attrib.get('Id')
        target = rel.attrib.get('Target')
        if rid and target:
            out[rid] = 'xl/' + target.replace('\\', '/').lstrip('/')
    return out


def _xlsx_sheet_summary(zf, sheet_xml_path, shared_strings):
    ns = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    root = et.fromstring(zf.read(sheet_xml_path))
    formula_cells = []
    value_cells = []
    sample = []
    total_cells = 0

    for c in root.findall('.//x:c', ns):
        total_cells += 1
        cell_ref = c.attrib.get('r', '')
        cell_type = c.attrib.get('t', '')
        f_node = c.find('x:f', ns)
        v_node = c.find('x:v', ns)
        formula = f_node.text if f_node is not None else None
        raw_value = v_node.text if v_node is not None else None

        value = raw_value
        if cell_type == 's' and raw_value is not None:
            try:
                idx = int(raw_value)
                value = shared_strings[idx] if idx < len(shared_strings) else raw_value
            except Exception:
                value = raw_value

        if formula:
            formula_cells.append({'cell': cell_ref, 'formula': formula, 'value': value})
        if value is not None and value != '':
            value_cells.append({'cell': cell_ref, 'value': value})

        if len(sample) < 30:
            sample.append({'cell': cell_ref, 'value': value, 'formula': formula})

    return {
        'total_cells': total_cells,
        'formula_cells_count': len(formula_cells),
        'value_cells_count': len(value_cells),
        'formula_examples': formula_cells[:20],
        'sample_cells': sample,
    }


def _read_xlsx(path_text):
    with zipfile.ZipFile(path_text, 'r') as zf:
        sheets = _xlsx_sheets(zf)
        rel_map = _xlsx_sheet_paths(zf)
        shared = _xlsx_shared_strings(zf)

        sheet_reports = []
        for sheet in sheets:
            rid = sheet.get('rid')
            xml_path = rel_map.get(rid)
            if not xml_path or xml_path not in zf.namelist():
                continue
            summary = _xlsx_sheet_summary(zf, xml_path, shared)
            sheet_reports.append({
                'name': sheet.get('name', ''),
                'sheet_id': sheet.get('sheet_id', ''),
                'path': xml_path,
                'summary': summary,
            })

    return {
        'format': 'xlsx',
        'sheets_count': len(sheet_reports),
        'sheets': sheet_reports[:8],
    }


def _table_report_from_file(path_text):
    ext = os.path.splitext(path_text)[1].lower()
    if ext == '.csv':
        return _read_csv_like(path_text, ',')
    if ext == '.tsv':
        return _read_csv_like(path_text, '\t')
    if ext == '.xlsx':
        return _read_xlsx(path_text)
    return {'format': ext, 'error': 'Неподдерживаемый формат'}


def main(text):
    if not hasattr(main, 'attr_names'):
        main.attr_names = (
            'no_files_text',
            'failed_text',
            'forbidden_text',
            'done_text',
            'prompt_pick_file',
        )
        main.no_files_text = 'Файлы таблиц не найдены'
        main.failed_text = 'Ошибка'
        main.forbidden_text = 'Запрещено'
        main.done_text = 'Готово'
        main.prompt_pick_file = """
Выбери один файл таблицы для запроса.
Верни только относительный путь без пояснений.
Запрос: """ + text + """
Список:"""
        return

    workspace = filesystem_project_path
    files = _scan_table_files(workspace)
    if not files:
        return main.no_files_text
    chosen = _pick_file_from_request(text, files)
    if not chosen:
        return main.no_files_text

    def read_actor(payload):
        report = _table_report_from_file(payload['temp_file'])
        report['file'] = chosen['rel']
        return {'status': status_success, 'data': report}

    result = pipeline(
        read_actor,
        chosen['abs'],
        action='read',
        repo_path=workspace,
        use_git=False,
    )
    if result.get('status') == status_forbidden:
        return f"{main.forbidden_text}: {result.get('reason')}"
    if result.get('status') != status_success:
        return f"{main.failed_text}: {result.get('reason')}"
    return f"{main.done_text}: {json.dumps(result.get('data'), ensure_ascii=False, indent=2)}"