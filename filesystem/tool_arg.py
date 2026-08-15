"""
Очистка аргументов FS-tools от «хвостов» слабых моделей.

1) one_line_arg  — cd / tree / move / run (одна команда)
2) clean_tool_arg — NL tools (smart_file, docx, excel): можно несколько строк
   полезного текста, но отрезаем ремарки модели после пустой строки / в скобках.

Надёжность: режем только то, что похоже на мета-комментарий, не трогаем
тело файла (print(...), csv-строки, код) в середине запроса.
"""

from __future__ import annotations

import re

# Строка целиком — ремарка модели (не содержимое файла)
_META_LINE = re.compile(
    r"""(?ix)^
    (
        \*\([^\)]*\)\*           # *(…)*
      | \([^\)]{0,200}\)         # (Продолжай…)
      | \*{1,2}[^*].*            # *italic* / **bold** only line
      | \#{1,3}\s                # markdown heading
      | -{3,}                    # ---
      | (продолж|дождись|проверь|проверьте|исправлено|внимание
        |повторяю|попробуй|третья\s+попытка|вторая\s+попытка
        |убедись|чтобы\s+продолжить|после\s+получения
        |continue|please\s+continue|note:|wait\s+for|check\s+that
        |only\s+after|do\s+not|don't)
    )
    """,
)

# Блок после \n\n — мета, если так начинается
_META_BLOCK_START = re.compile(
    r"""(?ix)^\s*
    (
        \*\( | \(
      | \*\*?
      | \#{1,3}\s
      | -{3,}
      | (продолж|дождись|проверь|проверьте|исправлено|внимание
        |повторяю|попробуй|третья|вторая
        |continue|please|note:|wait|check|only\s+after)
    )
    """,
)


def _looks_meta_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if _META_LINE.match(s):
        return True
    # короткая ремарка только в скобках / звёздочках
    if s.startswith("(") and s.endswith(")") and len(s) < 220:
        return True
    if s.startswith("*(") and ")*" in s:
        return True
    return False


def _looks_meta_block(block: str) -> bool:
    s = (block or "").strip()
    if not s:
        return True
    first = s.split("\n", 1)[0].strip()
    if _META_BLOCK_START.match(first) or _looks_meta_line(first):
        return True
    # весь блок — одни скобочные ремарки
    lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
    if lines and all(_looks_meta_line(ln) for ln in lines):
        return True
    return False


def _strip_trailing_remark_on_line(line: str) -> str:
    line = line.rstrip()
    line = re.sub(r"\s*\*\([^\)]*\)\*\s*$", "", line).strip()
    line = re.sub(r"\s+\((?=[^)]*[A-Za-zА-Яа-я])[^)]{0,200}\)\s*$", "", line).strip()
    line = re.sub(r"\s*\*\*[^*]+\*\*\s*$", "", line).strip()
    if len(line) >= 2 and line[0] == "`" and line[-1] == "`":
        line = line[1:-1].strip()
    return line


def _normalize_newlines(text) -> str:
    if text is None:
        return ""
    # Модель часто: !!!cmd!!!\narg  → в main приходит "\narg" или "\n\narg"
    return str(text).replace("\r\n", "\n").replace("\r", "\n")


def _lstrip_blank_lines(s: str) -> str:
    """Убрать ведущие пустые строки / пробелы (arg на следующей строке после !!!cmd!!!)."""
    if not s:
        return ""
    lines = s.split("\n")
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    return "\n".join(lines[i:])


def one_line_arg(text) -> str:
    """Первая непустая строка + срез хвостовой ремарки. Ведущие \\n игнорируются."""
    s = _lstrip_blank_lines(_normalize_newlines(text))
    if not s:
        return ""
    for line in s.split("\n"):
        line = line.strip()
        if not line:
            continue
        if _looks_meta_line(line):
            continue
        return _strip_trailing_remark_on_line(line)
    return ""


def clean_tool_arg(text) -> str:
    """
    NL-аргумент: сохраняем полезные строки (в т.ч. многострочный content),
    отрезаем блоки/строки-ремарки модели.
    Ведущие пустые строки (команда на одной строке, arg на следующей) снимаются.
    """
    s = _lstrip_blank_lines(_normalize_newlines(text)).strip()
    if not s:
        return ""

    # 1) Режем по двойному переводу строки, если дальше мета-блок
    paragraphs = re.split(r"\n\s*\n", s)
    kept_paras = []
    for i, para in enumerate(paragraphs):
        para = para.strip("\n")
        if not para.strip():
            continue
        if kept_paras and _looks_meta_block(para):
            break
        # внутри абзаца — построчно
        lines_out = []
        for line in para.split("\n"):
            st = line.strip()
            if lines_out and _looks_meta_line(st):
                break
            if not lines_out and _looks_meta_line(st) and not _looks_like_payload_line(st):
                # ведущая мета-строка без payload — пропуск
                continue
            lines_out.append(line.rstrip())
        if not lines_out:
            if kept_paras:
                break
            continue
        # хвостовая ремарка на последней строке абзаца
        lines_out[-1] = _strip_trailing_remark_on_line(lines_out[-1])
        kept_paras.append("\n".join(lines_out).strip())
    s = "\n\n".join(p for p in kept_paras if p).strip()

    # 2) ещё раз: хвостовые мета-строки
    lines = s.split("\n")
    while lines and _looks_meta_line(lines[-1].strip()):
        lines.pop()
    if lines:
        lines[-1] = _strip_trailing_remark_on_line(lines[-1])
    return "\n".join(lines).strip()


def _looks_like_payload_line(line: str) -> bool:
    """Строка похожа на path/code/csv, не на ремарку."""
    s = (line or "").strip()
    if not s:
        return False
    if re.search(r"[/\\.]|print\s*\(|,\s*\d|^=|:$", s):
        return True
    if re.match(r"^(созда|прочит|измен|замен|удал|копир|перемест|show|create|read|edit|delete|copy|move|replace)\b", s, re.I):
        return True
    return False


def load_cleaner():
    """Для модулей: вернуть (one_line_arg, clean_tool_arg) из этого файла."""
    return one_line_arg, clean_tool_arg
