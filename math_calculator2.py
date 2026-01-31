import re
import sympy
from sympy import (
    N, oo, integrate, diff, limit, sin, cos, pi, symbols, Eq, solve, simplify, Sum
)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
)

# Настройки парсера
transformations = standard_transformations + (implicit_multiplication_application, convert_xor)

ERRORS = {
    'intent_not_recognized': 'Не удалось распознать математическую задачу. Уточните запрос.',
    'computation_error': 'Ошибка при вычислении.',
    'intent_recognized_prefix': 'Распознана задача: ',
}

def format_result(result):
    """Приводит результат SymPy к человеческому виду."""
    if result is None: return None
    if isinstance(result, (list, tuple, set)):
        if not result: return "Нет решений"
        return ", ".join([f"x = {str(simplify(r)).replace('**', '^')}" for r in result])
    try:
        if hasattr(result, 'is_number') and result.is_number:
            # Если число целое или очень близко к нему
            if abs(result - round(float(result))) < 1e-10:
                return str(int(round(float(result))))
            return str(N(result, 7)).rstrip('0').rstrip('.')
    except: pass
    return str(result).replace('**', '^').replace(' ', '')

def clean_math_body(text):
    """Универсальная очистка математического тела от русского текста."""
    # Удаляем кириллицу
    text = re.sub(r'[а-яА-ЯёЁ]+', ' ', text)
    # Заменяем спецсимволы
    text = text.replace('^', '**').replace('→', '->').replace('∞', 'oo').replace('π', 'pi')
    # Удаляем dx
    text = text.replace('dx', '')
    return text.strip()

def solve_math(text):
    """Ядро логики: распределение по специализированным методам."""
    t_low = text.lower()
    
    # 1. Проценты (самый простой паттерн)
    perc = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:от|of)\s*(\d+(?:\.\d+)?)', t_low)
    if perc:
        return (float(perc.group(1)) / 100) * float(perc.group(2))

    # 2. Интеграл (нужна жесткая привязка к границам)
    if 'интеграл' in t_low:
        bounds = re.findall(r'(\d+|pi|oo)\s+до\s+(\d+|pi|oo)', t_low.replace('∞', 'oo').replace('пи', 'pi'))
        # Выделяем функцию: всё что между 'интеграл' и 'от'/'dx'
        body = re.sub(r'(интеграл|от|до|dx|pi|oo|\d+)', '', t_low).strip()
        expr = parse_expr(clean_math_body(body), transformations=transformations)
        x = list(expr.free_symbols)[0] if expr.free_symbols else symbols('x')
        if bounds:
            return integrate(expr, (x, parse_expr(bounds[0][0]), parse_expr(bounds[0][1])))
        return integrate(expr, x)

    # 3. Предел (отсекаем точку стремления)
    if 'предел' in t_low or 'лимит' in t_low:
        point_match = re.search(r'(?:->|→|к|при|стремится)\s*([a-z]\s*)?(?:->|→|к|)?\s*([\d\.\w\-oo]+)', t_low.replace('∞', 'oo'))
        p_val = parse_expr(point_match.group(2)) if point_match else 0
        # Формула — это то, что ДО слова 'при'/'стремится'
        body = re.split(r'при|стремится|->|→|к', t_low)[0]
        expr = parse_expr(clean_math_body(body), transformations=transformations)
        x = list(expr.free_symbols)[0] if expr.free_symbols else symbols('x')
        return limit(expr, x, p_val)

    # 4. Ряд (нужна переменная n)
    if 'ряд' in t_low or 'сумма' in t_low:
        body = re.split(r'от|при|для|n=', t_low)[0]
        expr = parse_expr(clean_math_body(body), transformations=transformations)
        n = symbols('n') if 'n' in str(expr) else (list(expr.free_symbols)[0] if expr.free_symbols else symbols('n'))
        return Sum(expr, (n, 1, oo)).doit()

    # 5. Уравнение (ищем знак равно)
    if '=' in t_low:
        body = clean_math_body(t_low)
        parts = body.split('=')
        left = parse_expr(parts[0], transformations=transformations)
        right = parse_expr(parts[1], transformations=transformations) if parts[1].strip() else 0
        return solve(Eq(left, right))

    # 6. Общие задачи: Производная, Минимум, Упрощение
    body = clean_math_body(t_low)
    expr = parse_expr(body, transformations=transformations)
    vars = list(expr.free_symbols)
    x = vars[0] if vars else symbols('x')

    if 'производная' in t_low or 'производную' in t_low:
        return diff(expr, x)
    
    if 'минимум' in t_low or 'максимум' in t_low:
        df = diff(expr, x)
        pts = solve(df, x)
        real_pts = [p for p in pts if p.is_real]
        if not real_pts: return None
        vals = [expr.subs(x, p) for p in real_pts]
        return min(vals) if 'минимум' in t_low else max(vals)

    return simplify(expr)

def main(query):
    # Определение интента для красивого вывода
    intent_map = {
        'алгебра': ['уравнение', 'упростить', 'сумма', 'ряд'],
        'мат. анализ': ['производная', 'интеграл', 'предел', 'лимит'],
        'оптимизация': ['минимум', 'максимум']
    }
    
    current_intent = None
    for intent, keys in intent_map.items():
        if any(k in query.lower() for k in keys):
            current_intent = intent
            break

    try:
        result = solve_math(query)
        formatted = format_result(result)
        
        prefix = f"{ERRORS['intent_recognized_prefix']}{current_intent}\n" if current_intent else ""
        return f"{prefix}Результат: {formatted}"
    except Exception as e:
        return f"Ошибка: {str(e)}"

if __name__ == "__main__":
    test_suite = [
        "Посчитай 2+2*2",
        "Сколько будет 15% от 200?",
        "Реши уравнение x^2 - 5x + 6 = 0",
        "Упрости выражение (a+b)^2 - (a-b)^2",
        "Найди производную sin(x^2)",
        "Вычисли интеграл от 0 до pi от sin(x) dx",
        "Найди предел sin(x)/x при x→0",
        "Найди минимум x^2 - 4x + 5",
        "Найди сумму ряда 1/n^2 от n=1 до ∞"
    ]
    
    for q in test_suite:
        print(f"Запрос: {q}")
        print(f"{main(q)}")
        print("-" * 40)