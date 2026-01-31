'''
math_intent_calculator
Полный математический интерпретатор с улучшенным парсингом и логикой обработки
'''
import re
import difflib
import time
import multiprocessing
from multiprocessing import Process, Queue
import mpmath
import sympy
from sympy import (
    sympify, N, oo, integrate, diff, limit, series, sqrt, sin, cos, tan, exp, 
    log, pi, E, symbols, solveset, simplify, Eq, solve, Derivative, Integral, 
    Sum, product, lambdify
)
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

# Глобальные константы для сообщений об ошибках
ERRORS = {
    'intent_not_recognized': 'Не удалось распознать математическую задачу. Уточните запрос.',
    'computation_error': 'Ошибка при вычислении выражения.',
    'no_expression_found': 'В запросе не найдено математического выражения.',
    'intent_recognized_prefix': 'Распознана задача: ',
    'ambiguous_request': 'Запрос содержит несколько возможных интерпретаций.',
    'too_vague': 'Запрос слишком общий.',
    'multiple_expressions_found': 'Найдено несколько выражений.',
    'parameters_extracted': 'Извлечены параметры: ',
    'confidence_low': 'Распознавание с низкой уверенностью.',
    'timeout_error': 'Превышено время вычисления.',
    'division_by_zero': 'Ошибка: деление на ноль.',
    'overflow_error': 'Ошибка: переполнение.',
    'invalid_operation': 'Ошибка: недопустимая операция.',
}

MAX_CALCULATION_TIME = 5.0
MAX_INPUT_LENGTH = 2000
mpmath.mp.dps = 50
transformations = standard_transformations + (implicit_multiplication_application, convert_xor)

def worker_process(expression, intent_info, text_raw, queue):
    try:
        result = calculate_expression(expression, intent_info, text_raw)
        queue.put(('success', result))
    except Exception as e:
        queue.put(('error', str(e)))

def calculate_with_timeout(expression, intent_info, text_raw, timeout):
    if not expression:
        return None
    queue = Queue()
    p = Process(target=worker_process, args=(expression, intent_info, text_raw, queue))
    p.start()
    p.join(timeout)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return ERRORS['timeout_error']
    
    if not queue.empty():
        status, result = queue.get()
        if status == 'success':
            return format_result(result)
        else:
            return f"{ERRORS['computation_error']}: {result}"
    return ERRORS['computation_error']

def calculate_expression(expression, intent_info, text_raw):
    try:
        expr_clean = clean_expression(expression)
        category = intent_info.get('category', '') if intent_info else ''
        desc = intent_info.get('description', '').lower() if intent_info else ''

        # 1. ОБРАБОТКА УРАВНЕНИЙ
        if '=' in expr_clean:
            parts = expr_clean.split('=')
            left = parse_expr(parts[0], transformations=transformations)
            right = parse_expr(parts[1], transformations=transformations)
            equation = Eq(left, right)
            vars = list(equation.free_symbols)
            if vars:
                return solve(equation, vars)

        # Парсим основное выражение
        expr = parse_expr(expr_clean, transformations=transformations)
        vars = list(expr.free_symbols)
        x = vars[0] if vars else symbols('x')

        # 2. МАТЕМАТИЧЕСКИЙ АНАЛИЗ (CALCULUS)
        if category == 'calculus':
            if 'производная' in desc or 'дифференцировать' in desc:
                return diff(expr, x)
            
            if 'интеграл' in desc or 'интегрировать' in desc:
                # Поиск пределов интегрирования в исходном тексте
                limits_found = re.findall(r'([\d\.\w\+\*\/]+)\s+до\s+([\d\.\w\+\*\/]+)', text_raw.lower())
                if limits_found:
                    low = parse_expr(limits_found[0][0].replace('π', 'pi'), transformations=transformations)
                    high = parse_expr(limits_found[0][1].replace('π', 'pi'), transformations=transformations)
                    return integrate(expr, (x, low, high))
                return integrate(expr, x)

            if 'предел' in desc or 'лимит' in desc:
                point_match = re.search(r'(?:x|→|->)\s*([\d\.\w]+)', text_raw.lower())
                point = parse_expr(point_match.group(1)) if point_match else 0
                return limit(expr, x, point)

        # 3. ОПТИМИЗАЦИЯ
        if category == 'optimization':
            derivative = diff(expr, x)
            critical_points = solve(derivative, x)
            if critical_points:
                # Для простоты возвращаем значение в первой критической точке
                return expr.subs(x, critical_points[0])

        # 4. СУММЫ
        if 'сумма' in desc and 'oo' in expr_clean:
            n = symbols('n') if 'n' in expr_clean else x
            return Sum(expr, (n, 1, oo)).doit()

        # 5. УПРОЩЕНИЕ (Default)
        res = simplify(expr)
        if res.is_number:
            return N(res, 50)
        return res

    except Exception as e:
        raise ValueError(str(e))

def clean_expression(expression):
    if not expression: return ""
    expr = expression.replace('^', '**').replace('→', '->').replace('π', 'pi').replace('∞', 'oo')
    # Обработка процентов: "15% от 200" -> "(15/100)*200"
    expr = re.sub(r'(\d+(?:\.\d+)?)\s*%\s*(?:от|of|)\s*(\d+(?:\.\d+)?)', r'(\1/100)*\2', expr)
    # Удаление лишних слов
    words_to_remove = ['уравнение', 'найди', 'реши', 'вычисли', 'упрости', 'функции']
    for w in words_to_remove:
        expr = re.sub(r'\b' + w + r'\b', '', expr, flags=re.IGNORECASE)
    return expr.strip()

def extract_math_expression(text):
    # LaTeX
    for p in [r'\$\$(.*?)\$\$', r'\$(.*?)\$', r'\\\[(.*?)\\\]']:
        m = re.search(p, text)
        if m: return m.group(1).strip()
    
    # Регулярка для уравнений и выражений (исключая кириллицу)
    clean_text = re.sub(r'[а-яА-ЯёЁ]+', ' ', text)
    # Ищем блоки с цифрами, операторами и латиницей
    candidates = re.findall(r'[a-zA-Z0-9\s\+\-\*/\^\(\)\.!=%]+', clean_text)
    if candidates:
        filtered = [c.strip() for c in candidates if len(c.strip()) > 1 and any(char.isdigit() or char.isalpha() for char in c)]
        if filtered:
            return max(filtered, key=len)
    return None

def format_result(result):
    if result is None: return ""
    if isinstance(result, (list, tuple, set)):
        return ", ".join([f"x = {format_result(r)}" for r in result])
    if hasattr(result, 'evalf'):
        try:
            val = result.evalf(20)
            if val.is_integer: return str(int(val))
            res_str = str(val).rstrip('0').rstrip('.')
            return res_str
        except: pass
    return str(result)

def lemmatize_words(words, synonym_map):
    lemmas = []
    for word in words:
        word_l = word.lower()
        if word_l in synonym_map: lemmas.append(synonym_map[word_l])
        else:
            matches = difflib.get_close_matches(word_l, synonym_map.keys(), n=1, cutoff=0.8)
            lemmas.append(synonym_map[matches[0]] if matches else word_l)
    return lemmas

def recognize_intent(lemmas, categories):
    scores = {cat: sum(1 for l in lemmas if l in keys) for cat, keys in categories.items()}
    best_cat = max(scores.items(), key=lambda x: x[1])
    if best_cat[1] > 0:
        names = {'optimization': 'оптимизация', 'calculus': 'мат. анализ', 'algebra': 'алгебра', 'geometry': 'геометрия'}
        return {'recognized': True, 'category': best_cat[0], 'description': names.get(best_cat[0], best_cat[0])}
    return {'recognized': False, 'description': ''}

def main(text):
    if not hasattr(main, 'initialized'):
        main.synonym_map = {
            'реши': 'решить', 'вычисли': 'вычислить', 'упрости': 'упростить', 'производная': 'производная',
            'интеграл': 'интеграл', 'предел': 'предел', 'минимум': 'минимум', 'максимум': 'максимум',
            'площадь': 'площадь', 'объем': 'объем', 'круг': 'круг', 'сфера': 'сфера', 'сумма': 'сумма'
        }
        main.categories = {
            'optimization': ['минимум', 'максимум', 'оптимизировать'],
            'calculus': ['производная', 'интеграл', 'предел', 'дифференцировать'],
            'algebra': ['уравнение', 'упростить', 'корень'],
            'geometry': ['площадь', 'объем', 'радиус', 'круг', 'сфера']
        }
        main.initialized = True

    if not text: return ERRORS['intent_not_recognized']
    
    math_expr = extract_math_expression(text)
    words = re.findall(r'[а-яa-z0-9]+', text.lower())
    lemmas = lemmatize_words(words, main.synonym_map)
    intent = recognize_intent(lemmas, main.categories)

    # Геометрический костыль из оригинала
    if (not math_expr or len(math_expr) < 2) and intent['category'] == 'geometry':
        nums = re.findall(r'\d+', text)
        if nums:
            r = nums[0]
            if 'площадь' in lemmas: math_expr = f"pi * {r}**2"
            if 'объем' in lemmas: math_expr = f"(4/3) * pi * {r}**3"

    if math_expr:
        res = calculate_with_timeout(math_expr, intent, text, MAX_CALCULATION_TIME)
        prefix = f"{ERRORS['intent_recognized_prefix']}{intent['description']}\n" if intent['recognized'] else ""
        return f"{prefix}Результат: {res}"
    
    return ERRORS['intent_not_recognized']

if __name__ == "__main__":
    test_cases = [
        ("Посчитай 2+2*2", "6"),
        ("Сколько будет 15% от 200?", "30"),
        ("Реши уравнение x^2 - 5x + 6 = 0", "x = 2, x = 3"),
        ("Упрости выражение (a+b)^2 - (a-b)^2", "4*a*b"),
        ("Найди производную sin(x^2)", "2*x*cos(x**2)"),
        ("Вычисли интеграл от 0 до pi от sin(x) dx", "2"),
        ("Найди предел sin(x)/x при x→0", "1"),
        ("Найди минимум x^2 - 4x + 5", "1"),
        ("Найди сумму ряда 1/n^2 от n=1 до ∞", "pi**2/6")
    ]
    for q, exp in test_cases:
        print(f"Запрос: {q}\nОжидалось: {exp}\n{main(q)}\n{'-'*30}")