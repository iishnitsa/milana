import re
import sympy
from sympy import (
    N, oo, integrate, diff, limit, sin, cos, pi, symbols, Eq, solve, simplify, Sum
)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
)

# --- КОНФИГУРАЦИЯ ---
transformations = standard_transformations + (implicit_multiplication_application, convert_xor)

class MathEngine:
    def __init__(self):
        # Реестр обработчиков: список функций, которые будут пробовать решить задачу
        self.handlers = [
            self.handle_percentages,
            self.handle_calculus,  # Интегралы, пределы, производные
            self.handle_series,    # Ряды
            self.handle_equations, # Уравнения
            self.handle_general    # Упрощение и арифметика (всегда последний)
        ]

    def format_result(self, result):
        if result is None: return None
        if isinstance(result, (list, tuple, set)):
            if not result: return "Нет решений"
            return ", ".join([f"x = {str(simplify(r)).replace('**', '^')}" for r in result])
        try:
            if hasattr(result, 'is_number') and result.is_number:
                if abs(result - round(float(result))) < 1e-10:
                    return str(int(round(float(result))))
                return str(N(result, 7)).rstrip('0').rstrip('.')
        except: pass
        return str(result).replace('**', '^').replace(' ', '')

    def clean_text(self, text):
        """Базовая чистка для всех обработчиков."""
        t = text.lower()
        t = t.replace('^', '**').replace('→', '->').replace('∞', 'oo').replace('π', 'pi').replace('пи', 'pi')
        # Удаляем dx только если это не часть переменной (упрощенно)
        t = re.sub(r'\bdx\b', '', t)
        return t

    # --- ОБРАБОТЧИКИ (HANDLERS) ---

    def handle_percentages(self, text):
        match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:от|of)\s*(\d+(?:\.\d+)?)', text)
        if match:
            return (float(match.group(1)) / 100) * float(match.group(2))
        return None

    def handle_calculus(self, text):
        # ПРЕДЕЛ
        if any(w in text for w in ['предел', 'лимит']):
            point_match = re.search(r'(?:->|→|к|при|стремится)\s*([a-z]\s*)?(?:->|→|к|)?\s*([\d\.\w\-oo]+)', text)
            p_val = parse_expr(point_match.group(2)) if point_match else 0
            body = re.split(r'при|стремится|->|→|к', text)[0]
            body = re.sub(r'[а-яА-ЯёЁ]+', ' ', body).strip()
            expr = parse_expr(body, transformations=transformations)
            x = list(expr.free_symbols)[0] if expr.free_symbols else symbols('x')
            return limit(expr, x, p_val)
        
        # ИНТЕГРАЛ
        if 'интеграл' in text:
            bounds = re.findall(r'(\d+|pi|oo)\s+до\s+(\d+|pi|oo)', text)
            body = text
            for w in ['интеграл', 'вычисли', 'найди', 'от', 'до']:
                body = re.sub(rf'\b{w}\b', ' ', body)
            if bounds:
                for b in bounds[0]: body = body.replace(b, ' ')
            body = re.sub(r'[а-яА-ЯёЁ]+', ' ', body).strip()
            expr = parse_expr(body, transformations=transformations)
            x = list(expr.free_symbols)[0] if expr.free_symbols else symbols('x')
            if bounds:
                return integrate(expr, (x, parse_expr(bounds[0][0]), parse_expr(bounds[0][1])))
            return integrate(expr, x)

        # ПРОИЗВОДНАЯ
        if 'производн' in text:
            body = re.sub(r'[а-яА-ЯёЁ]+', ' ', text).strip()
            expr = parse_expr(body, transformations=transformations)
            x = list(expr.free_symbols)[0] if expr.free_symbols else symbols('x')
            return diff(expr, x)
            
        return None

    def handle_series(self, text):
        if 'ряд' in text or 'сумма' in text:
            body = re.split(r' от | для | n=', text)[0]
            body = re.sub(r'[а-яА-ЯёЁ]+', ' ', body).strip()
            expr = parse_expr(body, transformations=transformations)
            n = symbols('n') if 'n' in str(expr) else (list(expr.free_symbols)[0] if expr.free_symbols else symbols('n'))
            return Sum(expr, (n, 1, oo)).doit()
        return None

    def handle_equations(self, text):
        if '=' in text:
            body = re.sub(r'[а-яА-ЯёЁ]+', ' ', text).strip()
            parts = body.split('=')
            left = parse_expr(parts[0], transformations=transformations)
            right = parse_expr(parts[1], transformations=transformations) if parts[1].strip() else 0
            return solve(Eq(left, right))
        return None

    def handle_general(self, text):
        # Последний шанс: просто пытаемся распарсить что осталось
        body = re.sub(r'[а-яА-ЯёЁ]+', ' ', text).strip()
        if not body: return None
        expr = parse_expr(body, transformations=transformations)
        
        # Если есть слова минимум/максимум
        if 'минимум' in text or 'максимум' in text:
            x = list(expr.free_symbols)[0] if expr.free_symbols else symbols('x')
            df = diff(expr, x)
            pts = solve(df, x)
            real_pts = [p for p in pts if p.is_real]
            vals = [expr.subs(x, p) for p in real_pts]
            return min(vals) if 'минимум' in text else max(vals)
            
        return simplify(expr)

    def run(self, query):
        clean_query = self.clean_text(query)
        # Пробуем каждый обработчик по очереди
        for handler in self.handlers:
            try:
                result = handler(clean_query)
                if result is not None:
                    return self.format_result(result)
            except:
                continue
        return "Не удалось решить задачу"

# --- LLM INTERFACE ---
def process_math(query):
    engine = MathEngine()
    # Определяем тип для красоты
    intent = "математика"
    if any(x in query.lower() for x in ['интеграл', 'предел', 'производн']): intent = "мат. анализ"
    elif any(x in query.lower() for x in ['уравнение', 'сумма', 'ряд']): intent = "алгебра"
    
    res = engine.run(query)
    return f"Распознана задача: {intent}\nРезультат: {res}"

# --- ТЕСТЫ ---
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
        print(f"Запрос: {q}\n{process_math(q)}\n{'-'*40}")