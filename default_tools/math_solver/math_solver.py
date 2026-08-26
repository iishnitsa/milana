'''
math_solver
Computes math expressions, solves equations, systems, limits, matrices, simplifies symbolic expressions. Supports ^, **, ². Input expression as text.
Math Calculator
Solves equations, integrals, derivatives, limits, matrices, simplifies formulas and computes arithmetic with high precision.
'''

import re
import sympy as sp
from sympy.parsing.latex import parse_latex
from sympy.ntheory import factorint
from sympy.core.sympify import SympifyError
import mpmath
import multiprocessing
import queue
from cross_gpt import let_log

# Set high precision for numerical calculations
mpmath.mp.dps = 50


def _preprocess(raw: str) -> str:
    """Replace Unicode powers and ^ with **."""
    superscript_map = {
        '²': '**2', '³': '**3', '⁴': '**4', '⁵': '**5',
        '⁶': '**6', '⁷': '**7', '⁸': '**8', '⁹': '**9', '⁰': '**0'
    }
    for uni, py in superscript_map.items():
        raw = raw.replace(uni, py)
    raw = raw.replace('^', '**')
    raw = re.sub(r'\s+', ' ', raw).strip()
    return raw


def _extract_math(raw_text: str) -> str:
    """
    Extracts mathematical expression, ignoring any words (multilingual).
    Returns empty string if nothing found.
    """
    # If there are LaTeX dollar signs, take content between them
    dollar_match = re.search(r'\$(.+?)\$', raw_text)
    if dollar_match:
        raw_text = dollar_match.group(1)

    result = []
    i = 0
    n = len(raw_text)

    # Long math tokens (longer priority)
    long_tokens = [
        'integrate', 'diff', 'limit', 'sum', 'prod',
        'Matrix', 'det', 'inv', 'transpose', 'eigenvals',
        'factor', 'expand', 'simplify', 'collect',
        'gcd', 'lcm', 'factorial', 'binomial', 'factorint',
        'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
        'asin', 'acos', 'atan', 'acot', 'asec', 'acsc',
        'sinh', 'cosh', 'tanh', 'coth',
        'log', 'ln', 'lg', 'exp', 'sqrt', 'cbrt', 'abs',
        'pi', 'e', 'I', 'oo'
    ]

    while i < n:
        char = raw_text[i]
        matched = False

        # Check long tokens
        for token in long_tokens:
            if raw_text[i:].lower().startswith(token.lower()):
                result.append(raw_text[i:i + len(token)])
                i += len(token)
                matched = True
                break
        if matched:
            continue

        # Handle LaTeX commands (starting with \)
        if char == '\\':
            j = i + 1
            while j < n and raw_text[j].isalpha():
                j += 1
            result.append(raw_text[i:j])
            i = j
            continue

        # Allowed single characters
        if re.match(r'[\d\+\-\*\/\(\)\=\[\]\{\}\,\.\;\:]', char):
            result.append(char)
        elif char.isalpha():
            # Allow only variables x, y, z and Greek letters
            if char.lower() in ['x', 'y', 'z'] or (0x0370 <= ord(char) <= 0x03FF):
                result.append(char)
            # otherwise ignore (it's a word in any language)
        elif char.isspace():
            result.append(' ')
        i += 1

    expr = ''.join(result)
    return _preprocess(expr)


def _process_math(text: str) -> dict:
    """
    Main math processing logic (called in child process).
    Returns dict:
        success: bool
        error_code: str (if success=False) — 'no_math', 'syntax', 'computation'
        detail: str (additional error info)
        result, latex, human, type (if success=True)
    """
    raw_math = _extract_math(text)
    if not raw_math:
        return {"success": False, "error_code": "no_math"}

    safe_locals = {
        'Matrix': sp.Matrix,
        'det': sp.det,
        'inv': sp.inv,
        'transpose': sp.transpose,
        'eigenvals': sp.eigenvals,
        'limit': sp.limit,
        'factor': sp.factor,
        'expand': sp.expand,
        'simplify': sp.simplify,
        'collect': sp.collect,
        'gcd': sp.gcd,
        'lcm': sp.lcm,
        'factorial': sp.factorial,
        'binomial': sp.binomial,
        'factorint': factorint,
        'I': sp.I,
        'oo': sp.oo,
        'pi': sp.pi,
        'e': sp.E
    }

    try:
        # --- Systems of equations ---
        if '=' in raw_math and (',' in raw_math or ';' in raw_math):
            parts = re.split(r'[,;]\s*', raw_math)
            equations = [p.strip() for p in parts if '=' in p]
            if len(equations) > 1:
                solutions = sp.solve(equations)
                if solutions:
                    return {"success": True, "result": solutions, "latex": sp.latex(solutions), "type": "system"}
                # Numeric solving fallback
                try:
                    vars = list(sp.sympify(equations[0]).free_symbols)
                    num_sol = sp.nsolve(equations, vars, [0] * len(vars))
                    return {"success": True, "result": num_sol, "latex": sp.latex(num_sol), "type": "system_numeric"}
                except Exception:
                    pass

        # --- Single equation ---
        if '=' in raw_math:
            solutions = sp.solve(raw_math)
            if solutions:
                return {"success": True, "result": solutions, "latex": sp.latex(solutions), "type": "equation"}

        # --- Limits ---
        if 'limit(' in raw_math:
            try:
                expr = sp.sympify(raw_math, locals=safe_locals)
                if isinstance(expr, sp.Limit):
                    result = expr.doit()
                else:
                    result = expr
                return {"success": True, "result": result, "latex": sp.latex(result), "type": "limit"}
            except Exception:
                # manual fallback
                match = re.search(r'limit\s*\((.+?)\)', raw_math)
                if match:
                    inner = match.group(1)
                    args = [a.strip() for a in inner.split(',')]
                    if len(args) == 3:
                        try:
                            res = sp.limit(sp.sympify(args[0]), sp.sympify(args[1]), sp.sympify(args[2]))
                            return {"success": True, "result": res, "latex": sp.latex(res), "type": "limit"}
                        except Exception:
                            pass

        # --- Matrices ---
        if 'Matrix' in raw_math or ('[' in raw_math and ']' in raw_math and ',' in raw_math):
            try:
                expr = sp.sympify(raw_math, locals=safe_locals)
                if isinstance(expr, sp.MatrixBase):
                    return {"success": True, "result": expr, "latex": sp.latex(expr), "type": "matrix"}
                else:
                    return {"success": True, "result": expr, "latex": sp.latex(expr), "type": "matrix_expr"}
            except Exception:
                pass

        # --- Number theory (factorization) ---
        try:
            test_val = sp.sympify(raw_math)
            if test_val.is_Integer and test_val > 1:
                factors = factorint(test_val)
                return {"success": True, "result": factors, "latex": sp.latex(factors), "type": "factorization",
                        "human": str(factors)}
        except Exception:
            pass

        # --- Symbolic algebra / arithmetic ---
        expr = sp.sympify(raw_math, locals=safe_locals)

        if expr.free_symbols:
            simplified = sp.simplify(expr)
            latex_res = sp.latex(simplified)
            numeric_val = None
            try:
                numeric_val = simplified.evalf(50)
            except Exception:
                pass
            return {
                "success": True,
                "result": simplified,
                "latex": latex_res,
                "type": "symbolic",
                "human": str(numeric_val) if numeric_val else None
            }
        else:
            val = expr.evalf(50)
            val_str = f"{val:.50f}".rstrip('0').rstrip('.')
            return {
                "success": True,
                "result": float(val) if abs(val) < 1e15 else val,
                "latex": sp.latex(val),
                "type": "numeric",
                "human": val_str
            }

    except (SympifyError, SyntaxError, TypeError, ZeroDivisionError, ValueError) as e:
        return {"success": False, "error_code": "syntax", "detail": str(e)}
    except Exception as e:
        return {"success": False, "error_code": "computation", "detail": str(e)}


def _run_with_timeout(text: str, timeout: float):
    """
    Runs _process_math in a separate process with a timeout.
    Returns the result dict or a timeout error dict.
    """
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=lambda: q.put(_process_math(text)))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return {"success": False, "error_code": "timeout"}
    try:
        return q.get_nowait()
    except queue.Empty:
        return {"success": False, "error_code": "timeout"}


def main(text):
    """Entry point for external call (LLM invokes this function)."""
    if not hasattr(main, 'attr_names'):
        # Initialize attributes (default English values)
        main.attr_names = (
            'error_no_math', 'error_syntax', 'error_computation', 'error_timeout',
            'latex_prefix', 'numeric_prefix', 'result_prefix'
        )
        main.error_no_math = 'No math expression found.'
        main.error_syntax = 'Math syntax error: '
        main.error_computation = 'Computation failed: '
        main.error_timeout = 'Calculation timed out (took too long).'
        main.latex_prefix = 'LaTeX: '
        main.numeric_prefix = 'Numeric: '
        main.result_prefix = 'Result: '
        main.timeout_seconds = 60.0   # default timeout
        return

    let_log('MATH SOLVER CALLED')
    let_log(f'Input text: {text}')

    # Run calculation with timeout (from attribute, can be overridden)
    timeout = getattr(main, 'timeout_seconds', 60.0)
    result = _run_with_timeout(text, timeout)

    if not result['success']:
        error_code = result.get('error_code', 'computation')
        prefix = getattr(main, 'error_' + error_code, 'Error: ')
        detail = result.get('detail', '')
        return f"{prefix}{detail}"

    parts = []
    if result.get('latex'):
        parts.append(f"{main.latex_prefix}{result['latex']}")
    if result.get('human'):
        parts.append(f"{main.numeric_prefix}{result['human']}")
    if not parts:
        parts.append(f"{main.result_prefix}{str(result.get('result', ''))}")

    return "\n".join(parts)