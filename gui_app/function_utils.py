"""
Утилиты для создания и регистрации функций
"""
import numpy as np
import functools
from Population_Algorithm import TestFunctions


def make_safe_function(formula_str: str):
    """Создает безопасную функцию из строки формулы"""
    import math

    def prod(iterable):
        return functools.reduce(lambda a, b: a * b, iterable, 1)

    namespace = {
        'np': np,
        'math': math,
        'functools': functools,
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'exp': np.exp,
        'log': np.log,
        'log10': np.log10,
        'sqrt': np.sqrt,
        'abs': abs,
        'len': len,
        'sum': sum,
        'range': range,
        'prod': prod,
        'pi': np.pi,
        'e': np.e,
    }

    exec(f"""
def func(x):
    x = np.array(x, dtype=float)
    return float({formula_str})
""", namespace)
    return namespace['func']


def register_custom_function(formula_str, bounds, name="custom"):
    """Регистрирует пользовательскую функцию в TestFunctions"""
    func = make_safe_function(formula_str)
    TestFunctions._registry[name] = {
        "func": func,
        "bounds": bounds,
        "global_min": 0.0
    }
    return name