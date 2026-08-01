"""
Парсер признаков для формул.

Разрешённые элементы:
    Функции:   sin, cos, exp, abs, sum, prod
    Операторы: +, -, *, /, **
    Служебные: x[i], len(x), range(), pi, e, числа
    Генераторы: [expr for i in range(n)]
"""
import ast
import numpy as np


class FunctionParser:
    """Парсер признаков формулы"""

    TRIG = {'sin', 'cos'}
    NESTING_FUNCS = {'sin', 'cos', 'exp', 'abs'}

    def __init__(self, formula_str: str):
        self.raw = formula_str.strip()
        self.tree = ast.parse(self.raw, mode='eval')

    # ═══════════════════ утилиты ═══════════════════

    @staticmethod
    def _get_func_name(node):
        """Имя вызываемой функции из ast.Call."""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id
            if isinstance(node.func, ast.Attribute):
                return node.func.attr
        return ''

    def _collect_unique_indices(self, node):
        """Возвращает множество индексов x[i] в выражении."""
        indices = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Subscript):
                if isinstance(child.value, ast.Name) and child.value.id == 'x':
                    indices.add(ast.dump(child.slice))
        return indices

    def _depends_on_x(self, node):
        """Зависит ли выражение от x (но не через len(x))."""
        for child in ast.walk(node):
            if isinstance(child, ast.Subscript):
                if isinstance(child.value, ast.Name) and child.value.id == 'x':
                    return True
            if isinstance(child, ast.Name) and child.id == 'x':
                if not self._is_arg_of_len(child):
                    return True
        return False

    def _is_arg_of_len(self, name_node):
        """Проверяет, является ли name_node аргументом len()."""
        parents = self._parent_map
        current = name_node
        while current in parents:
            parent = parents[current]
            if (isinstance(parent, ast.Call)
                    and isinstance(parent.func, ast.Name)
                    and parent.func.id == 'len'):
                return True
            current = parent
        return False

    @property
    def _parent_map(self):
        """Ленивое построение карты parent для AST."""
        if not hasattr(self, '_parents_cache'):
            parents = {}
            for node in ast.walk(self.tree):
                for child in ast.iter_child_nodes(node):
                    parents[child] = node
            self._parents_cache = parents
        return self._parents_cache

    def _find_calls(self, root, func_names):
        """Все вызовы функций из func_names, чей аргумент зависит от x."""
        results = []
        for node in ast.walk(root):
            if isinstance(node, ast.Call):
                name = self._get_func_name(node)
                if name in func_names and node.args:
                    if self._depends_on_x(node.args[0]):
                        results.append((name, node.args[0], node))
        return results

    def _try_eval_const(self, node):
        """Пытается вычислить узел как числовую константу."""
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id == 'pi':
                return np.pi
            if node.id == 'e':
                return np.e
        if not self._depends_on_x(node):
            try:
                code = compile(ast.Expression(body=node), '<const>', 'eval')
                val = eval(code, {"__builtins__": {}},
                           {"pi": np.pi, "e": np.e, "len": len})
                return float(val)
            except Exception:
                pass
        return None

    @staticmethod
    def _node_contains(parent, target):
        """Содержит ли поддерево parent узел target (по identity)."""
        for child in ast.walk(parent):
            if child is target:
                return True
        return False

    def _collect_x_index_exprs(self, node):
        """Множество строковых представлений индексов x[...] в узле."""
        indices = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Subscript):
                if isinstance(child.value, ast.Name) and child.value.id == 'x':
                    indices.add(ast.dump(child.slice))
        return indices

    def _is_sum_of_x(self, node):
        """Узел — вызов sum([f(x[i]) ...])."""
        if isinstance(node, ast.Call) and self._get_func_name(node) == 'sum':
            if node.args and self._depends_on_x(node.args[0]):
                return True
        return False

    def _contains_sum_of_x(self, node):
        """Содержит ли поддерево вызов sum(f(x))."""
        for child in ast.walk(node):
            if self._is_sum_of_x(child):
                return True
        return False

    # ═══════════════ 1. is_multimodal ═══════════════

    def _is_multimodal(self):
        """1 если есть sin/cos от выражения, зависящего от x."""
        return 1 if self._find_calls(self.tree, self.TRIG) else 0

    # ═══════════════ 2. is_periodic ═════════════════

    def _is_periodic(self):
        """1 если есть sin/cos с линейным аргументом, НЕ обёрнутый в exp."""
        has_linear_trig = False
        has_nonlinear_trig = False

        for _name, arg, call_node in self._find_calls(self.tree, self.TRIG):
            if self._arg_is_linear(arg) and not self._is_wrapped_in_exp(call_node):
                has_linear_trig = True
            else:
                has_nonlinear_trig = True

        if has_nonlinear_trig:
            return 0

        return 1 if has_linear_trig else 0

    def _arg_is_linear(self, node):
        """Аргумент линеен по x: нет x**n, abs(x) внутри."""
        for child in ast.walk(node):
            if isinstance(child, ast.BinOp) and isinstance(child.op, ast.Pow):
                if self._depends_on_x(child.left):
                    return False
            if isinstance(child, ast.Call):
                name = self._get_func_name(child)
                if name == 'abs' and child.args and self._depends_on_x(child.args[0]):
                    return False
        return True

    def _is_wrapped_in_exp(self, trig_call_node):
        """Проверяет, обёрнут ли trig-вызов в exp."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                name = self._get_func_name(node)
                if name == 'exp' and node is not trig_call_node:
                    if self._node_contains(node, trig_call_node):
                        return True
        return False

    # ═══════════════ 3. is_separable ════════════════

    def _has_coupling(self):
        """1 если переменные связаны (функция несепарабельна)"""
        # R1 — разные индексы в генераторе
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ListComp):
                if len(self._collect_x_index_exprs(node.elt)) >= 2:
                    return 1

        # R2 — prod(f(x))
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call) and self._get_func_name(node) == 'prod':
                if node.args and self._depends_on_x(node.args[0]):
                    return 1

        # R3 — (expr_containing_sum)**k, k ≠ 1
        for node in ast.walk(self.tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                if self._contains_sum_of_x(node.left):
                    exp_val = self._try_eval_const(node.right)
                    if exp_val is not None and exp_val != 1.0:
                        return 1

        # R4 — exp(expr_containing_sum)
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call) and self._get_func_name(node) == 'exp':
                if node.args and self._contains_sum_of_x(node.args[0]):
                    return 1

        return 0

    def _has_product_terms(self):
        """1 если присутствуют мультипликативные связи между РАЗНЫМИ переменными."""
        # P1 — prod(trig(x))
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call) and self._get_func_name(node) == 'prod':
                if node.args and self._depends_on_x(node.args[0]):
                    return 1

        # P2 — BinOp(Mult) с разными переменными
        for node in ast.walk(self.tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                if self._depends_on_x(node.left) and self._depends_on_x(node.right):
                    left_indices = self._collect_unique_indices(node.left)
                    right_indices = self._collect_unique_indices(node.right)

                    if len(left_indices) >= 2 or len(right_indices) >= 2:
                        return 1

                    if left_indices != right_indices:
                        return 1

        # P3 — Pow с разными индексами
        for node in ast.walk(self.tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                exp_val = self._try_eval_const(node.right)
                if exp_val is not None and exp_val >= 2.0:
                    if len(self._collect_unique_indices(node.left)) >= 2:
                        return 1

        # P4 — Pow с sum
        for node in ast.walk(self.tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                exp_val = self._try_eval_const(node.right)
                if exp_val is not None and exp_val >= 2.0:
                    if self._contains_sum_of_x(node.left):
                        return 1

        return 0

    # ═══════════════ 5. trig_nesting_depth ══════════

    def _trig_nesting_depth(self):
        """Максимальная глубина вложенности sin/cos/exp/abs от x."""
        if not self._find_calls(self.tree, self.TRIG):
            return 0
        return self._max_depth(self.tree.body)

    def _max_depth(self, node):
        max_d = 0
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Call):
                name = self._get_func_name(child)
                if name in self.NESTING_FUNCS and child.args:
                    if self._depends_on_x(child.args[0]):
                        max_d = max(max_d, 1 + self._max_depth(child))
                    else:
                        max_d = max(max_d, self._max_depth(child))
                else:
                    max_d = max(max_d, self._max_depth(child))
            else:
                max_d = max(max_d, self._max_depth(child))
        return max_d

    # ═══════════════ 6. trig_frequency ══════════════

    def _trig_frequency(self):
        """Максимальный числовой коэффициент при x в аргументе sin/cos."""
        trig_calls = self._find_calls(self.tree, self.TRIG)
        if not trig_calls:
            return 0.0
        max_freq = 0.0
        for _name, arg, _call in trig_calls:
            freq = self._extract_freq(arg)
            max_freq = max(max_freq, freq)
        return round(max_freq, 2)

    def _extract_freq(self, node):
        """Извлекает коэффициент при x[i] из аргумента trig."""
        if not self._arg_is_linear(node):
            return 1.0
        coeff = self._get_mult_coeff(node)
        return abs(coeff) if coeff is not None and coeff > 0 else 1.0

    def _get_mult_coeff(self, node):
        """Рекурсивно извлекает числовой множитель при x[i]."""
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id == 'x':
                return 1.0

        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Mult):
                lc = self._try_eval_const(node.left)
                rc = self._try_eval_const(node.right)
                if lc is not None and self._depends_on_x(node.right):
                    inner = self._get_mult_coeff(node.right)
                    return lc * inner if inner is not None else lc
                if rc is not None and self._depends_on_x(node.left):
                    inner = self._get_mult_coeff(node.left)
                    return rc * inner if inner is not None else rc

            elif isinstance(node.op, ast.Div):
                rc = self._try_eval_const(node.right)
                if rc is not None and rc != 0 and self._depends_on_x(node.left):
                    inner = self._get_mult_coeff(node.left)
                    return inner / rc if inner is not None else 1.0 / rc

            elif isinstance(node.op, (ast.Add, ast.Sub)):
                lc = (self._get_mult_coeff(node.left)
                      if self._depends_on_x(node.left) else None)
                rc = (self._get_mult_coeff(node.right)
                      if self._depends_on_x(node.right) else None)
                vals = [abs(v) for v in [lc, rc] if v is not None]
                return max(vals) if vals else 1.0

        return 1.0

    # ═══════════════ главный метод ══════════════════

    def parse_all(self):
        coupling = self._has_coupling()
        return {
            'is_multimodal':      self._is_multimodal(),
            'is_periodic':        self._is_periodic(),
            'is_separable':       1 - coupling,
            'has_product_terms':  self._has_product_terms(),
            'trig_nesting_depth': self._trig_nesting_depth(),
            'trig_frequency':     self._trig_frequency(),
        }