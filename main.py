"""简单安全的 Python 计算器：支持交互式 REPL 和命令行表达式求值。"""

import ast
import operator as _operator
import math
import sys


_OPS = {
    ast.Add: _operator.add,
    ast.Sub: _operator.sub,
    ast.Mult: _operator.mul,
    ast.Div: _operator.truediv,
    ast.Mod: _operator.mod,
    ast.Pow: _operator.pow,
    ast.FloorDiv: _operator.floordiv,
}

_UNARY_OPS = {
    ast.UAdd: lambda x: +x,
    ast.USub: lambda x: -x,
}

# 允许的常量
_NAMES = {
    'pi': math.pi,
    'e': math.e,
}

# 允许的函数（从 math 中挑选常用的）
_FUNCS = {name: getattr(math, name) for name in (
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sqrt', 'log', 'log10', 'exp', 'pow', 'fabs'
)}


def safe_eval(expr: str):
    """安全地计算表达式，只允许受控的 AST 节点、常量和函数。"""
    node = ast.parse(expr, mode='eval')

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return n.value
            raise ValueError('Unsupported constant type')
        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)
            op = type(n.op)
            if op in _OPS:
                return _OPS[op](left, right)
            raise ValueError('Unsupported binary operator')
        if isinstance(n, ast.UnaryOp):
            operand = _eval(n.operand)
            op = type(n.op)
            if op in _UNARY_OPS:
                return _UNARY_OPS[op](operand)
            raise ValueError('Unsupported unary operator')
        if isinstance(n, ast.Name):
            if n.id in _NAMES:
                return _NAMES[n.id]
            raise ValueError(f'Unknown name: {n.id}')
        if isinstance(n, ast.Call):
            # 仅允许直接的函数名调用，不允许属性访问或复杂表达式作为函数
            if isinstance(n.func, ast.Name) and n.func.id in _FUNCS:
                func = _FUNCS[n.func.id]
                args = [_eval(a) for a in n.args]
                return func(*args)
            raise ValueError('Unsupported function or call')
        if isinstance(n, ast.Tuple):
            return tuple(_eval(elt) for elt in n.elts)
        raise ValueError(f'Unsupported expression: {type(n).__name__}')

    return _eval(node)


def repl():
    print('简易计算器，输入表达式回车计算。输入 `exit` 或 `quit` 退出。支持：+ - * / % ** 括号，小数，以及 math 函数（sin, cos, sqrt 等）。')
    try:
        while True:
            s = input('>>> ').strip()
            if not s:
                continue
            if s.lower() in ('exit', 'quit'):
                break
            try:
                result = safe_eval(s)
                print(result)
            except Exception as e:
                print('Error:', e)
    except (EOFError, KeyboardInterrupt):
        print()  # 优雅退出


def main():
    if len(sys.argv) > 1:
        expr = ' '.join(sys.argv[1:])
        try:
            print(safe_eval(expr))
        except Exception as e:
            print('Error:', e)
            sys.exit(1)
    else:
        repl()


if __name__ == '__main__':
    main()