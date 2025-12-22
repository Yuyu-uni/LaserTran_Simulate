# LaserTran_Simulate
Laser Snow Scattering Simulation Related Papers, Using Python Reproduction Code Repository

# TEST:Python 简易计算器

这是一个使用安全 AST 求值实现的简易命令行计算器，支持交互式 REPL 与直接在命令行传入表达式求值。

特性：

- 安全解析表达式（不直接使用 eval）
- 支持基本算术：+ - * / % ** //
- 支持括号、小数与负数
- 支持常用数学函数：sin, cos, tan, asin, acos, atan, sqrt, log, log10, exp, pow, fabs
- 常量：pi, e

快速开始

在项目目录运行交互式 REPL：

```bash
python main.py
```

示例（命令行直接计算表达式）：

```bash
python main.py 2+3*4
python main.py "sin(pi/2)"
python main.py "sqrt(2)"
```

注意

- 表达式应只包含受支持的运算、函数与常量；其它使用会给出错误提示。

如果需要我可以：添加更多 math 函数、保存历史、或做一个简单的 GUI。欢迎告诉我你的需求。

