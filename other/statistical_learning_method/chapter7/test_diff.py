from sympy import *

# x = Symbol("x")
# y = diff(x**3+x, x)
# print(y)
# result = y.subs('x', 1)
# print(result)
if __name__=="__main__":

    x, y = symbols('x, y')

    z = x ** 2 + y ** 2 + x * y + 2
    print(z)
    result = z.subs({x: 1, y: 2})  # 用数值分别对x、y进行替换
    print(result)

    dx = diff(z, x)  # 对x求偏导
    print(dx)
    result = dx.subs({x: 1, y: 2})
    print(result)

    dy = diff(z, y)  # 对y求偏导
    print(dy)
    result = dy.subs({x: 1, y: 2})
    print(result)
