import sympy as sy

from average import average

x,y = sy.symbols("x y")

#Test getting free symbols
assert average(x, set([x])).free_symbols == set()
assert average(y*x, set([x])).free_symbols == set([y])

#Test simplification
assert average(sy.sin(y)*sy.cos(y), set([])).simplify() == sy.sin(2*y)/2
assert average(sy.sin(y), set([x])).simplify() == sy.sin(y)
assert average(sy.sin(y)*sy.cos(y), set([x])).simplify() == sy.sin(2*y)/2
assert average(sy.sin(y)*sy.cos(x), set([x])).simplify() == sy.sin(y)*average(sy.cos(x), set([x]))
assert average(sy.sin(x) + sy.cos(x), set([x])).simplify() == average(sy.sin(x), set([x])) + average(sy.cos(x), set([x]))
assert average(sy.sin(y) + sy.cos(x), set([x])).simplify() == sy.sin(y) + average(sy.cos(x), set([x]))
