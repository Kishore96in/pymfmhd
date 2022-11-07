import sympy as sy
import sympy.tensor.tensor

from average import average, do_epsilon_delta

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

#Test epsilon-delta identity
Cartesian = sy.tensor.tensor.TensorIndexType('Cartesian', dim=3)
p, q, r, s, t = sy.tensor.tensor.tensor_indices("p q r s t", Cartesian)
delta = Cartesian.delta
eps = Cartesian.epsilon

def check_tens_eq(expr1, expr2):
	"""
	Canonicalizes the two given tensor expressions and checks equality.
	"""
	diff = expr1 - expr2
	if diff != 0:
		diff = diff.canon_bp().simplify()
	assert diff == 0

check_tens_eq( do_epsilon_delta( eps(r,p,q) * eps(-r, s, t), eps, delta ),
	delta(p,s)*delta(q,t) - delta(p,t)*delta(q,s),
	)
check_tens_eq( do_epsilon_delta( eps(r,p,q) * eps(s, -r, t), eps, delta ),
	- delta(p,s)*delta(q,t) + delta(p,t)*delta(q,s),
	)
check_tens_eq( do_epsilon_delta( eps(r,p,q) * eps(-r, -p, t), eps, delta ).contract_delta(delta),
	2*delta(q,t),
	)
check_tens_eq( do_epsilon_delta( eps(r,p,q) * eps(-r, -p, -q), eps, delta ).contract_delta(delta),
	6,
	)

