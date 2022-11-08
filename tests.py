import sympy as sy
import sympy.tensor.tensor

from average import average
from tensor import do_epsilon_delta, do_angular_integral, partialdiff

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
p, q, r, s, t, u, w, i, j = sy.tensor.tensor.tensor_indices("p q r s t u w i j", Cartesian)
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
check_tens_eq( do_epsilon_delta( eps(-r,p,q) * eps(r, -s, t), eps, delta ),
	delta(p,-s)*delta(q,t) - delta(p,t)*delta(q,-s),
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

#Test angular integrals
K = sy.tensor.tensor.TensorHead("K", [Cartesian])

check_tens_eq(
	do_angular_integral( K(p)*K(q), K, delta),
	4*sy.pi/3 * K(r)*K(-r) * delta(p,q)
	)
check_tens_eq(
	do_angular_integral( K(p)*K(q)*K(r)*K(s), K, delta),
	4*sy.pi/15 * K(t)*K(-t) * K(u)*K(-u) * ( delta(p,q) * delta(r,s) + delta(p,r) * delta(q,s) + delta(p,s) * delta(r,q) )
	)
check_tens_eq(
	do_angular_integral( K(p)*K(q)*K(r)*K(s)*K(t)*K(u), K, delta),
	4*sy.pi/105 * K(i)*K(-i) * K(j)*K(-j) * K(w)*K(-w) * (
		delta(p,q) * delta(r,s) * delta(t,u)
		+ delta(p,q) * delta(r,t) * delta(s,u)
		+ delta(p,q) * delta(r,u) * delta(t,s)
		+ delta(p,r) * delta(q,s) * delta(t,u)
		+ delta(p,r) * delta(q,t) * delta(s,u)
		+ delta(p,r) * delta(q,u) * delta(t,s)
		+ delta(p,s) * delta(q,r) * delta(t,u)
		+ delta(p,s) * delta(q,t) * delta(r,u)
		+ delta(p,s) * delta(q,u) * delta(r,t)
		+ delta(p,t) * delta(q,r) * delta(s,u)
		+ delta(p,t) * delta(q,s) * delta(r,u)
		+ delta(p,t) * delta(q,u) * delta(r,s)
		+ delta(p,u) * delta(q,r) * delta(s,t)
		+ delta(p,u) * delta(q,s) * delta(r,t)
		+ delta(p,u) * delta(q,t) * delta(r,s)
		)
	)
check_tens_eq(
	do_angular_integral( K(p)*K(-p), K, delta),
	4*sy.pi/3 * K(r)*K(-r) * delta(p,-p)
	)

#Test partial derivatives
f = sympy.symbols('f', cls=sympy.Function)
k = sympy.symbols("K") #'amplitude' of K

check_tens_eq(
	partialdiff( K(p), K(q), Cartesian ),
	delta(p,-q)
	)
check_tens_eq(
	partialdiff( K(p)*K(-p), K(q), Cartesian ),
	2*K(-q)
	)
check_tens_eq(
	partialdiff( K(p)*K(r), K(q), Cartesian ),
	K(p) * delta(r,-q) + K(r) * delta(p,-q)
	)
check_tens_eq(
	partialdiff( K(p) * K(q) * f(k) , K(r), indextype=Cartesian, ampl=k ),
	f(k)*K(p)*delta(q, -r) + f(k)*K(q)*delta(p, -r) + 1/k * sympy.Derivative(f(k), k)*K(p)*K(q)*K(-r)
	)
