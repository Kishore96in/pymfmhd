import sympy as sy
import sympy.tensor.tensor

from average import average
from tensor import do_epsilon_delta, do_angular_integral, partialdiff, dive_matcher, mul_matcher

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
	try:
		diff = expr1 - expr2
		if diff != 0:
			diff = diff.canon_bp().simplify()
		assert diff == 0
	except:
		print("Something went wrong.")
		print(f"{expr1 = }")
		print(f"{expr2 = }")
		raise

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
V = sy.tensor.tensor.TensorHead("V", [Cartesian])

check_tens_eq(
	do_angular_integral( K(p)*K(q), K),
	4*sy.pi/3 * K(r)*K(-r) * delta(p,q)
	)
check_tens_eq(
	do_angular_integral( K(p)*K(q) + K(p)*V(q), K),
	4*sy.pi/3 * K(r)*K(-r) * delta(p,q)
	)
check_tens_eq(
	do_angular_integral( K(p)*K(q)*K(r)*K(s), K),
	4*sy.pi/15 * K(t)*K(-t) * K(u)*K(-u) * ( delta(p,q) * delta(r,s) + delta(p,r) * delta(q,s) + delta(p,s) * delta(r,q) )
	)
check_tens_eq(
	do_angular_integral( K(p)*K(q)*K(r)*K(s)*K(t)*K(u), K),
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
	do_angular_integral( K(p)*K(-p), K),
	4*sy.pi/3 * K(r)*K(-r) * delta(p,-p)
	)

#Test partial derivatives
f, g = sympy.symbols('f g', cls=sympy.Function)
k = sympy.symbols("K") #'amplitude' of K

check_tens_eq(
	partialdiff( K(p), K(q) ),
	delta(p,-q)
	)
check_tens_eq(
	partialdiff( K(p)*K(-p), K(q) ),
	2*K(-q)
	)
check_tens_eq(
	partialdiff( K(p)*K(r), K(q) ),
	K(p) * delta(r,-q) + K(r) * delta(p,-q)
	)
check_tens_eq(
	partialdiff( K(p) * K(q) * f(k) , K(r), ampl=k ),
	f(k)*K(p)*delta(q, -r) + f(k)*K(q)*delta(p, -r) + 1/k * sympy.Derivative(f(k), k)*K(p)*K(q)*K(-r)
	)
check_tens_eq(
	partialdiff( K(p) * K(q) * f(k) + Cartesian.delta(p,q) * g(k), K(r), ampl=k ),
	f(k)*K(p)*delta(q, -r) + f(k)*K(q)*delta(p, -r) + 1/k * sympy.Derivative(f(k), k)*K(p)*K(q)*K(-r) + 1/k*sympy.Derivative(g(k), k)*K(-r)*delta(p, q)
	)
check_tens_eq(
	partialdiff( K(p) / k , K(q), ampl=k ),
	( delta(p, -q) - K(p)*K(-q)/k**2 )/k
	)
check_tens_eq(
	partialdiff( 1 / k**2 , K(p), ampl=k ),
	-2 * K(-p)/k**4
	)

#Imposing zero-divergence
dive_match = dive_matcher(K, V)

check_tens_eq(
	( - K(q) * K(-p) * V(p) ).replace(dive_match, lambda Expr: 0),
	0
	)
check_tens_eq(
	( - V(q) * K(-p) * V(p) ).replace(dive_match, lambda Expr: 0),
	0
	)
check_tens_eq(
	( K(p) * V(-p) ).replace(dive_match, lambda Expr: 0),
	0
	)
check_tens_eq(
	( - K(p) * K(q) * V(r) + V(s) * K(-s) * K(p) * V(q) * V(r) ).replace(dive_match, lambda Expr: 0),
	- K(p) * K(q) * V(r)
	)

#Check mul_matcher
check_tens_eq(
	( - K(q) * K(-p) * V(p) ).replace( *mul_matcher( K(r)*V(-r), 0 ) ),
	0
	)
check_tens_eq(
	( - V(q) * K(-p) * V(p) ).replace( *mul_matcher( K(r)*V(-r), 0 ) ),
	0
	)
check_tens_eq(
	( - V(s)*V(-s) *  K(r) * K(-r) * V(q) * K(-p) * V(p) ).replace( *mul_matcher( K(r)*V(-r), 0 ) ),
	0
	)
check_tens_eq(
	( K(p) * V(-p) ).replace( *mul_matcher( K(r)*V(-r), 0 ) ),
	0
	)
check_tens_eq(
	( - K(p) * K(q) * V(r) + V(s) * K(-s) * K(p) * V(q) * V(r) ).replace( *mul_matcher( K(r)*V(-r), 0 ) ),
	- K(p) * K(q) * V(r)
	)
check_tens_eq(
	( - K(q) * K(-q) * V(p) + K(p) ).replace( *mul_matcher( K(r)*K(-r), k**2 ) ),
	- k**2 * V(p) + K(p)
	)
check_tens_eq(
	( 2 * K(q) * K(-q) * K(-r) * K(r) * V(p) + K(p) ).replace( *mul_matcher( K(r)*K(-r), k**2 ) ),
	2 * k**4 * V(p) + K(p)
	)
