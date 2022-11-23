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
wi = sy.Wild("wi")
p_2 = sy.tensor.tensor.WildTensorIndex("p_2", Cartesian, ignore_updown=True)
q_2 = sy.tensor.tensor.WildTensorIndex("q_2", Cartesian, ignore_updown=True)
r_2 = sy.tensor.tensor.WildTensorIndex("r_2", Cartesian, ignore_updown=True)
s_2 = sy.tensor.tensor.WildTensorIndex("s_2", Cartesian, ignore_updown=True)
t_2 = sy.tensor.tensor.WildTensorIndex("t_2", Cartesian, ignore_updown=True)
u_2 = sy.tensor.tensor.WildTensorIndex("u_2", Cartesian, ignore_updown=True)
p_3 = sy.tensor.tensor.WildTensorIndex("p_2", Cartesian)
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

#replace tests, sympy.tensor.tensor
W = sympy.tensor.tensor.WildTensorHead('W', unordered_indices=True)
U = sympy.tensor.tensor.WildTensorHead('U')
_WildTensExpr = sy.tensor.tensor._WildTensExpr

assert (
	W().matches( K(p)*V(q) )
	== {
		W(): K(p)*V(q),
		}
	)
assert (
	W(p,q).matches( K(p)*V(q) )
	== {
		W(p,q).head: _WildTensExpr(K(p)*V(q))
		}
	)
check_tens_eq(
	(K(p) * V(-p)).replace( W(q) * V(-q), 1),
	1
	)
check_tens_eq(
	(K(q) * K(p) ).replace( W(q,p), 1),
	1
	)
check_tens_eq(
	( K(q) * K(p) * V(-p) ).replace( W(q,p) * V(-p), 1),
	1
	)

check_tens_eq(
	( K(p) * V(-p) ).replace( K(-p)* V(p), 1 ),
	1
	)
check_tens_eq(
	( K(q) * K(p) * V(-p) ).replace( W(q)* U(p) * V(-p), 1),
	1
	)
check_tens_eq(
	(K(p)*V(q)).replace(
		W()*K(p)*V(q),
		W()*V(p)*V(q),
		),
	V(p)*V(q)
	)

assert (
	p_2.matches(q)
	== {p_2:q}
	)
assert (
	p_2.matches(-q)
	== {p_2:-q}
	)
assert (
	p_3.matches(-q)
	== None
	)
assert (
	p_3.matches(q)
	== {p_3:q}
	)
assert(
	( wi*K(p) ).matches( K(p) )
	== {wi: 1}
	)
assert (
	eps(p,-p_2,p_2).matches( eps(p,q,r) )
	== None
	)
assert(
	eps(p,-q_2,p_2).matches( eps(p,q,r) )
	== {p_2: r, -q_2: q}
	)

check_tens_eq(
	eps(p,-q,r).replace(
		eps(p_2,s_2,t_2), 1
		),
	1
	)

check_tens_eq(
	( eps(r,p,q) * eps(-r, -s, -t) ).replace(
		eps(r, p_2, q_2) * eps(-r, s_2, t_2),
		delta(p_2, s_2)*delta(q_2, t_2) - delta(p_2, t_2)*delta(q_2, s_2),
		),
	delta(p,-s)*delta(q,-t) - delta(p,-t)*delta(q,-s)
	)
check_tens_eq(
	( eps(r,p,q) * eps(-r, -p, -q) ).replace(
		eps(r_2, p_2, q_2) * eps(-r_2, s_2, t_2),
		delta(p_2, s_2)*delta(q_2, t_2) - delta(p_2, t_2)*delta(q_2, s_2),
		).contract_delta(delta),
	6,
	)
check_tens_eq(
	( eps(r,p,q) * eps(-p, -r, -q) ).replace(
			wi * W() * eps(r, p_2, q_2) * eps(-r, s_2, t_2),
			wi * W() *( delta(p_2, s_2)*delta(q_2, t_2) - delta(p_2, t_2)*delta(q_2, s_2) ),
			).contract_delta(delta),
	- 6
	)
check_tens_eq(
	( eps(r,p,s) * eps(-p, -r, -q) ).replace(
			wi * W() * eps(r, p_2, q_2) * eps(-r, s_2, t_2),
			wi * W() *( delta(p_2, s_2)*delta(q_2, t_2) - delta(p_2, t_2)*delta(q_2, s_2) ),
			).contract_delta(delta),
	- 2*delta(-q, s)
	)

# #Multiple occurrence of WildTensor in value
check_tens_eq(
	( K(p)*V(q) ).replace(W(q)*K(p), W(p)*W(q)),
	V(p)*V(q)
	)
check_tens_eq(
	( K(p)*V(q)*V(r) ).replace(W(q,r)*K(p), W(p,r)*W(q,s)*V(-s) ),
	V(p)*V(r)*V(q)*V(s)*V(-s)
	)

#Replace over TensAdd
check_tens_eq(
	( K(p) + V(p) ).replace(K(p), V(p)),
	2*V(p)
	)
check_tens_eq(
	( K(p)*V(q) + V(p)*V(q) ).replace(K(p), V(p)),
	2*V(p)*V(q)
	)
check_tens_eq(
	( K(p)*V(q) + K(p)*K(q) + K(q)*V(p) + V(q)*V(p) ).replace(
		W(p,q) + K(p)*K(q) + K(q)*V(p),
		5*K(p)*K(q) + W(p,q)
		),
	K(p)*V(q) + V(q)*V(p) + 5*K(p)*K(q)
	)

check_tens_eq(
	( K(p)*K(q) ).replace( W(p,q), W(p,-r)*W(q,r) ),
	K(p)*K(q)*K(r)*K(-r)
	)
check_tens_eq(
	( K(p)*V(q)*V(r) ).replace(W(q,r)*K(p), W(p,r)*W(q,s)*V(-s) ),
	V(p)*V(r)*V(q)*V(s)*V(-s)
	)
