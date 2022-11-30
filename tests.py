from termcolor import cprint #for bold terminal output

import sympy as sy
import sympy.tensor.tensor

from average import average
from tensor import do_epsilon_delta, partialdiff, dive_matcher, mul_matcher
from integral import do_angular_integral, AngularIntegral, create_scalar_integral

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
check_tens_eq(
	do_angular_integral( delta(p,q), K ),
	4*sy.pi*delta(p,q)
	)
check_tens_eq(
	do_angular_integral( V(p)*V(q), K ),
	4*sy.pi*V(p)*V(q)
	)
assert(
	do_angular_integral(K(p)*K(q)*K(r)*K(s)*K(t)*K(u)*K(i)*K(j), K)
	== AngularIntegral(K(p)*K(q)*K(r)*K(s)*K(t)*K(u)*K(i)*K(j), K)
	)
check_tens_eq(
	AngularIntegral( K(p)*K(q), K).doit(),
	4*sy.pi/3 * K(r)*K(-r) * delta(p,q)
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

#Tests for create_scalar_integral
assert create_scalar_integral(K(p)*K(q)*g(x), x) == sy.Integral(g(x), x)*K(p)*K(q)
assert create_scalar_integral(K(q)*g(x) + V(q)*f(x), x) == sy.Integral(g(x), x)*K(q) + sy.Integral(f(x), x)*V(q)


#################
cprint("All tests passed", attrs=['bold'])
