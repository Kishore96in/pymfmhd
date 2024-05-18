import sympy as sy
import sympy.tensor.tensor

from .helpers import check_tens_eq

from pymfmhd.tensor import do_epsilon_delta, partialdiff, PartialVectorDerivative

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

K = sy.tensor.tensor.TensorHead("K", [Cartesian])
V = sy.tensor.tensor.TensorHead("V", [Cartesian])

f, g = sympy.symbols('f g', cls=sympy.Function)
k = sympy.symbols("k") #'amplitude' of K

def test_eps_delta():
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

def test_partial_derivative():
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
	check_tens_eq(
		partialdiff(
			f(k)*( delta(p,q) - K(p)*K(q)/k**2 ), K(q), k
			),
		sympy.Derivative(f(k),k)*K(p)/k - 4/k**2 * f(k)*K(p) + sympy.Derivative(-f(k)/k**2,k)/k * K(p)*K(q)*K(-q)
		)

def test_PartialVectorDerivative():
	pd = PartialVectorDerivative
	expr = pd(pd(pd(f(k), K(p), k), K(-p), k), K(q), k)
	R0 = expr.get_indices()[1]
	assert expr._replace_indices({-q:R0, R0:q, -R0:-q}).get_indices()[1] != R0
	e1 = pd(pd(f(K), K(p), k), K(-p), k)
	e2 = pd(g(K), K(p), k)*K(p)
	e3 = e2.xreplace({g(K): e1})
	e4 = pd(pd(pd(f(K), K(p), k), K(-p), k), K(q), k)*K(q)
	assert e3.doit(deep=False) == e4
