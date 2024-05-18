import sympy as sy
import sympy.tensor.tensor

from .helpers import check_tens_eq

from pymfmhd.integral import do_angular_integral, AngularIntegral, create_scalar_integral

Cartesian = sy.tensor.tensor.TensorIndexType('Cartesian', dim=3)
p, q, r, s, t, u, w, i, j = sy.tensor.tensor.tensor_indices("p q r s t u w i j", Cartesian)
delta = Cartesian.delta
K = sy.tensor.tensor.TensorHead("K", [Cartesian])
V = sy.tensor.tensor.TensorHead("V", [Cartesian])
f, g = sympy.symbols('f g', cls=sympy.Function)
x = sy.symbols("x")

def test_ang_int():
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
	check_tens_eq(
		do_angular_integral(K(p)*K(q)*K(r)*K(s)*K(t)*K(u)*K(i)*K(-p), K).contract_delta(delta),
		K(p)*K(-p)*do_angular_integral(K(q)*K(r)*K(s)*K(t)*K(u)*K(i), K)
		)
	check_tens_eq(
		AngularIntegral( K(p)*K(q), K).doit(),
		4*sy.pi/3 * K(r)*K(-r) * delta(p,q)
		)

def test_create_scalar_integral():
	assert create_scalar_integral(K(p)*K(q)*g(x), x) == sy.Integral(g(x), x)*K(p)*K(q)
	assert create_scalar_integral(K(q)*g(x) + V(q)*f(x), x) == sy.Integral(g(x), x)*K(q) + sy.Integral(f(x), x)*V(q)
