import sympy as sy
import sympy.tensor.tensor
import pytest

from .helpers import check_tens_eq

from pymfmhd.tensor import (
	do_epsilon_delta,
	partialdiff,
	PartialVectorDerivative,
	PartialDiffScalarWarning,
	FunctionOfTensor,
	)

Cartesian = sy.tensor.tensor.TensorIndexType('Cartesian', dim=3)
p, q, r, s, t, u = sy.tensor.tensor.tensor_indices("p q r s t u", Cartesian)
delta = Cartesian.delta
eps = Cartesian.epsilon
K = sy.tensor.tensor.TensorHead("K", [Cartesian])
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
	
	with pytest.warns(PartialDiffScalarWarning):
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
	#TODO: The above was earlier failing, and the same bug is reproducible with PartialDerivative itself (before calling doit, so that more complicated cases are not flattened). Probably the fix can be similar to what I have done in <https://github.com/sympy/sympy/pull/24338>. Now that I have fixed this, I should also fix the one in PartialDerivative.
	
	e1 = pd(pd(f(K), K(p), k), K(-p), k)
	e2 = pd(g(K), K(p), k)*K(p)
	e3 = e2.xreplace({g(K): e1})
	e4 = pd(pd(pd(f(K), K(p), k), K(-p), k), K(q), k)*K(q)
	assert e3.doit(deep=False) == e4
	
	# assert pd(K(p), K(q), k).components == []

def test_FunctionOfTensor():
	F = FunctionOfTensor("F")
	a = sy.Symbol("a")
	
	with pytest.raises(ValueError):
		F(K(p))
	
	expr = K(p) * K(-p)
	fun = F(expr)
	assert len(fun.get_free_indices()) == 0
	assert fun.get_indices() == expr.get_indices()
	assert fun.replace_with_arrays({K(p): [1,2,3], Cartesian: sy.eye(3)}) == F(14).doit()
	
	fun = F(expr, k)
	assert len(fun.get_free_indices()) == 0
	assert fun.get_indices() == expr.get_indices()
	assert fun.replace_with_arrays({K(p): [1,2,3], Cartesian: sy.eye(3)}) == F(14,k).doit()
	
	fun = 3*F(expr)
	assert len(fun.get_free_indices()) == 0
	assert fun.get_indices() == expr.get_indices()
	assert fun.replace_with_arrays({K(p): [1,2,3], Cartesian: sy.eye(3)}) == 3*F(14).doit()
	
	assert not isinstance(F(14).doit(), FunctionOfTensor)
	
	fun = 2 + F(expr)
	assert len(fun.get_free_indices()) == 0
	assert fun.get_indices() == expr.get_indices()
	assert fun.replace_with_arrays({K(p): [1,2,3], Cartesian: sy.eye(3)}) == 2 + F(14).doit()
	
	fun = F(k, a)
	assert fun.replace(k, a) == F(a, a)
	
	#Test contract_delta
	fun = (1/2) * F(expr) * K(r) * delta(-r,s)
	# assert fun.contract_delta(delta) == (1/2) * F(expr) * K(s) # TODO: fails unless some assertions (`isinstance(..., Tensor)`) are removed in sympy.tensor.tensor. But first need to think about whether I should make FunctionOfTensor a subclass of Tensor.
	
	# assert fun.canon_bp() == (1/2) * F(expr) * K(s) # TODO: fails because commutes_with is not implemented yet.

@pytest.mark.xfail(reason="no way to check commutation relations between generic TensExpr instances")
def test_FunctionOfTensor_commute():
	TensorManager = sympy.tensor.tensor.TensorManager
	F = FunctionOfTensor("F")
	a = sy.Symbol("a")
	
	K = sy.tensor.tensor.TensorHead("K", [Cartesian], comm='A')
	P = sy.tensor.tensor.TensorHead("P", [Cartesian], comm='A')
	Q = sy.tensor.tensor.TensorHead("Q", [Cartesian], comm='B')
	
	TensorManager.set_comm('A', 'A', 0)
	TensorManager.set_comm('A', 'B', 1)
	
	f_1 = F(K(p)*K(-p))
	
	assert f_1.commutes_with(K(q)) == 0
	assert K(q).commutes_with(f_1) == 0
	
	assert f_1.commutes_with(Q(q)) != 0
	assert Q(q).commutes_with(f_1) != 0
