from sympy import symbols, Function, Derivative, diff
import pytest

from pymfmhd.rewrite import bring_to_form, revert_product_rule, CallLimitWarning, ResultRejectedWarning

def test_product_rule():
	F, G = symbols("F G", cls=Function)
	x, y = symbols("x y")
	
	dpr = lambda expr: revert_product_rule(expr, x)
	
	assert dpr(F(x)*diff(G(x),x) + G(x)*diff(F(x),x)) == Derivative(F(x)*G(x))
	assert dpr(x**2*F(x)*diff(G(x),x) + x**2*G(x)*diff(F(x),x) + 2*x*F(x)*G(x)) == Derivative(x**2*F(x)*G(x))
	assert dpr(x**2*diff(F(x),x)*diff(G(x),x) + x**2*G(x)*diff(F(x),x,x) + 2*x*diff(F(x),x)*G(x)) == Derivative(x**2*Derivative(F(x),x)*G(x))
	assert dpr(4*x**2*diff(F(x),x)*diff(G(x),x) + 4*x**2*G(x)*diff(F(x),x,x) + 4*2*x*diff(F(x),x)*G(x)) == 4*Derivative(x**2*Derivative(F(x),x)*G(x))
	assert dpr(diff(F(x)*G(x)*F(y), x)) == Derivative(F(x)*G(x)*F(y), x)
	
	with pytest.warns(CallLimitWarning):
		with pytest.warns(ResultRejectedWarning):
			assert dpr(F(x)*diff(G(x), x) + 1) == F(x)*Derivative(G(x), x) + 1
	
	assert dpr(diff(F(x)*G(x) + x**2*diff(F(x),x)*G(x), x)) == Derivative(F(x)*G(x), x) + Derivative(x**2*diff(F(x),x)*G(x), x)
	assert dpr( - 2*diff(x**3*diff(F(x), x)*diff(G(x), x), x) ) == - 2*Derivative(x**3*Derivative(F(x), x)*Derivative(G(x), x), x)
	assert dpr( 2*F(x)*diff(F(x), x) + 1 ) == Derivative(F(x)**2, x) + 1
	assert dpr( F(x)**2*diff(F(x), x) + 1 ) == Derivative(F(x)**3, x)/3 + 1
	assert dpr( F(x)**2*diff(F(x), x) + F(x)*diff(G(x),x) + G(x)*diff(F(x),x) ) == Derivative(F(x)**3, x)/3 + Derivative(F(x)*G(x), x)

def test_bring_to_form():
	F, G = symbols("F G", cls=Function)
	r = symbols("r")
	
	assert bring_to_form(Derivative(F(r),r) + F(r)/r, Derivative(F(r),r), Derivative(r*F(r),r)/r) == Derivative(r*F(r),r)/r
	assert bring_to_form(G(r)*Derivative(F(r),r) + G(r)*F(r)/r, Derivative(F(r),r), Derivative(r*F(r),r)/r) == G(r)*Derivative(r*F(r),r)/r
	assert bring_to_form(r*Derivative(F(r),r) + F(r), Derivative(F(r),r), Derivative(r*F(r),r)/r) == Derivative(r*F(r),r)
	assert bring_to_form(r*Derivative(F(r),r) + F(r), r*Derivative(F(r),r), Derivative(r*F(r),r), wild_properties=[lambda expr: not expr.has(r, F)]) == Derivative(r*F(r),r)
