from sympy import symbols, S, Wild, Integral, pi, I, Function, oo, Derivative, diff
from sympy.tensor.tensor import TensorHead, TensorIndex, TensorIndexType, WildTensorHead, WildTensor, WildTensorIndex

from pymfmhd.fourier import ift_derivative_rule, ift_convolution
from pymfmhd.tensor import replace_by_ampl_optimized as repl_wavevec

R3 = TensorIndexType('R3', dim=3)
R3.set_metric(R3.delta)
delta = R3.delta
eps = R3.epsilon

_a, _b, _c, _d, _e, _f = symbols("_a _b _c _d _e _f", cls=WildTensorIndex, tensor_index_type=R3, ignore_updown=True)
a,b,c,d,e,f,g,i,j = symbols("a b c d e f g i j", cls = TensorIndex, tensor_index_type=R3)
s,i_1,m,i_2,n,i_3 = symbols("s i_1 m i_2 n i_3", cls = TensorIndex, tensor_index_type=R3)
W = WildTensorHead("W")
w = Wild('w')

K, K1, Q, P, P2, R = symbols("K K1 Q P P^(2) R", cls=TensorHead, index_types=[R3])
k, k1, q, p, p2, r = symbols("k k1 q p p^(2) r", positive=True)

F = Function("F")
G = Function("G")
H = Function("H")

def test_ift_derivative_rule():
	fdr = lambda *args: ift_derivative_rule(*args).doit()
	assert fdr(F(k), [K,k], [R,r]) == F(r)
	assert fdr(3*F(k), [K,k], [R,r]) == 3*F(r)
	assert fdr(k**(-2)*K(a)*F(k)*K(-a), [K,k], [R,r]) == F(r)
	assert fdr(5*k**(-2)*K(a)*F(k)*K(-a), [K,k], [R,r]) == 5*F(r)
	assert fdr(k**(-2)*F(k), [K,k], [R,r]) == -Integral(Integral(r**2*F(r), (r, 0, r))/r**2, (r, 0, r))
	#TODO: fdr(k**(-4)*F(k), [K,k], [R,r]).simplify() looks too simple to be true. Figure out what is going on. The unsimplified output looks fine; need to check whether the simplification is legit or if it is a sympy bug.
	assert fdr(k**(4)*F(k), [K,k], [R,r]).doit().expand() == Derivative(F(r), (r, 4)) + 4*Derivative(F(r), (r, 3))/r

def test_ift_convolution():
	fc = lambda *args: ift_convolution(*args).doit()
	assert fc(F(p)*G(q), [[P,p], [Q,q]], [K,k], [R, r]) == F(r)*G(r)
	assert fc(2*F(p)*G(q), [[P,p], [Q,q]], [K,k], [R, r]) == 2*F(r)*G(r)
	assert fc(Q(a)*F(p)*G(q), [[P,p], [Q,q]], [K,k], [R, r]) == I/r*Derivative(G(r), r)*F(r)*R(a)
	assert repl_wavevec(
		fc(F(p)*G(q)*P(a)*Q(-a), [[P,p], [Q,q]], [K,k], [R, r]),
		R, r,
		).doit() == - Derivative(F(r),r)*Derivative(G(r),r)
	assert fc(k**2*F(p)*G(q), [[P,p], [Q,q]], [K,k], [R, r]) == -Derivative(r**2*Derivative(F(r)*G(r), r), r)/r**2
	assert fc(K(a)*F(p)*G(q), [[P,p], [Q,q]], [K,k], [R, r]) == I*(F(r)*Derivative(G(r), r)/r + G(r)*Derivative(F(r), r)/r)*R(a)
	assert fc(k**2*F(p)*G(q)*P(a)*Q(-a), [[P,p], [Q,q]], [K,k], [R, r]).expand()  == - Derivative(r**2*Derivative( - Derivative(F(r), r)*Derivative(G(r), r), r), r)/r**2 #TODO: Would be nicer to get the minus outside. Not sure why it's sticking inside.
	
	assert fc(F(p)*G(q)*H(p2), [[P,p], [Q,q], [P2,p2]], [K,k], [R, r]) == F(r)*G(r)*H(r)
	assert fc(Q(a)*F(p)*G(q)*H(p2), [[P,p], [Q,q], [P2,p2]], [K,k], [R, r]) == I/r*Derivative(G(r), r)*F(r)*H(r)*R(a)
