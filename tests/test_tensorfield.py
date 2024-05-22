import sympy as sy
import sympy.tensor.tensor
from sympy.tensor.toperators import PartialDerivative

from .helpers import check_tens_eq

from pymfmhd.tensorField import TensorFieldHead


Cartesian = sy.tensor.tensor.TensorIndexType('Cartesian', dim=3)
p, q = sy.tensor.tensor.tensor_indices("p q", Cartesian)
K = sy.tensor.tensor.TensorHead("K", [Cartesian])
V = sy.tensor.tensor.TensorHead("V", [Cartesian])
T = TensorFieldHead("T", [Cartesian], positions=[K])
T2 = TensorFieldHead("T2", [Cartesian], positions=[K, V])

def test_tensorfieldhead():
	assert T(p).component.name == "T"
	assert T(p).positions == (K,)
	assert T(p).get_indices() == [p]
	assert T(p, pos=[V]).positions == (V,)

def test_tensorField_derivative():
	PD = PartialDerivative
	PDdoit = lambda *args, **kwargs: PartialDerivative(*args, **kwargs)._perform_derivative()
	
	assert PDdoit(T(p), V(q)) == 0
	assert PDdoit(T(p), K(q)) == PD(T(p), K(q))
	assert PDdoit(T2(p), V(p)) == PD(T2(p), V(p))
