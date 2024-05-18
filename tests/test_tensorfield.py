import sympy as sy
import sympy.tensor.tensor

from .helpers import check_tens_eq

from pymfmhd.tensorField import TensorFieldHead


Cartesian = sy.tensor.tensor.TensorIndexType('Cartesian', dim=3)
p = sy.tensor.tensor.tensor_indices("p", Cartesian)
K = sy.tensor.tensor.TensorHead("K", [Cartesian])
V = sy.tensor.tensor.TensorHead("V", [Cartesian])
T = TensorFieldHead("T", [Cartesian], positions=[K])

def test_tensorfieldhead():
	assert T(p).component.name == "T"
	assert T(p).positions == (K,)
	assert T(p).get_indices() == [p]
	assert T(p, pos=[V]).positions == (V,)
