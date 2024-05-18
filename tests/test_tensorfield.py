import sympy as sy
import sympy.tensor.tensor

from .helpers import check_tens_eq

from pymfmhd.tensorField import TensorFieldHead


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

T = TensorFieldHead("T", [Cartesian], positions=[K])

def test_tensorfieldhead():
	assert T(p).component.name == "T"
	assert T(p).positions == (K,)
	assert T(p).get_indices() == [p]
	assert T(p, pos=[V]).positions == (V,)
