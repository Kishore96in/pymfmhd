import sympy as sy
import sympy.tensor.tensor

from .helpers import check_tens_eq

from pymfmhd.tensorField import TensorFieldHead
from pymfmhd.functionalDerivative import funDer, averagedFunDer

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

up = sy.symbols("↑") #Used to denote the average we are taking
t, tau = sy.symbols("t tau")
X = sy.tensor.tensor.TensorHead("X", [Cartesian])
Y = sy.tensor.tensor.TensorHead("Y", [Cartesian])
V = TensorFieldHead("V", [Cartesian], positions=[X,t,up])
rho = TensorFieldHead("rho", [], positions=[X,t,up])

def test_funDer():
	assert funDer(rho(pos=[X,t,up])) == rho(pos=[X,t,up])
	assert funDer(rho(pos=[X,t,up]), V(p, pos=[Y,tau,up]))._replace_indices({-p:-q}) == funDer(rho(pos=[X,t,up]), V(q, pos=[Y,tau,up]))

def test_avFunDer():
	assert averagedFunDer(rho(pos=[X,t,up]), [V(p, pos=[Y,tau,up])], up)._replace_indices({-p:-q}) == averagedFunDer(rho(pos=[X,t,up]), [V(q, pos=[Y,tau,up])], up)
