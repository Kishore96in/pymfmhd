import sympy as sy
import sympy.tensor.tensor

from .helpers import check_tens_eq

from pymfmhd.tensorField import TensorFieldHead
from pymfmhd.functionalDerivative import funDer, averagedFunDer

Cartesian = sy.tensor.tensor.TensorIndexType('Cartesian', dim=3)
p, q = sy.tensor.tensor.tensor_indices("p q", Cartesian)
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
