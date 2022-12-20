from sympy import Basic
from sympy.tensor.tensor import TensorIndex, TensExpr
from sympy.tensor.toperators import PartialDerivative

#Stuff from pymfmhd
from functionalDerivative import funDer
from average import average

class Integ(TensExpr):
	def __new__(expr, *variables):
		"""
		expr: TensExpr
		variables: Any sympy objects
		"""
		obj = Basic.__new__(cls, expr, tuple(variables))
		obj.expr = expr
		obj.variables = variables
		return obj
	
	def _latex(self, printer):
		expr = printer._print(self.expr)
		variables = [printer._print(v) for v in self.variables]
		variables = ",\, ".join(variables)
		return r"\int_{%s} \left(%s\right)" % (variables, expr)

class averagedFunDer(funDer):
	def __new__(cls, expr, *variables, **kwargs):
		obj = funDer.__new__(cls, expr, *variables)
		obj.wrt = kwargs.pop("wrt")
		obj.average = kwargs.pop("average", average)
		return obj

if __name__ == "__main__":
	from tensorField import TensorFieldHead
	from sympy.tensor.tensor import TensorIndexType, TensorIndex, TensorHead
	from sympy import symbols
	
	R3 = TensorIndexType('R3', dim=3)
	R3.set_metric(R3.delta)
	delta = R3.delta
	eps = R3.epsilon
	
	up = symbols("↑") #Used to denote the average we are taking
	t, tau = symbols("t tau")
	p, q, r, s = symbols("p q r s", cls=TensorIndex, tensor_index_type=R3)
	X = TensorHead("X", [R3])
	Y = TensorHead("Y", [R3])
	V = TensorFieldHead("V", [R3], positions=[X,t,up])
	rho = TensorFieldHead("rho", [], positions=[X,t,up])
	
	print(
		averagedFunDer(rho(pos=[X,t,up]), wrt=up)
		)
	
	print(
		averagedFunDer(
			rho(pos=[X,t,up]),
			V(p, pos=[Y,tau,up]),
			V(q, pos=[Y,tau,up]),
			wrt=up
			)
		)
	
