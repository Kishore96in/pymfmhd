from sympy.tensor.tensor import TensExpr
from sympy.tensor.toperators import PartialDerivative
from sympy import S, Basic

class funDer(PartialDerivative):
	def __new__(cls, expr, *variables, **kwargs):
		"""
		expr: TensExpr
		variables: list of TensorField
		
		Represents a functional derivative
		E.g. to represent \frac{\delta \rho(x,t)}{\delta v_i(y,\tau)}:
		>>> R3 = TensorIndexType('R', dim=3)
		>>> i, j = tensor_indices("i j", R3)
		>>> x = TensorHead('x', [R3])
		>>> rho = Function('rho')
		>>> v = TensorFieldHead('v', [R3], [x])
		>>> funDer(rho(x), v(i))
		
		A second-order functional derivative would be
		>>> funDer(rho(x), v(i), v(j))
		
		"""
		# Flatten:
		if isinstance(expr, funDer):
			variables = expr.variables + variables
			expr = expr.expr
		
		if len(variables) == 0:
			return expr
		
		args, indices, free, dum = cls._contract_indices_for_derivative(
			S(expr), variables)
		
		obj = TensExpr.__new__(cls, *args)
		
		obj._indices = indices
		obj._free = free
		obj._dum = dum
		return obj
	
	@property
	def order(self):
		return len(self.variables)
	
	def _perform_derivative(self):
		return self

class averagedFunDer(funDer):
	"""
	Used to denote the average of a functional derivative.
	"""
	def __new__(cls, expr, variables, wrt):
		"""
		expr: TensExpr
		variables: list of TensorField
		wrt: Symbol (used to distinguish different kinds of averages)
		"""
		
		# Flatten:
		if isinstance(expr, averagedFunDer):
			raise NotImplementedError
		elif isinstance(expr, funDer):
			variables = expr.variables + variables
			expr = expr.expr
		
		if len(variables) == 0:
			return expr
		
		args, indices, free, dum = cls._contract_indices_for_derivative(
			S(expr), variables)
		
		obj = Basic.__new__(cls, args[0], tuple(args[1:]), wrt)
		obj._indices = indices
		obj._free = free
		obj._dum = dum
		obj.wrt = wrt
		return obj
	
	@property
	def variables(self):
		return list(self.args[1])
	
	def _replace_indices(self, repl):
		expr = self.expr.xreplace(repl)
		mirrored = {-k: -v for k, v in repl.items()}
		variables = [i.xreplace(mirrored) for i in self.variables]
		return self.func(expr, variables, self.wrt)
	
	def _apply_recursion_relation(self, corr, average):
		return NotImplementedError

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
		funDer(
			rho(pos=[X,t,up]),
			V(p, pos=[Y,tau,up]),
			)
		)
	
	print(
		funDer(
			rho(pos=[X,t,up]),
			)
		)
