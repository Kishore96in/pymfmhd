from sympy import S, Basic, Function, Heaviside
from sympy.tensor.tensor import TensorIndex, TensExpr, get_indices
from sympy.tensor.toperators import PartialDerivative

#Stuff from pymfmhd
from functionalDerivative import funDer
from average import average
from tensorField import TensorFieldHead

class Integ(TensExpr):
	def __new__(cls, expr, *variables):
		"""
		expr: TensExpr
		variables: Any sympy objects
		"""
		obj = Basic.__new__(cls, expr, tuple(variables))
		obj.expr = expr
		obj.variables = variables
		
		indices = []
		indices.extend(get_indices(expr))
		for v in variables:
			indices.extend(get_indices(v))
		
		obj._indices = indices
		obj._free = [i for i in indices if -i not in indices]
		
		return obj
	
	@property
	def nocoeff(self):
		return self

	@property
	def coeff(self):
		return S.One

	def get_indices(self):
		return self._indices

	def get_free_indices(self):
		return self._free

	def _replace_indices(self, repl):
		return self.xreplace(repl)
	
	def _latex(self, printer):
		expr = printer._print(self.expr)
		variables = [printer._print(v) for v in self.variables]
		variables = ",\, ".join(variables)
		return r"\int_{%s} \left(%s\right)" % (variables, expr)

class dirac(Function):
	pass

class averagedFunDer(funDer):
	def __new__(cls, expr, *variables, **kwargs):
		wrt = kwargs.pop("wrt")
		av = kwargs.pop("average", average)
		
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
		
		obj = Basic.__new__(cls, args[0], wrt, av, *args[1:])
		obj._indices = indices
		obj._free = free
		obj._dum = dum
		obj.wrt = wrt
		obj.average = av
		return obj
	
	@property
	def variables(self):
		return self.args[3:]
	
	def _replace_indices(self, repl):
		expr = self.expr.xreplace(repl)
		mirrored = {-k: -v for k, v in repl.items()}
		variables = [i.xreplace(mirrored) for i in self.variables]
		return self.func(expr, *variables, wrt=self.wrt, average=self.average)
	
	def _apply_recursion_relation(self):
		assert len(self.expr.positions) > 2 #NOTE: I am allowing for >2 since I will need to have a wrt variable for the average
		head = self.variables[0].head
		assert len(head.positions) > 2 #NOTE: I am allowing for >2 since I will need to have a wrt variable for the average
		assert len(head.index_types) == 1
		for v in self.variables[1:]:
			assert v.head == head
		
		def make_dummies():
			pos = head.positions[0]
			time = head.positions[1]
			
			i_dum = TensorIndex(True, head.index_types[0])
			i = len(head.index_types[0]._autogenerated)
			pos_dum = pos.func(f"y^({i})", *pos.args[1:])
			time_dum = time.func(f"tau_{i}", *time.args[1:])
			
			return i_dum, pos_dum, time_dum
		
		x,t,_ = self.expr.positions
		i0, y0, tau0 = make_dummies()
		i1, y1, tau1 = make_dummies()
		i2, _, _ = make_dummies()
		
		corr = TensorFieldHead("Q", [head.index_types[0], head.index_types[0]], positions=[None, None])
		
		ret = 0
		ret += Integ(
			PartialDerivative(dirac(x(i2)-y0(i2)), y0(i0) ) * Heaviside(t-tau0) * self.func( self.expr.xreplace({x:y0, t:tau0}), *self.variables, wrt=self.wrt ) * self.average( head(i0, pos=[y0, tau0]) , self.wrt) ,
			y0, tau0
			)
		ret += Integ(
			PartialDerivative(dirac(x(i2)-y0(i2)), y0(i0) ) * Heaviside(t-tau0) * self.func( self.expr.xreplace({x:y0, t:tau0}), *self.variables, head(i1, pos=[y1,tau1]), wrt=self.wrt ) * corr(i0, i1, pos=[(y0,tau0), (y1, tau1)]),
			y0, tau0,
			y1, tau1
			)
		for alpha in range(len(self.variables)):
			other_vars = [self.variables[i] for i in range(self.order) if i != alpha]
			i_a = self.variables[alpha].get_indices()[0]
			y_a = self.variables[alpha].positions[0]
			tau_a = self.variables[alpha].positions[1]
			
			ret += PartialDerivative(dirac(x(i2)-y_a(i2)), ya(i_a) ) * Heaviside(t-tau_a) * self.func( self.expr.xreplace({x:y_alpha, t:tau_alpha}), *other_vars )
		
		return ret

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
	
