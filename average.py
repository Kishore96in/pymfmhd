import sympy
import sympy.tensor.tensor

from sympy import Basic, Add, Mul, S, Tuple
from sympy.tensor.tensor import TensExpr, get_indices, get_free_indices, TensAdd, TensMul
from collections.abc import Iterable

class average(TensExpr):
	"""
	Represents a Reynolds average. First argument is the expression averaged, second argument is the set of symbols over which the average is done.
	TODO: Actually, the need to specify 'wrt' makes this unable to represent an ensemble average, I think! Problem? Or does that just mean we need to add a dummy 'realization label' to each stochastic field?
	"""
	def __new__(cls, expr, wrt):
		"""
		expr: TensExpr
		wrt: set
		"""
		if isinstance(wrt, set):
			wrt = Tuple(*wrt)
		if isinstance(wrt, Iterable):
			wrt = Tuple(*set(wrt))
		else:
			wrt = Tuple(*set([wrt]))
		
		obj = Basic.__new__(cls, expr, wrt)
		obj.expr = expr
		obj.wrt = wrt
		
		obj._indices = get_indices(expr)
		obj._free = get_free_indices(expr)
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
		return r"\left<{%s}\right>" % (expr)
	
	@property
	def free_symbols(self):
		wrt = set(self.wrt) #Make sure this is a set. For some reason, sympy seems to instantiate it as a finiteset. Not sure if that is a sympy bug.
		sym = self.expr.free_symbols
		return {s for s in sym if s not in wrt}
	
	def _eval_simplify(self, **kwargs):
		from sympy.simplify.simplify import simplify
		"""
		TODO: Implement all Reynolds rules. Need to implement
		* some way to 'mark' a field as a fluctuation, such that all averages of it become zero. VEST basically assumes there is only a single spatial variable, and needs one to explicitly mark objects as constants at instantiation.
		* implement commutation with derivatives and integrals over independent variables.
		"""
		#TODO: Support a 'deep' argument like powsimp does.
		arg = self.expr
		wrt = self.wrt
		
		if isinstance(arg, (Add, TensAdd)):
			return TensAdd(*[ simplify( average(i,wrt), **kwargs) for i in arg.args]).doit(deep=False)
		elif isinstance(arg, (Mul, TensMul)):
			inside = []
			outside = []
			for a in arg.args:
				if any([ s in  a.free_symbols for s in wrt ]):
					inside.append(a)
				#TODO: Do I need to worry about the case of bound symbols? check for that is any([ s in  a.atoms(sympy.Symbol) for s in wrt ])
				else:
					outside.append(a)
			outmulsimp = simplify(arg.func(*outside).doit(deep=False), **kwargs)
			if len(inside) > 0:
				return TensMul(outmulsimp, average(simplify(arg.func(*inside).doit(deep=False), **kwargs), wrt) ).doit(deep=False)
			else:
				return outmulsimp
		elif isinstance(arg, average):
			new_wrt = wrt.union(arg.args[1])
			return average(arg.args[0].simplify(), new_wrt)
		else:
			#If any variable in wrt is not present in arg, remove that variable from wrt
			new_wrt = set()
			for s in wrt:
				if s in arg.free_symbols:
					new_wrt.add(s)
			if len(new_wrt) == 0:
				return simplify(arg, **kwargs)
			elif new_wrt != wrt:
				return average(simplify(arg, **kwargs), new_wrt)
			else:
				return self

if __name__ == "__main__":
	pass
