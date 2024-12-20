"""
Tools to perform inverse Fourier transforms
"""

import numpy as np
from sympy import (
	Symbol,
	Function,
	I,
	Add,
	Integral,
	Mul,
	Pow,
	Derivative,
	Basic,
	sympify,
	)
from sympy.tensor.tensor import (
	TensorHead,
	TensMul,
	TensAdd,
	Tensor,
	TensExpr,
	get_free_indices,
	get_indices,
	)

from .tensor import (
	PartialVectorDerivative,
	replace_by_ampl_optimized as repl_wavevec,
	contract_delta,
	)

def _depends_on_any(expr, syms):
	"""
	Return True if any of the variables in syms are present in expr. As a special case, it returns False when syms is None
	
	expr: sympy expression
	syms: list
	"""
	if syms is None:
		return False
	
	expr_atoms = expr.atoms(*[type(s) for s in syms])
	for s in syms:
		if s in expr_atoms:
			return True
	return False

class Laplacian(TensExpr):
	"""
	Represents an unevaluated Laplacian.
	
	Arguments:
		expr: TensExpr
		ampl: Symbol, represents the spatial variable
		wavevec: TensorHead, represents the spatial vector
	"""
	def __new__(cls, expr, ampl, wavevec):
		obj = Basic.__new__(cls, expr, ampl, sympify(wavevec))
		obj.expr = expr
		obj.ampl = ampl
		obj.wavevec = wavevec
		return obj
	
	@property
	def coeff(self):
		return S.One

	@property
	def nocoeff(self):
		return self
	
	@property
	def ext_rank(self):
		if hasattr(self.expr, "ext_rank"):
			return self.expr.ext_rank
		else:
			return 0
	
	def get_indices(self):
		return get_indices(self.expr)
	
	def get_free_indices(self):
		return get_free_indices(self.expr)
	
	def _replace_indices(self, repl):
		if hasattr(self.expr, "_replace_indices"):
			return self.func(self.expr._replace_indices(repl), self.ampl, self.wavevec)
		else:
			return self
	
	def _set_indices(self, *indices):
		repl = dict(zip(self.get_indices(), indices))
		return self._replace_indices(repl)
	
	def commutes_with(self, other):
		return None
	
	def _latex(self, printer):
		expr = printer._print(self.expr)
		ampl = printer._print(self.ampl)
		return rf"\nabla_{ampl}^{{2}} \left({expr}\right)"
	
	def doit(self, **hints):
		deep = hints.get('deep', True)
		if deep:
			expr = self.expr.doit(**hints)
		else:
			expr = self.expr
		
		r = self.ampl
		rvec = self.wavevec
		
		delta = rvec.index_types[0].delta
		expr = contract_delta(expr, delta)
		expr = repl_wavevec(expr, rvec, r)
		
		if rvec in expr.atoms(TensorHead):
			dum = TensorIndex(True, rvec.index_types[0])
			ret = PartialVectorDerivative(PartialVectorDerivative(expr, rvec(dum), r).doit(deep=False), rvec(-dum), r).doit(deep=False)
			ret = contract_delta(ret, delta)
			ret = repl_wavevec(ret, rvec, r)
			return ret
		else:
			return Derivative(r**2*Derivative(expr, r), r)/r**2

class InverseLaplacian(TensExpr):
	"""
	Represents an unevaluated inverse Laplacian.
	
	Arguments:
		expr: TensExpr
		ampl: Symbol, represents the spatial variable
		wavevec: TensorHead, represents the spatial vector
	"""
	def __new__(cls, expr, ampl, wavevec):
		obj = Basic.__new__(cls, expr, ampl, sympify(wavevec))
		obj.expr = expr
		obj.ampl = ampl
		obj.wavevec = wavevec
		return obj
	
	@property
	def coeff(self):
		return S.One

	@property
	def nocoeff(self):
		return self
	
	@property
	def ext_rank(self):
		if hasattr(self.expr, "ext_rank"):
			return self.expr.ext_rank
		else:
			return 0
	
	def get_indices(self):
		return get_indices(self.expr)
	
	def get_free_indices(self):
		return get_free_indices(self.expr)
	
	def _replace_indices(self, repl):
		if hasattr(self.expr, "_replace_indices"):
			return self.func(self.expr._replace_indices(repl), self.ampl, self.wavevec)
		else:
			return self
	
	def _set_indices(self, *indices):
		repl = dict(zip(self.get_indices(), indices))
		return self._replace_indices(repl)
	
	def commutes_with(self, other):
		return None
	
	def _latex(self, printer):
		expr = printer._print(self.expr)
		ampl = printer._print(self.ampl)
		return rf"\nabla_{ampl}^{{-2}} \left({expr}\right)"
	
	def doit(self, **hints):
		deep = hints.get('deep', True)
		if deep:
			expr = self.expr.doit(**hints)
		else:
			expr = self.expr
		
		r = self.ampl
		rvec = self.wavevec
		
		delta = rvec.index_types[0].delta
		expr = contract_delta(expr, delta)
		expr = repl_wavevec(expr, rvec, r)
		
		if rvec in expr.atoms(TensorHead):
			raise NotImplementedError(f"Argument of inverse Laplacian is a vector. Sympy's Integral function cannot handle such objects. {expr = }")
		
		return Integral(Integral(r**2*expr, (r,0,r))/r**2, (r,0,r))

def _check_wavevector(var):
	"""
	Check if the variable var is in the format required by my IFT routines.
	"""
	assert len(var) == 2
	assert isinstance(var[0], TensorHead)
	assert len(var[0].index_types) == 1
	assert isinstance(var[1], Symbol)

def ift_derivative_rule(expr, var_fourier, var_real):
	"""
	For any 3D wavevector K(a) (with K(a)*K(-a) = k**2), unknown function F(k), and even integer m, write the inverse Fourier transform of K(a_1)*…*K(a_n)*k**(m)*F(k) in terms of the inverse Fourier transform of F.
	
	expr: TensMul or Mul
	var_fourier: [TensorHead, Symbol], representing the Fourier wavevector and the symbol for its magnitude
	var_real: similar to var_fourier, but for the real-space position
	"""
	
	if isinstance(expr, (TensAdd, Add)):
		return expr.func(*[ift_derivative_rule(arg, var_fourier, var_real) for arg in expr.args])
	
	_check_wavevector(var_fourier)
	_check_wavevector(var_real)
	
	wavevec, mag = var_fourier
	rvec, r = var_real
	
	if isinstance(expr, (TensMul, Mul)):
		wavevecs = []
		powers = []
		funcs = []
		other = [1] #TensMul printer does not like having no args
		for arg in expr.args:
			atoms = arg.atoms(type(wavevec), type(mag))
			if (wavevec not in atoms) and (mag not in atoms):
				other.append(arg)
			elif isinstance(arg, Tensor) and arg.head == wavevec:
				wavevecs.append(arg)
			elif isinstance(arg, Function):
				funcs.append(arg)
			elif isinstance(arg, Pow):
				powers.append(arg)
			else:
				raise TypeError(f"Cannot handle {arg},\n\t{srepr(arg) = }\n\t{expr = }")
	elif isinstance(expr, Function):
		#Handle the case where just a single unknown function is passed.
		funcs = [expr]
		wavevecs = []
		powers = []
		other = [1] #TensMul printer does not like having no args
	else:
		raise TypeError(f"Don't know how to handle expression of type {type(expr)}")
	
	if len(funcs) != 1:
		raise ValueError("Expression must contain exactly one Function.")
	
	func = funcs[0]
	if not _depends_on_any(func, var_fourier):
		raise ValueError("The unknown function does not depend on the Fourier variable, so we cannot apply the derivative rule.")
	
	ret = func.xreplace({wavevec:rvec, mag:r})
	
	if len(powers) == 1:
		power = powers[0]
		assert power.args[0] == mag
		n = power.args[1]
		assert n.is_even
		
		if n < 0:
			for _ in range(abs(n)/2):
				ret = - InverseLaplacian(ret, r, rvec)
		else:
			for _ in range(abs(n)/2):
				ret = - Laplacian(ret, r, rvec)
	elif len(powers) > 1:
		raise NotImplementedError
	
	for w in wavevecs:
		i = w.indices[0]
		ret = I*PartialVectorDerivative(ret, rvec(-i), r)
	
	other = TensMul(*other).doit(deep=False)
	return ret*other

def ift_convolution(expr, vars_k, var_sum, var_r):
	r"""
	Given expr that depends on more than one Fourier variable (say \vec{p_i} for some integers i=1...n), assume \sum_i\vec{p_i} = \vec{k} and perform the inverse Fourier transform of Integral(expr, \vec{p_1}, ..., \vec{p_{n-1}}) from \vec{k}\to\vec{r}. None of the \vec{p_i} should be explicitly integrated over in the input expression. Note that this function will probably not be able to handle explicit appearance of \vec{k} in more complicated expressions.
	
	expr: sympy expression
	vars_k: list of the form [[TensorHead, Symbol],...], each element being a pair representing a wavevector and the symbol to use for its magnitude
	var_sum: [TensorHead, Symbol], representing the sum of the wavevectors in vars_k
	var_r: [TensorHead, Symbol], representing the spatial variable
	"""
	
	if isinstance(expr, (TensAdd, Add)):
		return expr.func(*[ift_convolution(arg, vars_k, var_sum, var_r) for arg in expr.args])
	
	assert isinstance(expr, (TensMul, Mul))
	for var in vars_k:
		_check_wavevector(var)
	_check_wavevector(var_r)
	_check_wavevector(var_sum)
	
	#Split terms of expr based on the functional dependence.
	parts_single = [[] for var in vars_k] #each element is a list of terms which depend on the corresponding element of vars_k
	part_sum = []
	other = []
	for arg in expr.args:
		dep_on_single = [_depends_on_any(arg, var) for var in vars_k]
		dep_on_sum = _depends_on_any(arg, var_sum)
		if np.sum(dep_on_single) + dep_on_sum > 1:
			raise RuntimeError(f"{arg = } depends on more than one wavevector")
		elif np.any(dep_on_single):
			w, = np.where(dep_on_single)
			assert len(w) == 1
			i = int(w[0])
			parts_single[i].append(arg)
		elif dep_on_sum:
			part_sum.append(arg)
		else:
			other.append(arg)
	
	parts_single = [expr.func(*l) for l in parts_single]
	part_sum = expr.func(*part_sum)
	other = expr.func(*other)
	
	ifts_single = [ift_derivative_rule(part, var, var_r) for part, var in zip(parts_single, vars_k)]
	
	dum = Function("_tmpfunction")(*var_sum)
	dum_ft = dum.xreplace({var_sum[i]:var_r[i] for i in range(len(var_sum))})
	
	ret = expr.func(other, ift_derivative_rule(expr.func(part_sum,dum).doit(deep=False), var_sum, var_r))
	assert dum_ft in ret.atoms(Function)
	ret = ret.xreplace({dum_ft: TensMul(*ifts_single).doit(deep=False)})
	
	return ret
