"""
TODO Implement
* Derivative object? Need to be able to represent curl, dive, grad etc. Check how much can be shared with sympy's inbuilt derivative object. Probably should be able to add a derivative method to the Tensor object.

TODO: function that prints an index-based expression in vector notation.
TODO: Function to replicate the 'ind=' functionality from VEST.
TODO: Something similar to VEST's FullSimplifyVectorForm
TODO: Figure out what exactly VEST's userFormSpec does. Sounds useful.

"""

import sympy
from sympy.core.function import (
	UndefinedFunction,
	AppliedUndef,
	)
import sympy.tensor.tensor
from sympy.tensor.tensor import (
	TensExpr,
	)
import sympy.tensor.toperators

import abc
import warnings
import itertools

from .rewrite import replace_repeat

class PartialDiffScalarWarning(RuntimeWarning): pass

def replace_by_ampl(expr, wavevec, ampl):
	"""
	expr: TensExpr
	wavevec: TensorHead
	ampl: Symbol
	
	Helper to call replace on expr and replace wavevec(p)*wavevec(-p) by ampl**2
	
	Also see replace_by_ampl_optimized
	"""
	a = sympy.tensor.tensor.WildTensorIndex(True, wavevec.index_types[0], ignore_updown=True)
	w = sympy.Wild('w')
	W = sympy.tensor.tensor.WildTensorHead('W')
	expr = replace_repeat(expr, w*W()*wavevec(a)*wavevec(-a), w*W()*ampl**2)
	return expr

def replace_by_ampl_optimized(expr, wavevec, ampl):
	"""
	expr: TensExpr
	wavevec: TensorHead
	ampl: Symbol
	
	Optimized version of replace_by_ampl (helper to call replace on expr and replace wavevec(p)*wavevec(-p) by ampl**2)
	"""
	return expr.replace(sympy.tensor.tensor.TensMul, lambda *args: _replace_by_ampl_for_mul(args, wavevec, ampl))

def _replace_by_ampl_for_mul(args, wavevec, ampl):
	"""
	args: list, args of a TensMul
	wavevec: TensorHead
	ampl: Symbol
	
	To be used as (...).replace(TensMul, lambda *args: _replace_by_ampl_for_mul(args, wavevec, ampl))
	See replace_by_ampl_optimized for a wrapper.
	
	The hope is that by reducing unnecessary walking of the expression tree, this will be more efficient than replace_by_ampl for huge expressions (it seems that replace_by_ampl is very slow on TensAdd instances which have many terms).
	"""
	expr = sympy.tensor.tensor.TensMul(*args).doit(deep=False)
	if wavevec not in expr.atoms(sympy.tensor.tensor.TensorHead):
		return expr
	
	a = sympy.tensor.tensor.WildTensorIndex(True, wavevec.index_types[0], ignore_updown=True)
	w = sympy.Wild('w')
	W = sympy.tensor.tensor.WildTensorHead('W')
	expr = replace_repeat(expr, w*W()*wavevec(a)*wavevec(-a), w*W()*ampl**2)
	return expr

def do_epsilon_delta(expr, eps, delta):
	"""
	expr: TensExpr
	eps: TensorHead, TensorIndexType.epsilon
	delta: TensorHead, , TensorIndexType.delta
	
	Perform epsilon-delta replacement by using expr.replace.
	"""
	def _eps_delta_for_mul(args, eps, delta):
		"""
		args: list, args of a TensMul
		eps: TensorHead
		delta: TensorHead
		"""
		expr = sympy.tensor.tensor.TensMul(*args).doit(deep=False)
		if eps not in expr.atoms(sympy.tensor.tensor.TensorHead):
			return expr
		
		index_type = eps.index_types[0]
		_a = sympy.tensor.tensor.WildTensorIndex(True, index_type, ignore_updown=True)
		_b = sympy.tensor.tensor.WildTensorIndex(True, index_type, ignore_updown=True)
		_c = sympy.tensor.tensor.WildTensorIndex(True, index_type, ignore_updown=True)
		_d = sympy.tensor.tensor.WildTensorIndex(True, index_type, ignore_updown=True)
		_e = sympy.tensor.tensor.WildTensorIndex(True, index_type, ignore_updown=True)
		w = sympy.Wild('w')
		W = sympy.tensor.tensor.WildTensorHead('W')
		expr = replace_repeat(
			expr,
			w*W()*eps(_a, _b, _c)*eps(-_a, _d, _e),
			w*W()*(delta(_b,_d)*delta(_c,_e) - delta(_b,_e)*delta(_d,_c)),
			)
		return expr
	
	return expr.replace(sympy.tensor.tensor.TensMul, lambda *args: _eps_delta_for_mul(args, eps, delta))

def _do_epsilon_delta(Expr, eps, delta):
	"""
	DEPRECATED: use replace
	
	Performs the replacement corresponding to the epsilon-delta identity.
	
	Arguments:
		Expr: any sympy expression (e.g. TensAdd, Add, Mul, TensMul)
		eps: .epsilon method of a sympy.tensor.tensor.TensorIndexType instance
		delta: .delta method of a sympy.tensor.tensor.TensorIndexType instance
	
	Returns:
		a sympy expression with epsilon-delta replacements performed.
	"""
	if Expr.func == sympy.tensor.tensor.TensMul:
		epsilons = []
		other = []
		
		for arg in Expr.args:
			if hasattr(arg, "component") and arg.component == eps:
				epsilons.append(arg)
			else:
				other.append(arg)
		
		for i1, e1 in enumerate(epsilons):
			for i2, e2 in enumerate(epsilons[i1+1:], start=i1+1):
				if e1 != 1 and e2 != 1: #1 is the placeholder we use to denote eliminated epsilons
					inds1 = e1.get_indices()
					inds2 = e2.get_indices()
					common_inds = [ind for ind in inds1 if -ind in inds2] #Get indices which are in both the epsilons
					if len(common_inds) > 0:
						to_elim = common_inds[0]
						other_indices = (
							[ind for ind in inds1 if ind != to_elim],
							[ind for ind in inds2 if ind != -to_elim]
							)
						
						prefactor = sympy.tensor.tensor.TensMul(e1, e2).canon_bp().coeff #Account for the ordering of indices. Should be +1 or -1
						
						#Make the epsilon-delta replacement. e1 will never be looped over after this, so we can just replace it by the required combination of deltas.
						epsilons[i2] = 1
						epsilons[i1] = prefactor * (
							delta(other_indices[0][0], other_indices[1][0]) * delta(other_indices[0][1], other_indices[1][1])
							- delta(other_indices[0][0], other_indices[1][1]) * delta(other_indices[0][1], other_indices[1][0])
							)
						
						break #We have found the epsilon that is contracted with this e1
		
		newargs = other + epsilons
		return Expr.func(*newargs)
	elif Expr.func == sympy.core.add.Add or sympy.core.mul.Mul or sympy.tensor.tensor.TensAdd:
		return Expr.func(*[do_epsilon_delta(i) for i in Expr.args])
	else:
		return Expr

def partialdiff(Expr, wavevec, ampl=None):
	"""
	Take partial derivative of a tensor expression with respect to a tensor. If the expression contains scalar functions dependent on the amplitude of the wavevector, you should specify that symbol as ampl
	
	Arguments:
		Expr: an instance of sympy.tensor.tensor.TensExpr
		wavevec: an instance of sympy.tensor.tensor.Tensor
		ampl: an instance of sympy.core.symbol.Symbol.
	
	Returns:
		ret: an instance of sympy.tensor.tensor.TensExpr
	"""
	indextype = wavevec.index_types[0]
	Expr = Expr.expand()
	
	if isinstance(Expr, sympy.tensor.tensor.TensAdd) or isinstance(Expr, sympy.core.add.Add):
		return Expr.func(*[ partialdiff(arg, wavevec, ampl=ampl) for arg in Expr.args ])
	else:
		ret = sympy.tensor.toperators.PartialDerivative( Expr, wavevec )
		ret = ret._perform_derivative()
		
		if ampl is not None:
			if indextype is None:
				raise TypeError("indextype needs to be specified to make use of ampl.")
			if len(wavevec.indices) > 1:
				raise NotImplementedError("Unsure how to define amplitude for tensor with more than one index.")
			
			lowered_wavevec = wavevec.head(- wavevec.indices[0] )
			
			if isinstance(Expr, sympy.tensor.tensor.TensMul):
				Expr = Expr.doit(deep=False) #make sure coeff is correctly populated.
				scalarpart = Expr.coeff
				tensorpart = Expr/scalarpart
			else:
				warnings.warn("Could not find any tensor part of {}. Is this correct?".format(Expr), PartialDiffScalarWarning)
				scalarpart = Expr
				tensorpart = sympy.S(1)
			
			if scalarpart.has(wavevec.head):
				warnings.warn("Ignoring {} dependence in {}".format(wavevec, scalarpart), PartialDiffScalarWarning)
			
			ret += lowered_wavevec/ampl * tensorpart * sympy.Derivative(scalarpart, ampl).doit(deep=False)
			
			if tensorpart.has(ampl):
				raise RuntimeError(f"Expanding the given expression failed; the non-coeff part depends on {ampl}.")
		
		if indextype is not None:
			ret = contract_delta(ret, indextype.delta)
			ret = sympy.tensor.tensor.contract_metric(ret, indextype.metric)
		
		return ret

class PartialVectorDerivative(sympy.tensor.tensor.TensExpr):
	"""
	Unevaluated form of partialdiff
	"""
	def __new__(cls, expr, wavevec, ampl, replace_indices=True):
		"""
		expr: TensExpr
		wavevec: Tensor, vector with respect to which the partial derivative is taken
		ampl: Symbol, variable representing magnitude of wavevec
		"""
		args, indices, free, dum = PartialVectorDerivative._contract_indices_for_derivative(sympy.S(expr), [wavevec], replace_indices=replace_indices)
		
		obj = sympy.tensor.tensor.Basic.__new__(cls, *args, ampl)
		
		obj._indices = indices
		obj._free = free
		obj._dum = dum
		obj._ext_rank = len(free) + 2*len(dum)
		
		return obj
	
	@property
	def expr(self):
		return self.args[0]
	
	@property
	def wavevec(self):
		return self.args[1]
	
	@property
	def ampl(self):
		return self.args[2]
	
	@property
	def variables(self):
		return [self.wavevec]
	
	@property
	def coeff(self):
		return sympy.S.One
	
	@property
	def nocoeff(self):
		return self
	
	@property
	def ext_rank(self):
		return self._ext_rank
	
	
	def get_indices(self):
		return self._indices
	
	def get_free_indices(self):
		free = sorted(self._free, key=lambda x: x[1])
		return [i[0] for i in free]
	
	def _replace_indices(self, repl):
		if hasattr(self.expr, "_replace_indices"):
			expr = self.expr._replace_indices(repl)
		else:
			expr = self.expr
		mirrored = {-k: -v for k, v in repl.items()}
		variables = [i.xreplace(mirrored) for i in self.variables]
		return self.func(expr, *variables, self.ampl, replace_indices=False)
	
	def _set_indices(self, *indices):
		repl = dict(zip(self.get_indices(), indices))
		return self._replace_indices(repl)
	
	
	def doit(self, **hints):
		deep = hints.get('deep', True)
		if deep:
			expr = self.expr.doit(**hints)
		else:
			expr = self.expr
		
		ret = partialdiff(expr, self.wavevec, self.ampl)
		ret = replace_by_ampl_optimized(ret, self.wavevec.component, self.ampl)
		ret = ret.as_dummy().expand() #to allow cancellation of terms
		
		if isinstance(ret, sympy.tensor.tensor.TensAdd):
			ret = ret.doit(deep=False)
		return ret
	
	@classmethod
	def _contract_indices_for_derivative(cls, expr, variables, replace_indices=True):
		"""
		Copied from PartialDerivative, but set replace_indices to False
		"""
		variables_opposite_valence = []
		
		for i in variables:
			if isinstance(i, sympy.tensor.tensor.Tensor):
				i_free_indices = i.get_free_indices()
				variables_opposite_valence.append(
						i.xreplace({k: -k for k in i_free_indices}))
			elif isinstance(i, sympy.Symbol):
				variables_opposite_valence.append(i)
		
		args, indices, free, dum = sympy.tensor.tensor.TensMul._tensMul_contract_indices(
			[expr] + variables_opposite_valence, replace_indices=replace_indices)
		
		for i in range(1, len(args)):
			args_i = args[i]
			if isinstance(args_i, sympy.tensor.tensor.Tensor):
				i_indices = args[i].get_free_indices()
				args[i] = args[i].xreplace({k: -k for k in i_indices})
		
		return args, indices, free, dum

def contract_delta(t, delta):
	if isinstance(t, sympy.tensor.tensor.TensExpr):
		return t.contract_delta(delta)
	else:
		return t

class _ScalarTensExpr(TensExpr):
	def __init__(self, *args):
		for arg in args:
			if isinstance(arg, TensExpr) and len(arg.get_free_indices()) > 0:
				raise ValueError("arguments of ScalarTensExpr cannot have free indices")
	
	@property
	def nocoeff(self):
		return self
	
	@property
	def coeff(self):
		return sympy.S.One
	
	def get_indices(self):
		indices = []
		for arg in self.args:
			if isinstance(arg, TensExpr):
				indices.extend([i for i in arg.get_indices() if i not in indices])
		return indices
	
	def get_free_indices(self):
		return set()
	
	def _replace_indices(self, repl):
		#TODO: perhaps makes sense to just keep this as a dummy method, since the args should not have any free indices anyway.
		return self.xreplace(repl)
	
	def substitute_indices(self, *index_tuples):
		#Just a dummy method, since the arguments are not allowed to have any free indices.
		return self
	
	def _extract_data(self, replacement_dict):
		data = []
		for arg in self.args:
			if isinstance(arg, TensExpr):
				data.append(arg._extract_data(replacement_dict))
			else:
				data.append([[], arg])
		args_indices, args_arrays = zip(*data)
		
		assert all(len(args_indices[i]) == 0 for i in range(len(self.args)))
		
		return [], self.func(*args_arrays).doit(deep=False)
	
	def doit(self, **hints):
		deep = hints.get('deep', True)
		if deep:
			args = [arg.doit(**hints) for arg in self.args]
		else:
			args = self.args
		
		if any(isinstance(arg, TensExpr) for arg in args):
			return self.func(*self.args)
		else:
			fun = sympy.Function(self.name)
			return fun(*self.args)
	
	@property
	def ext_rank(self):
		return len(self.get_indices())
	
	def _set_indices(self, *indices):
		# copied from TensMul
		if len(indices) != self.ext_rank:
			raise ValueError("indices length mismatch")
		args = list(self.args)
		pos = 0
		for i, arg in enumerate(args):
			if isinstance(arg, TensExpr):
				ext_rank = arg.ext_rank
				args[i] = arg._set_indices(*indices[pos:pos+ext_rank])
				pos += ext_rank
		return self.func(*args)
	
	# def commutes_with(self, other):
	# 	all_commute = True
	# 	for arg in self.args:
	# 		#TODO: problem here: commutes_with is only defined for Tensor (not even TensMul). Perhaps check on all elements of arg.atoms(TensorHead)?
	# 		if isinstance(arg, TensExpr) and arg.commutes_with(other) != 0:
	# 			all_commute = False
	# 	if all_commute:
	# 		return 0
	# 	else:
	# 		#Not really safe to say anything here.
	# 		return None

class FunctionOfTensor(
	UndefinedFunction,
	abc.ABCMeta, #TensExpr inherits from abc.ABC
	):
	"""
	Subclass of UndefinedFunction that represents a scalar function of a scalar TensExpr.
	"""
	def __new__(mcl, name, **kwargs):
		return super().__new__(mcl, name, (AppliedUndef, _ScalarTensExpr), {}, **kwargs)

if __name__ == "__main__":
	sy = sympy
	
	Cartesian = sy.tensor.tensor.TensorIndexType('Cartesian', dim=3)
	Cartesian.set_metric(Cartesian.delta)
	p, q, r, s, t, u = sy.tensor.tensor.tensor_indices("p q r s t u", Cartesian)
	f, g = sympy.symbols('f g', cls=sympy.Function)
	
	K = sy.tensor.tensor.TensorHead("K", [Cartesian])
	k = sympy.symbols("K")
	V = sy.tensor.tensor.TensorHead("V", [Cartesian])
	
