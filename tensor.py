"""
TODO Implement
* Vector fields
* Unit vectors (probably just implement as annotation on vector (fields)), but note that I need to be able to differentiate these guys.
* Tensor fields
* Derivative object? Need to be able to represent curl, dive, grad etc. Check how much can be shared with sympy's inbuilt derivative object. Probably should be able to add a derivative method to the Tensor object.

TODO: function that prints an index-based expression in vector notation.
TODO: Function to replicate the 'ind=' functionality from VEST.
TODO: Something similar to VEST's FullSimplifyVectorForm
TODO: Figure out what exactly VEST's userFormSpec does. Sounds useful.
TODO: Check how much of Cadabra's functionality I can reuse. At least Kroenecker delta seems to be implemented there. https://cadabra.science/. AUR package available.
TODO: Can I somehow tell sympy to not worry about the covariant/contravariant distinction?

TODO: May need to define a custom Tensor or TensorHead that becomes a non-tensor when contracted by itself. Currently, you cannot do something like 1/( K(p)*K(-p) ), even though it is mathematically correct. OR figure out if there's a more graceful way to do it. Also would allow one to do something like ( K(p)*K(-p) ) **2 . Another way may be to implement a pow method for TensMul that raises an error if there are any free indices? Or perhaps one may try to override __pow__, __truediv__, and __rtruediv__ in TensExpr (check what TensAdd and TensMul define these methods as). NOTE: See https://github.com/sympy/sympy/issues/18435 for a discussion.
"""

import sympy
import sympy.tensor.tensor
import sympy.tensor.toperators

import warnings
import itertools
import scipy

def do_epsilon_delta(Expr, eps, delta):
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

def _gen_ind_combs(inds):
	"""
	Given a list 'inds' such that len(inds)%2==0, construct permutations of the indices that appear in the corresponding angular integral over len(inds) instances of a unit vector. The number of combinations is listed as 'distinct FICTs' in table 1 of [Kearsley, Fong 1975 - Linearly Independent Sets of Isotropic Cartesian Tensors of Ranks up to Eight]. I believe the formula for the number of elements returned is len(ind_combs) = double_factorial(len(inds)-1).
	
	For our purposes, we do not care about linear independence (see https://mathematica.stackexchange.com/questions/77855/finding-basis-of-isotropic-tensors-of-rank-n for the more complicated approach required in that case)
	"""
	ind_combs = []
	
	if len(inds) > 0:
		ind1 = inds[0]
		for i2, ind2 in enumerate(inds[1:], 1):
			remaining_inds = [inds[i] for i in range(len(inds)) if (i != 0 and i != i2)]
			
			if len(remaining_inds) > 0:
				for comb in _gen_ind_combs(remaining_inds):
					ind_combs.append( [(ind1,ind2)] + comb )
			else:
				ind_combs.append([(ind1,ind2)])
	
	return ind_combs

def _gen_delta_combs(inds, delta):
	"""
	Used in do_angular_integral() to generate combinations of the Kronecker delta that appear in the angular integral over unit vectors.
	"""
	ind_combs = _gen_ind_combs(inds)
	
	delta_combs = []
	for comb in ind_combs:
		this_delta = 1
		for i in range(len(comb)):
			this_delta *= delta(*comb[i])
		
		delta_combs.append(this_delta)
	
	return delta_combs

class UnevaluatedAngularIntegral(sympy.tensor.tensor.TensExpr):
	"""
	UnevaluatedAngularIntegral(expr, wavevec): angular integral wrt wavevec over expr
	
	expr: TensExpr
	wavevec: TensorHead
	"""
	def __new__(cls, expr, wavevec):
		"""
		expr: TensExpr
		wavevec: TensorHead, vector with respect to which the angular integral was done.
		"""
		obj = sympy.tensor.tensor.Basic.__new__(cls, expr, wavevec)
		obj.expr = expr
		obj.wavevec = wavevec
		return obj
	
	@property
	def coeff(self):
		return S.One

	@property
	def nocoeff(self):
		return self
	
	def get_indices(self):
		return self.expr.get_indices()
	
	def get_free_indices(self):
		return self.expr.get_free_indices()
	
	def _replace_indices(self, repl):
		return self.func(self.expr._replace_indices(repl), self.wavevec)
	
	def _latex(self, printer):
		expr = printer._print(self.args[0])
		wavevec = self.args[1].name
		return r"\int\mathrm{d}\Omega_{%s} \left(%s\right)" % (wavevec, expr)

def do_angular_integral(Expr, wavevec):
	"""
	Perform angular integrals over the vector wavevec.
	
	Arguments:
		Expr: sympy expression
		wavevec: sympy.tensor.tensor.TensorHead instance
	
	Returns:
		A sympy expression
	"""
	if isinstance(Expr, sympy.tensor.tensor.TensMul):
		wavevecs = []
		other = []
		
		for arg in Expr.args:
			if hasattr(arg, "component") and arg.component == wavevec:
				wavevecs.append(arg)
			else:
				other.append(arg)
				if arg.has(wavevec):
					raise RuntimeError("Please expand the input expression. Input was {}".format(Expr))
		
		prod_wavevecs = Expr.func(*wavevecs)
		inds = prod_wavevecs.get_indices()
		n = len(inds)
		delta = wavevec.index_types[0].delta #This is the Kronecker delta
		
		if n % 2 == 1:
			angint = 0
		else:
			if n == 0:
				angint = 4*sympy.pi
			elif n <= 6:
				p = sympy.symbols("p")
				prefactor = sympy.tensor.tensor.TensMul(*[ ( wavevec(p) * wavevec(-p) ).as_dummy()  for i in range(int(n/2))]) #multiply appropriate power of the wavenumber
				delta_combs = _gen_delta_combs(inds, delta)
				angint = 4*sympy.pi/int(scipy.special.factorial2(n+1)) * prefactor * sympy.tensor.tensor.TensAdd(*delta_combs)
			else:
				#TODO: I believe the above should work for any order, but am being a bit careful. I should think about this.
				angint = UnevaluatedAngularIntegral(prod_wavevecs, wavevec)
		
		newargs = other + [ angint ]
		return Expr.func(*newargs).doit()
	elif isinstance(Expr, (sympy.core.add.Add, sympy.tensor.tensor.TensAdd)):
		return Expr.func(*[do_angular_integral(i, wavevec) for i in Expr.args])
	else:
		return 4*sympy.pi*Expr

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
				scalarpart = Expr.coeff
				tensorpart = Expr/scalarpart
			else:
				warnings.warn("Could not find any tensor part of {}. Is this correct?".format(Expr), RuntimeWarning)
				scalarpart = Expr
				tensorpart = 1
			
			if scalarpart.has(wavevec.head):
				warnings.warn("Ignoring {} dependence in {}".format(wavevec, scalarpart), RuntimeWarning)
			
			ret += lowered_wavevec/ampl * tensorpart * sympy.Derivative(scalarpart, ampl)
		
		if indextype is not None:
			#NOTE: a separate call to contract_metric does not seem to be needed when we have already set the metric of the TensorIndexType to delta
			ret = ret.contract_delta(indextype.delta).contract_metric(indextype.metric)
		
		return ret

class dive_matcher():
	"""
	DEPRECATED: use replace
	
	Instantiates an object which, when called with an expression, returns True if the expression is a TensMul and it contains a wavevector dotted with a velocity.
	
	Arguments:
		wavevec: sympy.tensor.tensor.TensorHead instance
		velocity: sympy.tensor.tensor.TensorHead instance
	"""
	def __init__(self, wavevec, velocity):
		warnings.warn("DEPRECATED: Use mul_matcher instead of dive_matcher", RuntimeWarning)
		
		self.wavevec = wavevec
		self.velocity = velocity
	
	def __call__(self, Expr):
		"""
		Checks if Expr is a sympy.tensor.tensor.TensMul which contains K(p)*V(-p)
		
		Arguments:
			Expr: sympy.tensor.tensor.TensMul instance
		"""
		if isinstance(Expr, sympy.tensor.tensor.TensMul):
			wavevecs = [ arg for arg in Expr.args if hasattr(arg, "component") and arg.component == self.wavevec ]
			velocities = [ arg for arg in Expr.args if hasattr(arg, "component") and arg.component == self.velocity ]
			
			for v in velocities:
				for w in wavevecs:
					if w.get_indices()[0] == - v.get_indices()[0]:
						return True
		
		return False

def get_symmetries(tens):
	"""
	Given a tensor, return a list giving all its index-permuted forms that are compatible with its symmetries.
	
	Arguments:
		tens: sympy.tensor.tensor.Tensor instance
	
	Returns:
		List of sympy.tensor.tensor.Tensor instances
	"""
	def permute(tens):
		n = tens.rank
		comp = tens.components[0]
		gens = comp.symmetry.generators #Assuming that if this is a TensMul, it has at most one tensor.
		inds = tens.get_indices()
		
		tens_perms = []
		for gen in gens:
			perm = [ gen.apply(i) for i in range(n) ]
			sign = (-1)**(gen.apply(n) - n)
			tens_perms.append( sign * tens.coeff * comp(*[inds[i] for i in perm]) )
		
		return tens_perms
	
	old_perms = None
	new_perms = set([tens])
	while new_perms != old_perms:
		old_perms = new_perms.copy()
		for te in old_perms:
			for perm in permute(te):
				new_perms.add(perm)
	
	return new_perms

def flip_dummies(*args, **kwargs):
	"""
	DEPRECATED: no longer needed, I think.
	Given the arguments of a TensMul, construct all possible TensMuls with dummies flipped e.g. K(p) * V(-p) -> K(-p) * V(p)
	
	TODO: In mul_matcher.replacer, might do
	for Ex in flip_dummies(Expr):
		for subset in itertools.combinations(Ex.args, self.r):
			...
	but I don't see any benefit. Need to find a test case that will benefit from this before actually doing it. RN, only seems to be needed if I don't do Expr.canon_bp() in replacer. Also note that we might as well make this function flip_dummies(Expr) in that case.
	"""
	Expr = sympy.tensor.tensor.TensMul(*args, **kwargs)
	inds = Expr.get_indices()
	dummy_pairs = []
	for i1, ind1 in enumerate(inds):
		for i2, ind2 in enumerate(inds[i1+1:], i1+1):
			if ind1 == - ind2:
				dummy_pairs.append((i1, i2))
	
	flipped = []
	for seq in itertools.product(*[(0,1)]*len(dummy_pairs)):
		new_inds = inds.copy()
		for i, pair in enumerate(dummy_pairs):
			if seq[i] == 1:
				new_inds[pair[0]], new_inds[pair[1]] = new_inds[pair[1]], new_inds[pair[0]]
		flipped.append( Expr._set_indices(*new_inds) )
	
	return flipped

def pull_out_metric(Expr):
	"""
	Given a TensMul, construct a TensMul with the free indices pulled out using the metric. E.g. K(p) * V(q) -> K(p_1) * V(q_1) * metric(p,-p_1) * metric(q, -q_1)
	"""
	inds = Expr.get_indices()
	
	dummies = []
	for i1, ind1 in enumerate(inds):
		for i2, ind2 in enumerate(inds[i1+1:], i1+1):
			if ind1 == - ind2:
				dummies.append(i1)
				dummies.append(i2)
	
	new_inds = inds.copy()
	metrics = 1
	for i in range(len(inds)):
		if i not in dummies:
			ind = inds[i]
			if ind.is_up:
				newdum = ind.func( ind.name + "_dum", ind.tensor_index_type )
				new_inds[i] = newdum
				metrics *= ind.tensor_index_type.metric(ind, -newdum)
			else:
				newdum = ind.func( ind.name + "_dum_2", ind.tensor_index_type )
				new_inds[i] = - newdum
				metrics *= ind.tensor_index_type.metric(ind, newdum)
	
	return metrics * Expr._set_indices(*new_inds)

class mul_matcher():
	"""
	DEPRECATED: use replace
	Given two TensMuls, check if one is a subset of the other.
	
	Arguments:
		query: sympy.tensor.tensor.TensMul instance
		replacement: any sympy object.
		debug: bool. Whether to print any debug output (Default: False)
	"""
	def __init__(self, query, replacement, debug=False):
		query = sympy.core.sympify(query)
		replacement = sympy.core.sympify(replacement)
		
		if not isinstance(query, sympy.tensor.tensor.TensMul):
			#At least self.r assumes we are given a TensMul
			raise TypeError("query is not a TensMul instance.")
		
		if hasattr(query, "canon_bp"):
			query = query.canon_bp()
		
		if hasattr(query, "get_free_indices"):
			free_to_wilds_dict = self._indices_to_wilds(query.get_indices())
			query = query.subs(free_to_wilds_dict)
			replacement = replacement.subs(free_to_wilds_dict)
		
		self.query = query
		self.repl = replacement
		self.r = len(query.args)
		self.debug = debug
		
		try:
			self.free_to_wilds_dict = free_to_wilds_dict
			self.wilds_to_free_dict = self._invert_dict(free_to_wilds_dict)
		except NameError:
			self.free_to_wilds_dict = {}
			self.wilds_to_free_dict = {}
		
		self.dprint(f"init: {query = }, {replacement = }, {self.free_to_wilds_dict}")
		
		#Generate every permutation of query (that is allowed by its index symmetries) and store it as a list.
		self.query_permutations = set()
		tensors_list = [] #List of tensors in the query
		query.replace(sympy.tensor.tensor.Tensor, lambda *x: tensors_list.append(sympy.tensor.tensor.Tensor(*x)))
		
		tensors_perm_list = [ get_symmetries(tens) for tens in tensors_list ]
		for tensor_perms in itertools.product(*tensors_perm_list):
			d = {}
			self.dprint(f"init: {tensor_perms = }")
			for i,j in zip(tensors_list, tensor_perms):
				self.dprint(f"init: {i = }, {j = }")
				if i.coeff.could_extract_minus_sign():
					d[-i] = -j
				else:
					d[i] = j
			self.dprint(f"init: {d = }")
			self.query_permutations.add( query.xreplace(d) )
	
	def dprint(self, *args, **kwargs):
		"""
		To print debug output
		"""
		if self.debug:
			print(*args, **kwargs)
	
	def _indices_to_wilds(self, indices):
		"""
		Given a list of indices, return a dictionary such that the indices are keys of this dictionary, with values given by Wilds
		"""
		ret = {}
		for i in indices:
			if i.is_up: #Preserve the information on co-/contra-variance.
				ret[i] = sympy.core.Wild(i.name + "_wild")
			else:
				ret[i] = - sympy.core.Wild(i.name + "_wild")
		return ret

	def _invert_dict(self, d):
		"""
		Invert the key-value association of a dictionary
		"""
		return {v: k for k, v in d.items()}
	
	def matcher(self, Expr):
		"""
		Heuristic matcher. May give false positives, but should never give false negatives. This is useful if we expect self.replacer to be an expensive function.
		"""
		if not isinstance(Expr, (sympy.tensor.tensor.TensMul, sympy.tensor.tensor.Tensor)):
			return False
		else:
			return True
	
	def replacer(self, Expr):
		self.dprint(f"{Expr = }, {self.query = }")
		if hasattr(Expr, "canon_bp") and len(Expr.components) > 0:
			Expr = Expr.canon_bp()
		
		for subset in itertools.combinations(Expr.args, self.r):
			sub_expr = Expr.func(*subset)
			for query in self.query_permutations:
				replaced, m = sub_expr.replace(query, self.repl, map=True)
				if len(m) > 0:
					self.dprint(f"replacer: {sub_expr.canon_bp() = }, {m = }")
					
					rest_args = [a for a in Expr.args if a not in subset]
					
					if len(rest_args) > 0:
						rest = Expr.func(*rest_args) #We assume the same argument cannot appear twice (I think sympy consolidates them and makes sure that they are not repeated).
						return Expr.func(replaced, self.replacer(rest)).subs(self.wilds_to_free_dict).doit()
					else:
						return replaced.subs(self.wilds_to_free_dict)
		
		#If we reached here, no exact matches were found, so return the expression unchanged.
		return Expr
	
	def __getitem__(self, key):
		"""
		To allow syntax like Expr.replace( *mul_matcher(query, replacement) )
		"""
		if key == 0:
			return self.matcher
		elif key == 1:
			return self.replacer
		else:
			raise IndexError

if __name__ == "__main__":
	sy = sympy
	
	Cartesian = sy.tensor.tensor.TensorIndexType('Cartesian', dim=3)
	Cartesian.set_metric(Cartesian.delta)
	p, q, r, s, t, u = sy.tensor.tensor.tensor_indices("p q r s t u", Cartesian)
	f, g = sympy.symbols('f g', cls=sympy.Function)
	
	K = sy.tensor.tensor.TensorHead("K", [Cartesian])
	k = sympy.symbols("K")
	V = sy.tensor.tensor.TensorHead("V", [Cartesian])
	
	print( ( - V(s)*V(-s) *  K(r) * K(-r) * V(q) * K(-p) * V(p) ).replace( *mul_matcher( K(r)*V(-r), 0, debug=True ) ) )
