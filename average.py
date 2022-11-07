"""
TODO Implement
* Vectors
* Vector fields
* Unit vectors (probably just implement as annotation on vector (fields)), but note that I need to be able to differentiate these guys.
* Tensor fields
* Derivative object? Need to be able to represent curl, dive, grad etc. Check how much can be shared with sympy's inbuilt derivative object.
TODO: Think about how to implement summation convention. Just define a einsum(expr, index) function? IF I make the summation convention implicit, will probably be best to define a separate 'index' class like VEST does. Otherwise, interaction with other parts of sympy may become too messy. NOTE: whatever I do, need to make sure that e.g. (v_ib_i)^2 is interpreted as v_ib_iv_jb_j. NOTE: see discussion in https://github.com/sympy/sympy/issues/9284
TODO: function that prints an index-based expression in vector notation.
TODO: Think about how I would handle, say, a divergenceless vector field. I don't think the assumptions system works yet, so what would be the best way to do it? I don't think subs would work. Workaround could be to add 'divergenceless' and 'curlless' annotations to vector fields.
TODO: Function to replicate the 'ind=' functionality from VEST.
TODO: Something similar to VEST's FullSimplifyVectorForm
TODO: Figure out what exactly VEST's userFormSpec does. Sounds useful.
TODO: Check how much of Cadabra's functionality I can reuse. At least Kroenecker delta seems to be implemented there. https://cadabra.science/. AUR package available.
"""
import sympy

class average(sympy.Function):
	"""
	Represents a Reynolds average. First argument is the expression averaged, second argument is the set of symbols over which the average is done.
	TODO: Actually, the need to specify 'wrt' makes this unable to represent an ensemble average, I think! Problem? Or does that just mean we need to add a dummy 'realization label' to each stochastic field?
	"""
	nargs=2
	
	def _latex(self, printer=None):
		args = "\,".join([ printer.doprint(i) for i in self.args])
		return r"\left<{%s}\right>" % (args)
	
	@property
	def free_symbols(self):
		wrt = set(self.args[1]) #Make sure this is a set. For some reason, sympy seems to instantiate it as a finiteset. Not sure if that is a sympy bug.
		sym = self.args[0].free_symbols
		return {s for s in sym if s not in wrt}
	
	def _eval_simplify(self, **kwargs):
		from sympy.simplify.simplify import simplify
		"""
		TODO: Implement all Reynolds rules. Need to implement
		* some way to 'mark' a field as a fluctuation, such that all averages of it become zero. VEST basically assumes there is only a single spatial variable, and needs one to explicitly mark objects as constants at instantiation.
		* implement commutation with derivatives and integrals over independent variables.
		"""
		#TODO: Support a 'deep' argument like powsimp does.
		arg = self.args[0]
		wrt = self.args[1]
		
		if arg.func == sympy.core.add.Add:
			return sympy.core.add.Add(*[ simplify( average(i,wrt), **kwargs) for i in arg.args])
		elif arg.func == sympy.core.mul.Mul:
			inside = []
			outside = []
			for a in arg.args:
				if any([ s in  a.free_symbols for s in wrt ]):
					inside.append(a)
				#TODO: Do I need to worry about the case of bound symbols? check for that is any([ s in  a.atoms(sympy.Symbol) for s in wrt ])
				else:
					outside.append(a)
			outmulsimp = simplify(sympy.core.mul.Mul(*outside), **kwargs)
			if len(inside) > 0:
				return sympy.core.mul.Mul(outmulsimp, average(simplify(sympy.core.mul.Mul(*inside), **kwargs), wrt) )
			else:
				return outmulsimp
		elif arg.func == average:
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


def do_epsilon_delta(Expr, eps, delta):
	"""
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
		return sympy.core.mul.Mul(*newargs)
	elif Expr.func == sympy.core.add.Add or sympy.core.mul.Mul or sympy.tensor.tensor.TensAdd:
		return Expr.func(*[do_epsilon_delta(i) for i in Expr.args])
	else:
		return Expr

