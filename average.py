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

