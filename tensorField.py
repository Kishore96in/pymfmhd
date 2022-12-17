import sympy
import sympy.tensor.tensor
from sympy import Basic, Symbol, Tuple, S
from sympy.tensor.tensor import TensorHead, Tensor, TensorSymmetry, TensorManager, _IndexStructure

class TensorFieldHead(TensorHead):
	"""
	name: str
	index_types: list of TensorIndexType
	positions: list
	"""
	def __new__(cls, name, index_types, positions, symmetry=None, comm=0):
		if isinstance(name, str):
			name_symbol = Symbol(name)
		elif isinstance(name, Symbol):
			name_symbol = name
		else:
			raise ValueError("invalid name")

		if symmetry is None:
			symmetry = TensorSymmetry.no_symmetry(len(index_types))
		else:
			assert symmetry.rank == len(index_types)

		obj = Basic.__new__(cls, name_symbol, Tuple(*index_types), symmetry, Tuple(*positions))
		obj.comm = TensorManager.comm_symbols2i(comm)
		return obj
	
	@property
	def positions(self):
		return self.args[3]
	
	def _print(self):
		return '%s(%s;%s)' %(self.name, ','.join([str(x) for x in self.index_types]), ','.join([str(x) for x in self.positions]))
	
	def __call__(self, *indices, **kw_args):
		"""
		indices: list of TensorIndex
		pos (kwarg): override positions set in TensorFieldHead
		"""
		pos = kw_args.pop("pos", self.positions)
		return TensorField(self, indices, pos, **kw_args)

class TensorField(Tensor):
	def __new__(cls, tensor_head, indices, positions, is_canon_bp=False, **kw_args):
		indices = cls._parse_indices(tensor_head, indices)
		obj = Basic.__new__(cls, tensor_head, Tuple(*indices), Tuple(*positions))
		obj._index_structure = _IndexStructure.from_indices(*indices)
		obj._free = obj._index_structure.free[:]
		obj._dum = obj._index_structure.dum[:]
		obj._ext_rank = obj._index_structure._ext_rank
		obj._coeff = S.One
		obj._nocoeff = obj
		obj._component = tensor_head
		obj._components = [tensor_head]
		if tensor_head.rank != len(indices):
			raise ValueError("wrong number of indices")
		obj.is_canon_bp = is_canon_bp
		obj._index_map = Tensor._build_index_map(indices, obj._index_structure)
		return obj
	
	@property
	def positions(self):
		return self.args[2]
	
	def _print(self):
		return '%s(%s;%s)' %(self.component.name, ','.join([str(x) for x in self.indices]), ','.join([str(x) for x in self.positions]))

if __name__ == "__main__":
	from sympy.tensor.tensor import TensorIndexType, WildTensorIndex, TensorIndex
	from sympy import symbols
	
	R3 = TensorIndexType('R3', dim=3)
	R3.set_metric(R3.delta)
	delta = R3.delta
	eps = R3.epsilon
	
	p, q, r, s, t, u = symbols("p q r s t u", cls=TensorIndex, tensor_index_type=R3)
	a,b,c,d,e,f = symbols("a b c d e f", cls = WildTensorIndex, tensor_index_type=R3, ignore_updown=True)
	
	K = TensorHead("K", [R3])
	B = TensorFieldHead("B", [R3], positions=[K])
	
	print( B(p) )
