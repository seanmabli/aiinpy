import numpy as np

class tensor:
	def __init__(self, data, ops=[], dtype=None, marker=None):
		self.data = (data if dtype == None else data.astype(dtype)) if type(data) == np.ndarray else np.array(data, dtype=dtype)
		self.dtype = self.data.dtype if dtype == None else dtype
		self.ops = ['init', [], self.data] if ops == [] else ops
		self.shape = self.data.shape

	def toTensor(self, other): return tensor(other) if type(other) != tensor else other

	def __add__(self, other): other = self.toTensor(other); return tensor(self.data + other.data, ops=['add', self.ops, other.ops, self.data + other.data])
	def __mul__(self, other): other = self.toTensor(other); return tensor(self.data * other.data, ops=['mul', self.ops, other.ops, self.data * other.data])
	def __sub__(self, other): other = self.toTensor(other); return tensor(self.data - other.data, ops=['add', self.ops, ['mul', other.ops, ['init', [], tensor(-1)]], self.data - other.data])
	def __truediv__(self, other): other = self.toTensor(other); return tensor(self.data / other.data, ops=['mul', self.ops, ['pow', other.ops, ['init', [], tensor(-1)]], self.data / other.data])
	def __pow__(self, other): other = self.toTensor(other); return tensor(self.data ** other.data, ops=['pow', self.ops, other.ops, self.data ** other.data])
	def __mod__(self, other): other = self.toTensor(other); return tensor(self.data % other.data, ops=['mod', self.ops, other.ops, self.data % other.data])
	def __matmul__(self, other): other = self.toTensor(other); return tensor(self.data @ other.data, ops=['matmul', self.ops, other.ops, self.data @ other.data])
	def __neg__(self): return tensor(-self.data, ops=['mul', self.ops, ['init', tensor(-1), self.dtype], -self.data])
	def __abs__(self): return tensor(abs(self.data), ops=['abs', self.ops, abs(self.data)])

	def __radd__(self, other): return self + other
	def __rmul__(self, other): return self * other
	def __rsub__(self, other): return self - other
	def __rtruediv__(self, other): return self / other
	def __rpow__(self, other): return self ** other
	def __rmod__(self, other): return self % other
	def __rmatmul__(self, other): return self @ other

	def reshape(self, shape, other=None): return tensor(other.data.reshape(shape), ops=['reshape', other.ops, shape]) if other != None else tensor(self.data.reshape(shape), ops=['reshape', self.ops, shape])
	def exp(self, other=None): return tensor(np.exp(other.data), ops=['pow', other.ops, ['init', tensor(np.e), other.dtype]]) if other != None else tensor(np.exp(self.data), ops=['pow', self.ops, ['init', tensor(np.e), self.dtype]])

	def zeros(self, shape): return tensor(np.zeros(shape))
	def ones(self, shape): return tensor(np.ones(shape))
	def random(self, shape): return tensor(np.random.random(shape))
	def uniform(self, low, high, shape): return tensor(np.random.uniform(low, high, shape))
	def random_binomial(self, shape, p): return tensor(np.random.binomial(1, p, shape))
	def full(self, value, shape): return tensor(np.full(shape, value))

	def __repr__(self): return 'tensor(' + str(self.data) + ')'
	def __getitem__(self, key): return self.data[key]
	def __setitem__(self, key, value): self.data[key] = value
	def __len__(self): return len(self.data)
	def __iter__(self): return iter(self.data)
	def __copy__(self): return tensor(self.data.copy())
	def __eq__(self, other) -> bool: return self.data.all() == other.data.all() if type(other) == tensor else self.data.all() == other
	def __ne__(self, other) -> bool: return self.data.all() != other.data.all() if type(other) == tensor else self.data.all() != other
	def mean(self, axis=None): return np.mean(self.data, axis=axis)
	def sum(self, axis=None): return np.sum(self.data, axis=axis)
	def max(self, axis=None): return np.max(self.data, axis=axis)
	def min(self, axis=None): return np.min(self.data, axis=axis)

	def vectorize(self, function): return tensor(np.vectorize(function)(self.data))
	def where(self, condition, x, y): return tensor(np.where(condition, x, y))

	'''
	def autograd(self, input):
		if self.ops[0] == 'init': return tensor(input\)
		elif self.ops[0] == 'add': return self.autograd(tensor(self, ops=self.ops[1])) + self.autograd(tensor(self, ops=self.ops[2]))
		else: print(self.ops[0])
		# if self.ops[0] == 'mul': return self.autograd(tensor(self, ops=self.ops[1])) * self.ops[3] + self.ops[3] * self.autograd(tensor(self, ops=self.ops[2]))
	'''