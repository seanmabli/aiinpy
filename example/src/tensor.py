import numpy as np

class tensor:
	def __init__(self, data, ops=[], dtype=None, marker=None):
		self.data = (data if dtype == None else data.astype(dtype)) if type(data) == np.ndarray else np.array(data, dtype=dtype)
		self.dtype = self.data.dtype if dtype == None else dtype
		self.ops = ['init', [], self.data] if ops == [] else ops
		self.shape = self.data.shape
	
	newaxis = np.newaxis

	def toTensor(other): return tensor(other) if type(other) != tensor else other
	def reshape(self, shape): return tensor(self.data.reshape(shape), ops=['reshape', self.ops, shape])
	def __int__(self): return int(self.data)
	def __float__(self): return float(self.data)
	def __bool__(self): return bool(self.data)
	def astype(self, dtype): return tensor(self.data.astype(dtype), ops=['astype', self.ops, dtype])

	def __add__(self, other): other = tensor.toTensor(other); return tensor(self.data + other.data, ops=['add', self.ops, other.ops, self.data + other.data])
	def __mul__(self, other): other = tensor.toTensor(other); return tensor(self.data * other.data, ops=['mul', self.ops, other.ops, self.data * other.data])
	def __sub__(self, other): other = tensor.toTensor(other); return tensor(self.data - other.data, ops=['add', self.ops, ['mul', other.ops, ['init', [], tensor(-1)]], self.data - other.data])
	def __truediv__(self, other): other = tensor.toTensor(other); return tensor(self.data / other.data, ops=['mul', self.ops, ['pow', other.ops, ['init', [], tensor(-1)]], self.data / other.data])
	def __pow__(self, other): other = tensor.toTensor(other); return tensor(self.data ** other.data, ops=['pow', self.ops, other.ops, self.data ** other.data])
	def __mod__(self, other): other = tensor.toTensor(other); return tensor(self.data % other.data, ops=['mod', self.ops, other.ops, self.data % other.data])
	def __matmul__(self, other): other = tensor.toTensor(other); return tensor(self.data @ other.data, ops=['matmul', self.ops, other.ops, self.data @ other.data])
	def __neg__(self): return tensor(-self.data, ops=['mul', self.ops, ['init', tensor(-1), self.dtype], -self.data])
	def __abs__(self): return tensor(abs(self.data), ops=['abs', self.ops, abs(self.data)])

	def __radd__(self, other): return self + tensor.toTensor(other)
	def __rmul__(self, other): return self * tensor.toTensor(other)
	def __rsub__(self, other): return self - tensor.toTensor(other)
	def __rtruediv__(self, other): return self / tensor.toTensor(other)
	def __rpow__(self, other): return self ** tensor.toTensor(other)
	def __rmod__(self, other): return self % tensor.toTensor(other)
	def __rmatmul__(self, other): return self @ tensor.toTensor(other)

	def zeros(shape): return tensor(np.zeros(shape))
	def ones(shape): return tensor(np.ones(shape))
	def random(shape): return tensor(np.random.random(shape))
	def uniform(low=0, high=1, shape=1): return tensor(np.random.uniform(low, high, shape))
	def random_binomial(shape, p): return tensor(np.random.binomial(1, p, shape))
	def full(value, shape): return tensor(np.full(shape, value))

	def __repr__(self): return 'tensor(' + str(self.data) + ')'
	def __getitem__(self, key): return self.data[key]
	def __setitem__(self, key, value): self.data[key] = value
	def __len__(self): return len(self.data)
	def __iter__(self): return iter(self.data)
	def __copy__(self): return tensor(self.data.copy())
	def __eq__(self, other): return tensor(self.data == other.data)
	def __ne__(self, other): return tensor(self.data != other.data)
	def mean(self, axis=None): return tensor(np.mean(self.data, axis=axis))
	def sum(self, axis=None): return tensor(np.sum(self.data, axis=axis))
	def max(self, axis=None): return np.max(self.data, axis=axis)
	def min(self, axis=None): return np.min(self.data, axis=axis)

	# make the following 4 functions return the input type
	def exp(other): other = tensor.toTensor(other); return tensor(np.exp(other.data), ops=['pow', other.ops, ['init', tensor(np.e), other.dtype]])
	def floor(other): other = tensor.toTensor(other); return tensor(np.floor(other.data), ops=['floor', other.ops])
	def ceil(other): other = tensor.toTensor(other); return tensor(np.ceil(other.data), ops=['ceil', other.ops])
	def prod(other): other = tensor.toTensor(other); return np.prod(other.data)

	def maximum(a, b): return np.maximum(tensor.toTensor(a).data, tensor.toTensor(b).data)
	def minimum(a, b): return np.minimum(tensor.toTensor(a).data, tensor.toTensor(b).data)
	def outer(a, b): return tensor(np.outer(tensor.toTensor(a).data, tensor.toTensor(b).data))
	def inner(a, b): return tensor(np.inner(tensor.toTensor(a).data, tensor.toTensor(b).data))
	def transpose(other, axes=None): other = tensor.toTensor(other); return tensor(np.transpose(other.data, axes), ops=['tbd'])

	# bad functions (try to remove)
	def vectorize(function): return np.vectorize(function)
	def where(condition, x, y): return tensor(np.where(condition, x, y))
	def concat(tensors, axis=0): return tensor(np.concatenate(list(map(lambda i: i.data, tensors)), axis=axis))
	def repeat(other, repeats, axis=None): other = tensor.toTensor(other); return tensor(np.repeat(other.data, repeats, axis=axis))
	def rot90(other, repeats): other = tensor.toTensor(other); return tensor(np.rot90(other.data, repeats), ops=['tbd'])
	def pad(other, pad_width, mode='constant'): other = tensor.toTensor(other); return tensor(np.pad(other.data, pad_width, mode=mode), ops=['tbd'])
	def index(other, value): other = tensor.toTensor(other); return np.where(other.data == value)
	def clip(other, min, max): other = tensor.toTensor(other); return tensor(np.clip(other.data, min, max), ops=['tbd'])

	'''
	def autograd(self, input):
		if self.ops[0] == 'init': return tensor(input\)
		elif self.ops[0] == 'add': return self.autograd(tensor(self, ops=self.ops[1])) + self.autograd(tensor(self, ops=self.ops[2]))
		else: print(self.ops[0])
		# if self.ops[0] == 'mul': return self.autograd(tensor(self, ops=self.ops[1])) * self.ops[3] + self.ops[3] * self.autograd(tensor(self, ops=self.ops[2]))
	'''