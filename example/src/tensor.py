import numpy as np

class tensor:
	def __init__(self, data, dtype=None):
		self.data = (data if dtype == None else data.astype(dtype)) if type(data) == np.ndarray else np.array(data, dtype=dtype)
		self.dtype = self.data.dtype if dtype == None else dtype
		self.ops = [('init', self.data, self.dtype)]
		self.shape = self.data.shape

	def __add__(self, other): other = self.ops.append(('add', other)); return tensor(self.data + other.data) if type(other) == tensor else tensor(self.data + other)
	def __mul__(self, other): other = self.ops.append(('mul', other)); return tensor(self.data * other.data) if type(other) == tensor else tensor(self.data * other)
	def __sub__(self, other): other = self.ops.append(('add', -other)); return tensor(self.data - other.data) if type(other) == tensor else tensor(self.data - other)
	def __truediv__(self, other): other = self.ops.append(('mul', 1 / other)); return tensor(self.data / other.data) if type(other) == tensor else tensor(self.data / other)
	def __pow__(self, other): other = self.ops.append(('pow', other)); return tensor(self.data ** other.data) if type(other) == tensor else tensor(self.data ** other)
	def __mod__(self, other): other = self.ops.append(('mod', other)); return tensor(self.data % other.data) if type(other) == tensor else tensor(self.data % other)
	def __matmul__(self, other): other = self.ops.append(('matmul', other)); return tensor(self.data @ other.data) if type(other) == tensor else tensor(self.data @ other)
	def __neg__(self): other = self.ops.append(('mul', tensor(-1))); return tensor(-self.data)
	def reshape(self, shape): other = self.ops.append(('reshape', shape)); return tensor(self.data.reshape(shape))
	def marker(self, marker:str): other = self.ops.append(('marker', marker)); return self

	def zeros(self, shape): return tensor(np.zeros(shape))
	def ones(self, shape): return tensor(np.ones(shape))
	def random(self, shape): return tensor(np.random.random(shape))
	def uniform(self, low, high, shape): return tensor(np.random.uniform(low, high, shape))
	def full(self, value, shape): return tensor(np.full(shape, value))

	def __repr__(self): return 'tensor(' + str(self.data) + ')'
	def __getitem__(self, key): return self.data[key]
	def __setitem__(self, key, value): self.data[key] = value
	def __len__(self): return len(self.data)
	def __iter__(self): return iter(self.data)
	def __copy__(self): return tensor(self.data.copy())
	def __eq__(self, other) -> bool: return self.data.all() == other.data.all()
	def __ne__(self, other) -> bool: return self.data.all() != other.data.all()
	def mean(self, axis=None): return np.mean(self.data, axis=axis)
	def sum(self, axis=None): return np.sum(self.data, axis=axis)