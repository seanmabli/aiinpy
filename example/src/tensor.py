class tensor:
    def __init__(self, data):
        self.data = data

    def __add__(self, other):
        return tensor(self.data + other.data)
    def __mul__(self, other):
        return tensor(self.data * other.data)
    def __sub__(self, other):
        return tensor(self.data - other.data)
    def __truediv__(self, other):
        return tensor(self.data / other.data)
    def __pow__(self, other):
        return tensor(self.data ** other.data)
    def __mod__(self, other):
        return tensor(self.data % other.data)
    def __floordiv__(self, other):  
        return tensor(self.data // other.data)
    def __matmul__(self, other):
        return tensor(self.data @ other.data)