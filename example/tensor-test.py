from src import tensor

a = tensor([1, 2, 3])
b = tensor([4, 5, 6])
print(a + b)
print(a * b)
print(a - b)
# print(a / b) # failed
print(a ** b)
print(a % b)
print(a @ b)
print(a == b)
print(a != b)
print(-a)
print(a.ops)
print(a[1])

a = tensor([1, 2, 3])
print(a.mean())
print(a.sum())

b = tensor([[1, 2, 3], [4, 5, 6]])
print(b.mean())
print(b.sum())
print(b.shape)
print(b.reshape((3, 2)))

c = tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(c.mean())
print(c.sum())
print(b.shape)
print(c.reshape((3, 2, 2)))