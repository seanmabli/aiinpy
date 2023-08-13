from src import tensor, sigmoid

a = tensor([1, 2, 3])
c = tensor([4, 5, 6])
b = sigmoid()
d = a + c
print(d)
print(d.autograd(a))