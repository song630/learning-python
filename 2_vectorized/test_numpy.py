import numpy


a = numpy.zeros((5, 2))
print(a.transpose())
b = numpy.array([
    [1],
    [2],
    [3],
    [4]
])
print(b.T)
print(1 - b)
W = numpy.random.uniform(-0.1, 0.1, (3, 4))
print(W.shape[0])
print(W.ndim)
index = 1
if not index:
    print('0')
else:
    print('1')
c = [1, 2, 3, 4]
for i, j in enumerate(reversed(c[0:-1])):
    print((c[i + 1], j))
