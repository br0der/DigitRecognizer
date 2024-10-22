import math
from network import Network

from mnist import MNIST

mndata = MNIST('samples')

images, labels = mndata.load_training()

# net = Network(28*28, 1, 16, 10)
w1 = [[1, 1, 0, 0], [0, 0, 1, 1]]
w2 = [[1, -1], [-1, 1]]
b2 = [0 for i in range(2)]
b3 = [0 for i in range(2)]
testNet = Network(4, 1, 2, 2, [w1, w2], [b2, b3])
# net.setSigmoid()

# net.train()
testImages = [[1, 0.5, 0, 1]]
testLablels = [1]


print(testNet.getError(testImages[0], testLablels[0]))
print(testNet.weights)
print(testNet.biases)

print()

testNet.train(testImages, testLablels)
print(testNet.getError(testImages[0], testLablels[0]))
print(testNet.weights)
print(testNet.biases)
