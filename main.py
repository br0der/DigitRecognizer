import math
from network import Network

from mnist import MNIST

mndata = MNIST('samples')

images, labels = mndata.load_training()

print(images[0])

net = Network(28*28, 2, 32, 10)
net.setSigmoid()

net.train(images, labels)