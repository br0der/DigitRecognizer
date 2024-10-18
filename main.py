import math
from network import Network

from mnist import MNIST

mndata = MNIST('samples')

images, labels = mndata.load_training()

net = Network(28*28, 1, 16, 10)
# net.setSigmoid()

net.train(images, labels)