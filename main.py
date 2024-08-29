from network import Network
# from image import PIL
from mnist import download_and_parse_mnist_file

mndata = download_and_parse_mnist_file('samples')

images, labels = mndata.load_training()
print(images)
net = Network(90*140, 2, 5, 10)

print(net.getWeight(0, 0, 0))
