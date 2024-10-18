import random
import math

class Variable:
    def __init__(self, index, value):
        self.index = index
        self.value = value

var = Variable(1, True)
print(var.index)
print(var.value)



class Network:
    def __init__(self, inputDim, layers, nodesPerLayer, outputDim):
        assert(inputDim >= 1)
        self.inputDim = inputDim

        assert(layers >= 1)
        self.layers = layers

        assert(nodesPerLayer >= 1)
        self.nodesPerLayer = nodesPerLayer

        self.weights = [[] for i in range(layers + 1)]
        self.weights[0] = [[(random.random()*2) - 1 for j in range(inputDim)]
                            for i in range(nodesPerLayer)]
        for i in range(1, layers):
            mtx = [[(random.random()*2) - 1 for j in range(nodesPerLayer)] 
                   for i in range(nodesPerLayer)]
            self.weights[i] = mtx
        self.weights[-1] = [[(random.random()*2) - 1 for j in range(nodesPerLayer)]
                            for i in range(outputDim)]

        self.biases = [[(-1, 1)] for i in range(layers + 1)]
        for i in range(layers):
            mtx = [(random.random()*2) - 1 
                   for i in range(nodesPerLayer)]
            self.biases[i] = mtx
        self.biases[-1] = [(random.random()*2) - 1
                            for i in range(outputDim)]

        assert(outputDim >= 1)
        self.outputDim = outputDim

        self.activation = lambda x: x if x>=0 else 0
    
    def setActivation(self, f):
        self.activation = f

    def setSigmoid(self):
        def sigmoid(val):
            if val < -100: return 0
            if val > 100: return 1
            print(val)
            return 1.0/(1+math.pow(math.e, -val))
        
        self.setActivation(sigmoid)

    def activate(self, val):
        return self.activation(val)
    
    def getWeight(self, layer, st, en):
        return self.weights[layer][en][st]

    def getBias(self, layer, node):
        return self.biases[layer][node]

    def scaleOutput(self, output):
        assert(len(output)==10)
        l = [math.pow(math.e, item) for item in output]
        s = sum(l)
        return [item/s for item in l]

    def forwardPropogate(self, picture):
        assert(len(picture) == self.inputDim)

        # Pass from the input to the first layer
        currLayer = [0.0 for i in range(self.nodesPerLayer)]

        for i in range(self.inputDim):
            for j in range(self.nodesPerLayer):
                currLayer[j]+=picture[i]*self.getWeight(0, i, j)
        for j in range(self.nodesPerLayer):
            currLayer[j] += self.getBias(0, j)
            currLayer[j] = self.activate(currLayer[j])
            print(currLayer[j])

        print()

        # For all layers, propogate it through
        for layerIdx in range(1, self.layers):
            nextLayer = [0.0 for i in range(self.nodesPerLayer)]
            for i in range(self.nodesPerLayer):
                for j in range(self.nodesPerLayer):
                    nextLayer[j]+=currLayer[i]*self.getWeight(layerIdx, i, j)
            for j in range(self.nodesPerLayer):
                nextLayer[j]+=self.getBias(layerIdx, j)
                nextLayer[j] = self.activate(nextLayer[j])
                print(nextLayer[j])

            print()

            currLayer = nextLayer
        
        # Pass from the last layer to the output
        output = [0.0 for i in range(self.outputDim)]

        for i in range(self.nodesPerLayer):
            for j in range(self.outputDim):
                output[j]+=currLayer[i]*self.getWeight(self.layers, i, j)
        for j in range(self.outputDim):
            output[j]+=self.getBias(self.layers, j)
            output[j] = self.activate(output[j])
            print(output[j])

        print()
        print(output)
        return self.scaleOutput(output)

    def train(self, images, labels):
        training = list(zip(images, labels))
        random.shuffle(training)
        ideal = [0.0 if i!=training[0][1] else 1.0 for i in range(10)]
        output = self.forwardPropogate(training[0][0])
        error = sum([(output[i]-ideal[i])**2 for i in range(10)])
        print(output, error)
