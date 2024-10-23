import random
import math


class Network:
    def __init__(self, inputDim, hiddenLayers, nodesPerLayer, outputDim):
        assert(inputDim >= 1)
        self.inputDim = inputDim

        assert(hiddenLayers >= 1)
        self.hiddenLayers = hiddenLayers

        assert(nodesPerLayer >= 1)
        self.nodesPerLayer = nodesPerLayer

        assert(outputDim >= 1)
        self.outputDim = outputDim

        self.activation = lambda x: x if x >= 0 else 0
        
        self.weights = [[] for i in range(hiddenLayers + 1)]
        for layer in range(hiddenLayers+1):
            self.weights[layer] = [\
                [(random.random()*2) - 1 for leftNode in range(self.nodesInLayer(layer))] \
                for rightNode in range(self.nodesInLayer(layer+1)) \
            ]

        self.biases = [[] for i in range(hiddenLayers + 2)]
        for layer in range(hiddenLayers+2):
            if layer > 0:
                self.biases[layer] = [ \
                    (random.random()*2) - 1 \
                    for node in range(self.nodesInLayer(layer)) ]
            else:
                self.biases[layer] = [0 for node in range(self.nodesInLayer(layer))]

        self.activationValues = [[] for i in range(hiddenLayers + 2)]
        for layer in range(hiddenLayers+2):
            self.activationValues[layer] = [None for i in range(self.nodesInLayer(layer))]

        self.activationDeriv = []

        self.derivativeStep = 0.0001

        self.learningRate = 0.01
    
    def __init__(self, inputDim, hiddenLayers, nodesPerLayer, outputDim, weights, biases):
        assert(inputDim >= 1)
        self.inputDim = inputDim

        assert(hiddenLayers >= 1)
        self.hiddenLayers = hiddenLayers

        assert(nodesPerLayer >= 1)
        self.nodesPerLayer = nodesPerLayer

        assert(outputDim >= 1)
        self.outputDim = outputDim

        self.activation = lambda x: x if x >= 0 else 0
        
        self.weights = weights

        self.biases = biases

        self.activationValues = [[] for i in range(hiddenLayers + 2)]
        for layer in range(hiddenLayers+2):
            self.activationValues[layer] = [None for i in range(self.nodesInLayer(layer))]

        self.activationDeriv = []

        self.derivativeStep = 0.0001

        self.learningRate = 0.01

    def setActivation(self, f):
        self.activation = f

    def sigmoid(self, val):
            if val < -100: return 0
            if val > 100: return 1
            # print(val)
            return 1.0/(1+math.pow(math.e, -val))

    def setSigmoid(self):
        self.setActivation(self.sigmoid)

    def activate(self, val):
        return self.activation(val)
    
    def getWeight(self, layer, st, en):
        assert(0 <= layer <= self.hiddenLayers+1)
        assert(0 <= en < self.nodesInLayer(layer+1))
        assert(0 <= st < self.nodesInLayer(layer))
        return self.weights[layer][en][st]

    def getBias(self, layer, node):
        return self.biases[layer-1][node]
    
    def nodesInLayer(self, layer):
        assert(0<=layer<=self.hiddenLayers+1)
        if layer==0:
            return self.inputDim
        if layer==self.hiddenLayers+1:
            return self.outputDim
        return self.nodesPerLayer
        
    def scaleOutput(self, output):
        assert(len(output)==self.outputDim)
        mx_logit = max(output)

        if mx_logit==0: l = [1 for item in output]
        else: l = [math.pow(math.e, (item/mx_logit)) for item in output]

        s = sum(l)
        return [item/s for item in l]

    def forwardPropogate(self, picture):
        # TODO: Keep track of all activations to use in back prop. Weight error is activation[left] * error[right]
        assert(len(picture) == self.inputDim)

        self.activationValues[0] = picture

        self.activationDeriv = [[0.0 for j in range(self.nodesInLayer(i))] for i in range(self.hiddenLayers + 1)]

        # Pass it through each layer
        for layerIdx in range(1, self.hiddenLayers+2):
            self.activationValues[layerIdx] = [0.0 for i in range(self.nodesInLayer(layerIdx))]
            for leftNode in range(self.nodesInLayer(layerIdx-1)):
                for rightNode in range(self.nodesInLayer(layerIdx)):
                    self.activationValues[layerIdx][rightNode] += self.activationValues[layerIdx-1][leftNode] * self.getWeight(layerIdx-1, leftNode, rightNode)
            for rightNode in range(self.nodesInLayer(layerIdx)):
                self.activationValues[layerIdx][rightNode] += self.getBias(layerIdx, rightNode)

                nodeVal = self.activationValues[layerIdx][rightNode]

                self.activationDeriv[layerIdx-1][rightNode] = \
                    (self.activate(nodeVal + self.derivativeStep)
                        - self.activate(nodeVal - self.derivativeStep)) \
                    / (2*self.derivativeStep)

                if layerIdx != self.hiddenLayers+1:
                    self.activationValues[layerIdx][rightNode] = self.activate(nodeVal)
                else:
                    self.activationValues[layerIdx][rightNode] = self.sigmoid(nodeVal)

        return self.scaleOutput(self.activationValues[-1])

    def getError(self, image, label):
        ideal = [0.0 if i!=label else 1.0 for i in range(self.outputDim)]
        output = self.forwardPropogate(image)
        # print(f"output: {output}")
        error = [-(ideal[i]*(math.log(output[i])) + (1-ideal[i])*(math.log(1-output[i]))) for i in range(self.outputDim)]
        return error

    def backPropogate(self, image, label):
        error = self.getError(image, label)
        weightError = [[] for i in range(self.hiddenLayers + 1)]
        for layer in range(self.hiddenLayers+1):
            weightError[layer] = [\
                [0.0 for leftNode in range(self.nodesInLayer(layer))] \
                for rightNode in range(self.nodesInLayer(layer+1)) \
            ]
        
        biasError = [[0.0 for j in range(self.nodesInLayer(i+1))] for i in range(self.hiddenLayers + 1)]

        # print()
        # print(output)
        # print(ideal)
        # print(error)
        for layer in reversed(range(self.hiddenLayers+1)):
            for node in range(len(error)):
                biasError[layer][node] = error[node]
            nextError = [0.0]*self.nodesInLayer(layer)
            for leftNode in range(self.nodesInLayer(layer)):
                for rightNode in range(len(error)):
                    nextError[leftNode] += error[rightNode] * self.activationDeriv[layer][rightNode] * self.getWeight(layer, leftNode, rightNode)
                    weightError[layer][rightNode][leftNode] = error[rightNode] * self.activationValues[layer][leftNode]
            error = nextError
            
        allErrors = []

        for layer in range(self.hiddenLayers+1):
            for rightNode in range(self.nodesInLayer(layer+1)):
                # print(layer, rightNode, weightError[layer][rightNode])
                for leftNode in range(self.nodesInLayer(layer)):
                    allErrors.append(((layer, rightNode, leftNode), weightError[layer][rightNode][leftNode]))
            # print()

        for layer in range(self.hiddenLayers+1):
            # print(layer, biasError[layer])
            for node in range(self.nodesInLayer(layer+1)):
                allErrors.append(((layer, node), biasError[layer][node]))
        # print()

        return allErrors

    def train(self, images, labels):
        training = list(zip(images, labels))
        random.shuffle(training)
        allErrors = self.backPropogate(training[0][0], training[0][1])
        # print(f"errors: {allErrors}")
        for key, value in allErrors:
            if len(key)==3:
                layer, rightNode, leftNode = key
                self.weights[layer][rightNode][leftNode] -= self.learningRate * value
            elif len(key)==2:
                layer, node = key
                self.biases[layer][node] -= self.learningRate * value
            else:
                raise KeyError
