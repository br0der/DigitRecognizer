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
        # self.weights[0] = [[(random.random()*2) - 1 for j in range(inputDim)]
        #                     for i in range(nodesPerLayer)]
        # for i in range(1, hiddenLayers+1):
        #     mtx = [[(random.random()*2) - 1 for j in range(nodesPerLayer)] 
        #            for i in range(nodesPerLayer)]
        #     self.weights[i] = mtx
        # self.weights[-1] = [[(random.random()*2) - 1 for j in range(nodesPerLayer)]
        #                     for i in range(outputDim)]

        self.biases = [[] for i in range(hiddenLayers + 2)]
        for layer in range(hiddenLayers+2):
            if layer > 0:
                self.biases[layer] = [ \
                    (random.random()*2) - 1 \
                    for node in range(self.nodesInLayer(layer)) ]
            else:
                self.biases[layer] = [0 for node in range(self.nodesInLayer(layer))]
        # for i in range(hiddenLayers):
        #     mtx = [(random.random()*2) - 1 
        #            for i in range(nodesPerLayer)]
        #     self.biases[i] = mtx
        # self.biases[-1] = [(random.random()*2) - 1
        #                     for i in range(outputDim)]
        self.derivatives = []

        self.derivativeStep = 0.0001
    
    def setActivation(self, f):
        self.activation = f

    def setSigmoid(self):
        def sigmoid(val):
            if val < -100: return 0
            if val > 100: return 1
            # print(val)
            return 1.0/(1+math.pow(math.e, -val))
        
        self.setActivation(sigmoid)

    def activate(self, val):
        return self.activation(val)
    
    def getWeight(self, layer, st, en):
        assert(0 <= layer <= self.hiddenLayers+1)
        assert(0 <= en < self.nodesInLayer(layer+1))
        assert(0 <= st < self.nodesInLayer(layer))
        return self.weights[layer][en][st]

    def getBias(self, layer, node):
        return self.biases[layer][node]
    
    def nodesInLayer(self, layer):
        assert(0<=layer<=self.hiddenLayers+1)
        if layer==0:
            return self.inputDim
        if layer==self.hiddenLayers+1:
            return self.outputDim
        return self.nodesPerLayer
        
    def scaleOutput(self, output):
        assert(len(output)==10)
        mx_logit = max(output)

        l = [math.pow(math.e, (item/mx_logit)) for item in output]
        s = sum(l)
        return [item/s for item in l]

    def forwardPropogate(self, picture):
        assert(len(picture) == self.inputDim)

        currLayer = picture

        self.derivatives = [[] for i in range(self.hiddenLayers + 1)]
        for layer in range(self.hiddenLayers+1):
            self.derivatives[layer] = [\
                [0.0 for leftNode in range(self.nodesInLayer(layer))] \
                for rightNode in range(self.nodesInLayer(layer+1)) \
            ]

        # Pass it through each layer
        for layerIdx in range(0, self.hiddenLayers+1):
            nextLayer = [0.0 for i in range(self.nodesInLayer(layerIdx+1))]
            for leftNode in range(self.nodesInLayer(layerIdx)):
                for rightNode in range(self.nodesInLayer(layerIdx+1)):
                    nextLayer[rightNode]+=currLayer[leftNode]*self.getWeight(layerIdx, leftNode, rightNode)
            for rightNode in range(self.nodesInLayer(layerIdx+1)):
                nextLayer[rightNode] += self.getBias(layerIdx+1, rightNode)

                for leftNode in range(self.nodesInLayer(layerIdx)):
                    self.derivatives[layerIdx][rightNode][leftNode] = \
                        (self.activate(nextLayer[rightNode] + self.derivativeStep)
                         - self.activate(nextLayer[rightNode] - self.derivativeStep)) \
                        / (2*self.derivativeStep)
                    
                nextLayer[rightNode] = self.activate(nextLayer[rightNode])
                # print(nextLayer[j])
            currLayer = nextLayer
        
        output = currLayer
        print(output)
        return self.scaleOutput(output)

    def train(self, images, labels):
        training = list(zip(images, labels))
        random.shuffle(training)
        ideal = [0.0 if i!=training[0][1] else 1.0 for i in range(10)]
        output = self.forwardPropogate(training[0][0])
        print(output)
        
        error = [-(ideal[i]*(math.log(output[i])) + (1-ideal[i])*(math.log(1-output[i]))) for i in range(self.outputDim)]
        weightError = [[] for i in range(self.hiddenLayers + 1)]
        for layer in range(self.hiddenLayers+1):
            weightError[layer] = [\
                [0.0 for leftNode in range(self.nodesInLayer(layer))] \
                for rightNode in range(self.nodesInLayer(layer+1)) \
            ]

        print(output)
        print(ideal)
        print(error)
        for layer in reversed(range(self.hiddenLayers+1)):
            nextError = [0.0]*self.nodesInLayer(layer)
            for leftNode in range(self.nodesInLayer(layer)):
                for rightNode in range(len(error)):
                    nextError[leftNode] += error[rightNode] * self.derivatives[layer][rightNode][leftNode] * self.getWeight(layer, leftNode, rightNode)
                    weightError[layer][rightNode][leftNode] = error[rightNode] * self.derivatives[layer][rightNode][leftNode]
            error = nextError
        for layer in range(self.hiddenLayers+1):
            for rightNode in range(self.nodesInLayer(layer+1)):
                print(weightError[layer][rightNode])
            print()
        # print(error)
