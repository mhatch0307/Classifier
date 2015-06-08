from numpy import *
import csv
import math


class Node(object):

    def __init__(self):

        self.threshold = 1
        self.inputWeightSum = 0
        self.size = 0

    def addInput(self, inputWeight):
        self.size += 1
        self.inputWeightSum += inputWeight

    def activate(self):
        self.output = 1 / (1 + math.exp(-self.inputWeightSum))
        return self.output

    def setTarget(self, target):
        self.target = target


class Layer():

    def __init__(self, inputs, bias, numNodes, weights):
        self.inputs = inputs
        self.numInputs = len(inputs)
        self.numNodes = numNodes
        self.bias = bias
        self.weights = weights
        self.initializeNodes()

    def initializeNodes(self):
        self.nodes = list()
        for i in range(self.numNodes):
            node = Node()
            node.addInput(self.weights[i][0] * self.bias)
            for j in range(self.numInputs):
                node.addInput(self.weights[i][j + 1] * float(self.inputs[j]))
            self.nodes.append(node)

    def updateWeights(self, learningRate):
        self.newWeights = list()
        for i in range(self.numNodes):
            rWeights = list()
            rWeights.append(self.weights[i][0])
            for j in range(self.numInputs):
                rWeights.append(self.weights[i][j + 1] -
                (learningRate * self.nodes[i].nodeError *
                  float(self.inputs[j])))
            self.newWeights.append(rWeights)

    def getOutputs(self):
        self.outputs = list()
        for node in self.nodes:
            self.outputs.append(node.activate())
        return self.outputs


class MLP():

    def __init__(self, numLayersNodes, inputs, target, learningRate):
        self.numLayersNodes = numLayersNodes
        self.numLayers = len(numLayersNodes)
        self.bias = -1
        self.inputs = inputs
        self.layers = list()
        self.target = target
        self.learningRate = learningRate
        self.initializeWeights()

    def initializeWeights(self):
        self.weights = list()
        numInputs = len(self.inputs)
        random.seed()
        for numNodes in (self.numLayersNodes):
            weights = list()
            for i in range(numNodes):
                rWeights = list()
                for j in range(numInputs + 1):
                    rWeights.append(random.randint(-100, 100) / 100.0)
                weights.append(rWeights)
            self.weights.append(weights)

    def getWeights(self):
        return self.weights

    def setWeights(self, weights):
        self.weights = weights

    def runNetwork(self):
        layer = Layer(self.inputs, -1, self.numLayersNodes[0], self.weights[0])
        self.layers.append(layer)
        for i in range(1, self.numLayers):
            layer = Layer(layer.getOutputs(), self.bias,
                          self.numLayersNodes[i], self.weights[i])
            self.layers.append(layer)
        self.outputs = layer.getOutputs()
        return self.outputs

    def backpropagate(self):
        rightLayer = self.layers[self.numLayers - 1]
        self.calcOutputError(rightLayer.nodes)
        rightLayer.updateWeights(self.learningRate)
        for i in range(self.numLayers - 1):
            leftLayer = self.layers[self.numLayers - 2 - i]
            self.calcNodeError(leftLayer, rightLayer)
            leftLayer.updateWeights(self.learningRate)
            rightLayer = leftLayer
        self.weights = list()
        for layer in self.layers:
            self.weights.append(layer.newWeights)

    def calcOutputError(self, nodes):
        for node in nodes:
            node.nodeError = node.output * (1 - node.output) * \
            (node.output - self.target)

    def calcNodeError(self, leftLayer, rightLayer):
        for i in range(leftLayer.numNodes):
            rightWeightSum = 0
            for j in range(rightLayer.numNodes):
                rightWeightSum += rightLayer.weights[j][i] *\
                rightLayer.nodes[j].nodeError
            leftLayer.nodes[i].nodeError = (leftLayer.nodes[i].output *
             (1 - leftLayer.nodes[i].output)) * rightWeightSum


class Classifier():

    def __init__(self, data, classifications, numLayersNodes):
        data.pop()
        self.data = data
        self.classifications = classifications
        self.numCorrect = 0
        self.size = len(self.data)
        self.numLayersNodes = numLayersNodes

    def splitData(self, split):
            random.shuffle(self.data)
            size = len(self.data)
            end1 = int(size * split)
            self.testing = self.data[0:end1]
            self.training = self.data[end1:size]

    def train(self):
        size = len(self.training[0]) - 1
        weights = 0
        for d in self.training:
            target = 0
            dataClass = d[size]
            for classification in self.classifications:
                if dataClass == classification[0]:
                    target = classification[2]
            inputs = d[0:size]
            try:
                float(d[0])
                inputs = d[0:size]
            except ValueError:
                for i in range(0, size):
                    inputs[i] = i
            network = MLP(self.numLayersNodes, inputs, target, .3)

            #for classification in self.classifications:
             #   if classification[0] == dataClass and classification[3] != 0:
             #      network.setWeights(classification[3])

            if weights != 0:
                network.setWeights(weights)

            result = network.runNetwork()
            print((result, target))
            result = network.backpropagate()
            weights = network.getWeights()
            print((result, target))
            print("")

            #for classification in self.classifications:
             #   if classification[0] == dataClass and \
             #      result[0] > classification[1] and \
             #      result[0] <= classification[2]:
              #      classification[3] = weights
        return weights

    def classify(self, weights):
        size = len(self.testing[0]) - 1
        for d in self.testing:
            target = 0
            dataClass = d[size]
            for classification in self.classifications:
                if dataClass == classification[0]:
                    target = (classification[1] + classification[2]) / 2
            inputs = d[0:size]
            try:
                float(d[0])
                inputs = d[0:size]
            except ValueError:
                for i in range(0, size):
                    inputs[i] = i

            network = MLP(self.numLayersNodes, inputs, target, .9)

            network.setWeights(weights)

            result = network.runNetwork()
            print((result, target))

            for classification in self.classifications:
                if classification[0] == dataClass and \
                   result[0] > classification[1] and \
                   result[0] <= classification[2]:
                    self.numCorrect += 1

    def calc_accuracy(self):
        return float(self.numCorrect) / float(self.size)


with open('iris.data', 'rb') as csvfile:
        data = list(csv.reader(csvfile, delimiter=',', quotechar='|'))

size = len(data[0]) - 1

classifications = list([['Iris-setosa', 0, .33, 0],
    ['Iris-versicolor', .33, .66, 0], ['Iris-virginica', .66, 1.0, 0]])

numLayersNodes = [4, 4, 1]

classifier = Classifier(data, classifications, numLayersNodes)
classifier.splitData(.3)
weights = classifier.train()
classifier.classify(weights)
print((classifier.calc_accuracy()))



