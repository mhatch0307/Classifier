# -*- coding: utf-8 -*-

import csv
import random
import math
import np


# HardCode Classifier Class
class Classifier:

# Constructor
# @param: filename - string of the csv filename
    def __init__(self, filename):
        with open(filename, 'rb') as csvfile:
            self.data = list(csv.reader(csvfile, delimiter=',', quotechar='|'))
            self.vectorSize = len(self.data[0])
            self.data.pop()

# Splits the data read from the csv file into testing
# and training lists
# @param: split - float representing the part it should be
# split at
    def splitData(self, split):
            random.shuffle(self.data)
            size = len(self.data)
            end1 = int(size * split)
            self.testing = self.data[0:end1]
            self.training = self.data[end1:size]

# displays the data read in from the csv file
    def displayData(self):
        for row in self.training:
            print(row)

# classifies the data
# @param data: vector to be classified
    def classify(self, inputs):
        return self.data[0][4]

# calculates the accuracy of the data
# @return float containing the accuracy
    def calculateAccuracy(self):
        correctCount = 0
        for row in self.testing:
            print((row[4], self.classify(row)))
            if row[4] == self.classify(row):
                correctCount = correctCount + 1
        return correctCount / float(len(self.data))

#classifier = Classifier('iris.data')
#classifier.displayData()
#classifier.splitData(.3)
#print((classifier.calculateAccuracy()))


class KNNClassifier(Classifier):

    #calculates ecludian distance between vector1 and vector2
    def distance(self, vector1, vector2):
        total = 0
        for i in range(self.vectorSize - 1):
            try:
                float(vector1[i])
                float(vector2[i])
                total += math.fabs(float(vector1[i]) - float(vector2[i]))
            except ValueError:
                total += 0 if vector1[i] == vector2[i] else 1
        return total

    #finds the nearest k neighbors to inputs from self.training
    def knn(self, k, inputs):

        distances = []

        for row in self.training:
            dataClass = row[self.vectorSize - 1]
            distances.append((self.distance(row, inputs), dataClass))

        distances.sort()
        kClosests = dict()
        dClass = ['', 0]

        if(k == 1):
            return distances[0][1]
        else:
            for i in range(k):
                if(distances[i][1] in kClosests):
                    kClosests[distances[i][1]] += 1
                else:
                    kClosests[distances[i][1]] = 0
                if dClass[0] != distances[i][1] and \
                kClosests[distances[i][1]] > dClass[1]:
                    dClass[0] = distances[i][1]
                    dClass[1] = kClosests[distances[i][1]]
        return dClass[0]

    #find accuracy of the classifier
    def calculateAccuracy(self, k):
        correctCount = 0
        for row in self.testing:
            dataClass = row[self.vectorSize - 1]
            if dataClass == self.knn(k, row):
                correctCount = correctCount + 1
        return correctCount / float(len(self.testing))


class ID3DecisionTree(Classifier):
    # Source: Machine Learning by Stephen Marsland p.251
    def calc_entropy(p):
        if p != 0:
            return -p * np.log2(p)
        else:
            return 0
    # Source: Machine Learning by Stephen Marsland p.253-254

    def calc_info_gain(self, data, classes, feature):
        gain = 0
        nData = len(data)
        # List the values that feature can take
        values = []
        for datapoint in data:
            if datapoint[feature] not in values:
                values.append(datapoint[feature])

        featureCounts = np.zeros(len(values))
        entropy = np.zeros(len(values))
        valueIndex = 0
        #Find where those values appear in data[feature] and the corresponding
        # class
        for value in values:
            dataIndex = 0
            newClasses = []
            for datapoint in data:
                if datapoint[feature] == value:
                    featureCounts[valueIndex] += 1
                    newClasses.append(classes[dataIndex])
                dataIndex += 1

            # Get the values in newClasses
            classValues = []
            for aclass in newClasses:
                if classValues.count(aclass) == 0:
                    classValues.append(aclass)
            classCounts = np.zeros(len(classValues))
            classIndex = 0
            for classValue in classValues:
                for aclass in newClasses:
                    if aclass == classValue:
                        classCounts[classIndex] += 1
                classIndex += 1

            for classIndex in range(len(classValues)):
                entropy[valueIndex] += self.calc_entropy(float(
                classCounts[classIndex]) / sum(classCounts))
            gain += float(featureCounts[valueIndex]) / \
            nData * entropy[valueIndex]
            valueIndex += 1
        return gain

classifier = KNNClassifier('car.data')
classifier.splitData(.3)
#classifier.displayData()
print((classifier.calculateAccuracy(1)))
print((classifier.calculateAccuracy(2)))
print((classifier.calculateAccuracy(3)))
print((classifier.calculateAccuracy(4)))
print((classifier.calculateAccuracy(5)))
print((classifier.calculateAccuracy(6)))
print((classifier.calculateAccuracy(7)))
print((classifier.calculateAccuracy(8)))
print((classifier.calculateAccuracy(10)))