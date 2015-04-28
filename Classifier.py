# -*- coding: utf-8 -*-

import csv
import random


# HardCode Classifier Class
class Classifier:

# Constructor
# @param: filename - string of the csv filename
    def __init__(self, fileName):
        with open(fileName, 'rb') as csvfile:
            self.data = list(csv.reader(csvfile, delimiter=',', quotechar='|'))
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
        for row in self.data:
            print(row)

# classifies the data
# @param data: vector to be classified
    def classify(self, data):
        return 'Iris-setosa'

# calculates the accuracy of the data
# @return float containing the accuracy
    def calculateAccuracy(self):
        correctCount = 0
        for row in self.testing:
            print((row[4], self.classify(row)))
            if row[4] == self.classify(row):
                correctCount = correctCount + 1
        return correctCount / float(len(self.data))

classifier = Classifier('iris.data')
classifier.displayData()
classifier.splitData(.3)
print((classifier.calculateAccuracy()))
