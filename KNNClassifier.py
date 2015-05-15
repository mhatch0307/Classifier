import sys
sys.path.insert(0, 'Classifier.py')

import Classifier


class KNNClassifier(Classifier):
    def knn(self, k, inputs):

        distances = (self.training - inputs[0, :])
        for row in distances:
            print(row)

classifier = KNNClassifier('iris.data')
classifier.splitData(3)
classifier.knn(3, [5.9, 3.0, 5.1, 1.8, 'Iris-virginica'])