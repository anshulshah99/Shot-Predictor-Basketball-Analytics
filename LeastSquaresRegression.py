'''
Created on Jun 18, 2019

@author: anshul
'''
import getData
import numpy as np

def lsq():
    train_features, train_results = getData.get_training()
    testing_features, testing_results = getData.get_testing()
    equation = np.linalg.lstsq(train_features, train_results, rcond = None)[0]
    predictions = []
    correct = 0
    total = 0
    for i in range(len(testing_features)):
        output = np.dot(testing_features[i], equation)
        predictions.append((output, testing_results[i]))
        if output * testing_results[i] > 0:
            correct += 1
        total += 1
    print(correct/total)

    makes = [k[0] for k in predictions if k[1] == 1]
    misses = [k[0] for k in predictions if k[1] == -1]


        
if __name__ == '__main__':
    lsq()