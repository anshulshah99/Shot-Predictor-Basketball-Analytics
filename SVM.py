'''
Created on Jun 21, 2019

@author: anshul
'''
from sklearn import svm
import numpy as np
import PCA

def predict(data, components):
    pca, train_features, train_results, test_features, test_results, values = PCA.transform(components)
    clf = svm.SVC(kernel = "rbf", gamma = 'auto', probability = True)
    PCA_data = pca.transform(data)
    clf.fit(train_features, train_results)
    outcome = clf.predict_proba(PCA_data)
    return(outcome, test_features, test_results, values)

def compare():
    prob, features, results, values = predict()
    scored = 0
    expected = 0
    count = 0
    total = 0

    for i in range(len(features)):
        if (prob[i][1] >= 0.5 and results[i] == 1) or (prob[i][1] < 0.5 and results[i] == 0):
            count += 1
        total += 1
        value = values[i]
        #print(prob[i])
        if results[i] == 1:
            scored += value
        expected += prob[i][1]*value
    #print(count/total)
    return("Expected points: ", str(expected), "Actual points: ", str(scored))
    
if __name__ == '__main__':
    predict()
    compare()