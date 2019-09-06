'''
Created on Jun 19, 2019

@author: anshul
'''
from sklearn.naive_bayes import GaussianNB
import PCA
import numpy as np

def predict(data, components):
    pca, train_features, train_targets, test_features, test_results, values = PCA.transform(components)
    model = GaussianNB()
    # Train the model using the training sets
    model.fit(train_features, train_targets)
    PCA_data = pca.transform(data)
    predicted= model.predict_proba(PCA_data)
    return(predicted, test_features, test_results, values) 

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
        if results[i] == 1:
            scored += value
        expected += prob[i][1]*value
    print(scored, expected)
    print(count/total)

if __name__ == '__main__':
    compare()