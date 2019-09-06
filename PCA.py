'''
Created on Jun 20, 2019

@author: anshul
'''
from sklearn.decomposition import PCA
import getData
import matplotlib.pyplot as plt 
import pandas as pd

def transform(components):
    train_features, train_results = getData.get_training()
    test_features, test_results, values = getData.get_testing()
    
    pca = PCA(n_components= components)
    pca.fit(train_features) 
    PCA_train_features = pca.transform(train_features)
    PCA_test_features = pca.transform(test_features)
    explained_variance = pca.explained_variance_ratio_
    #return explained_variance
    return pca, PCA_train_features, train_results, PCA_test_features, test_results, values

if __name__ == '__main__':
    x = []
    y = []
    print(sum(transform(10)))
    for i in range(16):
        x.append(i)
        y.append(sum(transform(i)))
    plt.plot(x, y, 'd-b')
    plt.ylabel("Explained Variance")
    plt.xlabel("Number of Components")
    plt.title("PCA Variance")
    plt.show()