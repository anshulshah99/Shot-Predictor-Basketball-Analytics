'''
Created on Jun 18, 2019

@author: anshul
'''
import PCA
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn import linear_model
from keras.layers.advanced_activations import LeakyReLU
# generate regression dataset

def predict(inputs):
    pca, train_features, train_results, test_features, test_results, values = PCA.transform(inputs)
    clf = linear_model.Ridge(alpha=1)
    #print(train_features)
    clf.fit(train_features, train_results)
    #print(train_features)
    #print(clf.coef_)
    #PCA_data = pca.transform(10)
    model = Sequential()
    model.add(Dense(inputs, input_dim=inputs, activation = "relu"))
    model.add(Dense(5))
    model.add(LeakyReLU(alpha = 0.01))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics = ['accuracy'])
    one_hot_labels = keras.utils.to_categorical(train_results, num_classes=2)
    model.fit(train_features, one_hot_labels, epochs=3000, verbose=0)
    predicted = model.predict_proba(train_features)
    
    makes = []
    miss = []
    scored = 0
    expected = 0
    #print(predicted)
    for i in range(len(predicted)):
        """if(train_results[i] == 1 and predicted[i][1] >= 0.5) or (train_results[i] == 0 and predicted[i][1] < 0.5):
            count += 1
        total += 1"""
        if(train_results[i] == 1):
            scored += values[i]
            makes.append(predicted[i][1])
        if(train_results[i] == 0):
            miss.append(predicted[i][1])
        expected += predicted[i][1]*values[i]
    return predicted

if __name__ == '__main__':
    predict(10)
    """x = 0
    basket = []
    out = []
    for i in range(3):
        expected, makes, miss = predict(10)
        x += expected
        basket.extend(makes)
        out.extend(miss)
    print(x)
    plt.boxplot([basket, out])
    plt.show()"""
    