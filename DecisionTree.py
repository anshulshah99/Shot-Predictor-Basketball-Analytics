# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:36:13 2019

@author: schmi
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
import getData
from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict


def readIn(testfile = "data/TestingData.csv",trainfile = "data/TrainingData.csv"):
    testDf = pd.read_csv(testfile)
    testing = testDf.to_numpy()
    trainDf = pd.read_csv(trainfile)
    training = trainDf.to_numpy()
    return (training, testing)

# Read the data from the new clustered shots file to better understand how the model
# works on different types of shots.
def readClustered(filename = "clustered_shots_v2.csv"):
    df = pd.read_csv(filename)
    ret = []
    for row in df.itertuples():
        cluster = row.cluster
        value = row.value
        result = row.result
        ret.append((cluster,value,result))
    allData = df.to_numpy()
    return (ret,allData)
    
# Process and analyze the training and testing data, storing it in a dict for later 
# access.
dataTuple = readIn()
train = dataTuple[0]
splitData = np.split(train,[train.shape[1] - 2, train.shape[1] - 1],axis = 1)
train = splitData[0]
trainResult = np.ravel(splitData[2])
test = dataTuple[1]
splitData2 = np.split(test,[test.shape[1] - 2, test.shape[1] - 1],axis = 1)
test = splitData2[0]
testResult = np.ravel(splitData2[2])
ret,allData = readClustered()
splitAll = np.split(allData,[2,17],axis = 1)
allData = splitAll[1]
dataDict = {"test":test,
            "train":train,
            "trainResult":trainResult,
            "testResult":testResult,
            "ret":ret,
            "allData":allData}

# List of trees in manually generated (non-sklearn) random forest. Whenever makeForest is 
# called, this list is cleared to prevent creating too many trees.
trees = []
# Stores the randomly selected parameters for each tree so they can be later accessed during testing.
paramDict = {}

# Run PCA transformation on data with 10 principal components.
def transform():
    train_features, train_results = getData.get_training()
    test_features, test_results = getData.get_testing()
    pca = PCA(n_components= 10)
    pca.fit(train_features)  
    PCA_train_features = pca.transform(train_features)  
    PCA_test_features = pca.transform(test_features)
    explained_variance = pca.explained_variance_ratio_
    return PCA_train_features, train_results, PCA_test_features, test_results

# Makes an individual decision tree. k_params is the number of randomly selected
# parameters to use when creating the tree; entOrGini specifies the criterion for splitting 
# nodes in the tree ("ent" being information entropy, "gini" being gini index);
# verbose controls whether details like parameters, criterion, and accuracy are printed;
# PCAfirst will run Principal Component Analysis first if set to True (NB: it seems to produce less
# accurate results); depth is the maximum depth of the tree, e.g. there are at most 
# 2^depth - 1 leaves in a tree. Forest is basically a flag variable, if it doesn't equal -1
# then makeForests() should have been called. Don't use the forest argument if you call
# makeTree() individually, it should only be called via makeForests().
def makeTree(k_params = 3,
             entOrGini = "gini",
             verbose = False,
             forest = -1, 
             PCAfirst = False,
             depth=4,
             forceDist = False):
    train = dataDict.get("train")
    test = dataDict.get("test")
    trainResult = dataDict.get("trainResult")
    testResult  = dataDict.get("testResult")
    if(PCAfirst):
        train, trainResult, test, testResult = transform()
    #randomOrder = np.random.permutation(8)[:k_params]
    if(forceDist):
        randomOrder = np.random.permutation(train.shape[1] - 2)
        shotDistIndex = 9
        randomOrder = [i for i in randomOrder if i != shotDistIndex]
        randomOrder = randomOrder[:k_params - 1]
        randomOrder = np.concatenate((np.asarray([shotDistIndex]),randomOrder))
    else:
        randomOrder = np.random.permutation(train.shape[1] - 1)[:k_params]
    # prints the parameters used if verbose is True
    if(verbose):
        print("Random parameters are " + str(randomOrder))
    # print(train.shape)
    
    # Selects k_params columns of data for training and testing
    train = train[:,randomOrder]
    test = test[:,randomOrder]
    # If the forest argument has been passed from makeForest(), the randomOrder is stored
    # in the paramDict for later access and evaluation.
    if(forest != -1):
        paramDict.update({forest:randomOrder})
    # Initialize the tree
    myTree = DecisionTreeClassifier(criterion = entOrGini,
                                   max_depth=depth, min_samples_leaf=5,random_state=0)
    # Fit the tree model using training input data and results
    myTree.fit(train, trainResult)
    
    # If verbose is True, prints out binary accuracy and other details
    if(verbose):
        test_pred = myTree.predict(test)
        accuracy = accuracy_score(testResult,test_pred) * 100
        print("Tree created using " + entOrGini + " criterion and " + str(k_params) + " random parameters.")
        print("Accuracy for this tree is " + str(accuracy) + "%.")
    return(myTree)

# Generates num_of_trees decision trees, effectively making a random forest. All the parameters 
# get passed directly to makeTree, with the addition of forest which serves as an enumeration/storage 
# tool. trees[i] will have its random parameters stored in paramDict.get(i), thanks to the forest argument.
def makeForest(num_of_trees = 100,
               k_params = 3,
               entOrGini = "gini",
               verbose = False,
               PCAfirst = False,
               depth = 3,
               forceDist = False):
    trees.clear()
    for i in range(num_of_trees):
        treeI = makeTree(k_params = k_params,
                         entOrGini = entOrGini,
                         verbose = verbose,
                         forest = i,
                         PCAfirst = PCAfirst,
                         depth = depth,
                         forceDist = forceDist)
        trees.append(treeI)
    return

def queryForest(scoretype = "probability", ratio = 0.5, dataset = "test"):
# Once the forest has been made, queryForest accesses it to get the "decision" from each
# tree. scoretype can be count, which simply returns a count of the number of trees deciding a 
# shot will go in, binary, which returns a 0 or 1 if the fraction of trees voting "1" is greater than
# ratio, or probability, which returns the averaged probability of each shot going in. Dataset can be
# test or allData. The default is test, but if the overall expected score is desired then allData is used.

    if len(trees) <= 1:
        print("Must initialize forest with multiple trees first. Use the makeForest() method to do so.")
        return
    if(dataset == "test"):
        test = dataDict.get("test")
    else:
        test = dataDict.get("allData")
    results = np.zeros(test.shape[0])
    probs = np.zeros(test.shape[0])
    num = 0
    for j in range(len(trees)):
        thisTree = trees[j]
        order = paramDict.get(j)
        if(scoretype == "probability"):
            thisProb = thisTree.predict_proba(test[:,order])
            probs = probs + thisProb[:,1]
            num = num + 1
        else:
            thisResult = thisTree.predict(test[:,order])
            results = results + thisResult
    if(scoretype == "count"):
        return results
    elif(scoretype == "binary"):
        return (results > (len(trees) * ratio)).astype(int)
    else:
        return(probs / num)
    
                    
# Compares binary forest query to actual results of shots going in/out
def findScore(ratio = 0.5):
    testResult = dataDict.get("testResult")
    forestResults = queryForest("binary",ratio)
    length = len(testResult)
    count = 0
    for i in range(length):
        if(testResult[i] == forestResults[i]):
            count = count + 1
    score = count / length
    return score

# Tool for visualizing how changing the decision ratio changes the overall shot
# prediction accuracy. findOptimalParams() is better overall for finding the best ratio,
# but this function is useful for visualizing it.
def createResponseCurve(k_params,
                        ratio_start,
                        ratio_end,
                        num_of_trees = 20,
                        ratio_inc=0.02,
                        PCAfirst = False):
    makeForest(num_of_trees,k_params,PCAfirst = PCAfirst)
    currRatio = ratio_start
    while(currRatio <= ratio_end):
        ratioScore = findScore(currRatio)
        plt.plot(currRatio,ratioScore,'bo')
        currRatio = currRatio + ratio_inc
    plt.show()

# Finds the optimal parameters for forest creation. param_start and param_end are the 
# lower and upper bounds for the number of random parameters to use. depth_start and depth_end
# are the respective bounds for tree depth, as are ratio_start and ratio_end. ratio_inc is the
# increment to be used when scanning different ratios (the increments for the others are 1). PCAfirst
# runs PCA if set to True, use_sklearn will create an sklearn forest rather than using makeForest().
def findOptimalParams(param_start,
                      param_end,
                      depth_start,
                      depth_end,
                      ratio_start,
                      ratio_end,
                      ratio_inc = 0.02,
                      PCAfirst = False,
                      use_sklearn = False,
                      forceDist = False):
    bestScore = 0
    bestParam = 0
    bestRatio = 0
    bestDepth = 0
    if(not use_sklearn):
        for d in range(depth_start,depth_end):
            for p in range(param_start,param_end):
                    makeForest(num_of_trees = 50,k_params=p,PCAfirst = PCAfirst,depth = d,forceDist=forceDist)
                    currRatio = ratio_start
                    while(currRatio <= ratio_end):
                        ratioScore = findScore(currRatio)
                        if(ratioScore > bestScore):
                            bestScore = ratioScore
                            bestParam = p
                            bestRatio = currRatio
                            bestDepth = d
                        currRatio = currRatio + ratio_inc
        ans = "Score: " + str(bestScore)+"\nParams: " + str(bestParam) + "\nRatio: " + str(bestRatio) + "\nDepth: " + str(bestDepth)
    else:
        for d in range(depth_start,depth_end):
            depthScore = sklearnScore(depth = d)
            if(depthScore > bestScore):
                bestScore = ratioScore
                bestRatio = currRatio
                bestDepth = d
        ans = "Score: " + str(bestScore) + "\nDepth: " + str(bestDepth)
    print(ans)
    return
    
# Creates a random forest using sklearn's random forest classifier.
def sklearnForest(entOrGini = "gini",PCAfirst = False,depth=4):
    train = dataDict.get("train")
    test = dataDict.get("test")
    trainResult = dataDict.get("trainResult")
    testResult  = dataDict.get("testResult")
    if(PCAfirst):
        train, trainResult, test, testResult = transform()
    clf = RandomForestClassifier(max_depth=depth,
                                 min_samples_leaf=5,
                                 random_state=0,
                                 criterion=entOrGini,
                                 n_estimators=50,
                                 bootstrap=True)
    clf.fit(train,trainResult)
    return clf

# Evaluates the score of the sklearn random forest.
def sklearnScore(entOrGini = "gini",PCAfirst = False,depth = 4,scoretype = "binary",dataset = "test"):       
    test = dataDict.get("test")
    testResult  = dataDict.get("testResult")   
    allData = dataDict.get("allData")
    clf = sklearnForest(entOrGini,PCAfirst,depth)
    print(clf.classes_)
    if(scoretype == "binary"):
        return clf.score(test,testResult)
    elif(scoretype == "probability"):
        if(dataset == "test"):
            return clf.predict_proba(test)
        else:
            return clf.predict_proba(allData)
        
# Lists the most important features of the sklearn forest. The larger the value,
# the greater the relative importance of the feature in determining whether a shot goes in.
def evalFeatures(entOrGini = "gini",PCAfirst = False,depth = 4,testfile = "TestingData.csv"):    
    clf = sklearnForest(entOrGini,PCAfirst,depth)
    featureVals = clf.feature_importances_
    if(not PCAfirst):
        testDf = pd.read_csv(testfile)
        featureNames = list(testDf.columns)
        for i in range(len(featureVals)):
            val = featureVals[i]
            name = featureNames[i]
            print(name + ": " + str(val))
    else: 
        for i in range(len(featureVals)):
            print("Feature "+str(i)+": " + str(featureVals[i]))
    return

def getExpectedScores(use_sklearn = False,forceDist=False):
    if(not use_sklearn):
        makeForest(forceDist=forceDist)
        probs = queryForest(scoretype = "probability",dataset = "allData")
    else:
        probs = sklearnScore(scoretype = "probability",dataset = "allData")
        probs = probs[:,1]
    clusterScores = {}
    ret = dataDict.get("ret")
    for count,item in enumerate(ret):
        cluster = item[0]
        value = item[1]
        result = item[2]
        if (cluster not in clusterScores):
            clusterScores.update({cluster:[0,0]})
        scoresArr = clusterScores.get(cluster)
        scoresArr[0] = scoresArr[0] + value * result
        scoresArr[1] = scoresArr[1] + value * probs[count]
    for cluster in clusterScores:
        finalArr = clusterScores.get(cluster)
        actual = finalArr[0]
        predicted = finalArr[1]
        error = abs((actual - predicted)) / predicted * 100
        print("Actual score for cluster " + str(cluster) + " was: " +
        str(actual) + ".\n Predicted score was: " + str(predicted) + ".\n Error: " + 
        str(error) + "%.")
    return


#
#if __name__ == '__main__':
#    makeForest()
#    print(queryForest())

