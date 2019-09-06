"""
Created on Tue Jun 18 10:19:08 2019

@author: schmi
"""

import numpy as np
import pandas as pd

clusterDict = {}


class Shot:
    def __init__(self):
        self.distance_ten_seconds = 0
        self.distance_total_game = 0
        self.velocity = 0
        self.distance_closest_def = 0 #DONE
        self.angle_closest_def = 0  #DONE
        self.shot_distance = 0 #DONE
        self.shot_angle = 0
        self.catch_and_shoot = 0 #DONE
        self.shot_clock = 0 #DONE
        self.result = 0 #DONE
        self.angle_second_def = 0
        self.distance_second_def = 0
        self.angle_closest_teammate = 0
        self.distance_closest_teammate = 0
        self.offense_hull = 0
        self.defense_hull = 0
        self.value = 0

# Read in clusters and store them in a dict with cluster numbers as keys
def readClusters(inputname = "data/shots_standardized.csv"):
    df = pd.read_csv(inputname)
    df.columns = df.columns.str.replace('.', '_')
    for row in df.itertuples():
        clusterNum = row.cluster
        myShot = Shot()
        myShot.distance_ten_seconds = row.distance_ten_seconds
        myShot.distance_total_game = row.distance_game
        myShot.velocity = row.velocity
        myShot.distance_closest_def = row.distance_closest_def
        myShot.angle_closest_def = row.angle_closest_def
        myShot.shot_distance = row.shot_dist
        myShot.shot_angle = row.shot_angle
        myShot.catch_and_shoot = row.catch_shoot
        myShot.result = row.result
        myShot.shot_clock = row.shot_clock
        myShot.distance_second_def = row.distance_second_def
        myShot.angle_second_def = row.angle_second_def
        myShot.distance_closest_teammate = row.distance_closest_teammate
        myShot.angle_closest_teammate = row.angle_closest_teammate
        myShot.offense_hull = row.offense_hull
        myShot.defense_hull = row.defense_hull
        myShot.value = row.value
        if(clusterNum not in clusterDict.keys()):
            clusterDict.update({clusterNum:[myShot]})
        else:
            clusterDict.get(clusterNum).append(myShot)
    dataLength = len(df)
    print("A total of " + str(dataLength) + " shots clustered into " + str(max(clusterDict.keys())) + " clusters was read")
    return



def splitClusters(trainingFraction = 0.8):
    readClusters()
    trainingSet = np.array([])
    testingSet = np.array([])
    for n in range(1,len(clusterDict.keys()) + 1):
        clusterList = clusterDict.get(n)
        splitIndex = round(trainingFraction * len(clusterList))
        trainingList = clusterList[:splitIndex]
        testingList = clusterList[splitIndex:]
        training = np.array(trainingList)
        testing = np.array(testingList)
        trainingSet = np.concatenate([training, trainingSet])
        testingSet = np.concatenate([testing, testingSet])
    trainTestPercent = (len(trainingSet) / (len(trainingSet) + len(testingSet))) * 100
    print("The training set has been made using " + str(trainTestPercent) + "% of the data")
    trainOrder = np.random.permutation(len(trainingSet))
    testOrder = np.random.permutation(len(testingSet))
    trainingSet = trainingSet[trainOrder]
    testingSet = testingSet[testOrder]
    return (trainingSet, testingSet)

def createDataFrame(shotSet):
    dist10 = []
    distGame = []
    vel = []
    shotClock = []
    distDef = []
    angleDef = []
    distShot = []
    angle2ndDef = []
    dist2ndDef = []
    angleMate = []
    distMate = []
    offenseHull = []
    defenseHull = []
    angleShot = []
    catch = []
    value = []
    result = []
    for shot in shotSet:
        dist10.append(shot.distance_ten_seconds)
        distGame.append(shot.distance_total_game)
        vel.append(shot.velocity)
        shotClock.append(shot.shot_clock)
        distDef.append(shot.distance_closest_def)
        angleDef.append(shot.angle_closest_def)
        distShot.append(shot.shot_distance)
        angleShot.append(shot.shot_angle)
        catch.append(shot.catch_and_shoot)
        result.append(shot.result)
        angle2ndDef.append(shot.angle_second_def)
        dist2ndDef.append(shot.distance_second_def)
        distMate.append(shot.distance_closest_teammate)
        angleMate.append(shot.angle_closest_teammate)
        offenseHull.append(shot.offense_hull)
        defenseHull.append(shot.defense_hull)
        value.append(shot.value)
    Shots = {'distance_ten_seconds':dist10,'distance_total_game':distGame,'velocity':vel,'distance_closest_def':distDef,
             'angle_closest_def':angleDef,'distance_second_def': dist2ndDef, 'angle_second_def': angle2ndDef, 'distance_closest_team': distMate, 'angle_closest_team': angleMate,
             'shot_distance':distShot,'shot_angle':angleShot,'offense_hull': offenseHull, 'defense_hull': defenseHull, 'shot_clock': shotClock, 'catch_and_shoot':catch, 
             'value': value, 'result':result}
    df = pd.DataFrame(data = Shots)
    return df

def writeCSV(testName = "TestingData.csv",trainName = "TrainingData.csv"):
    myTuple = splitClusters()
    trainingSet = myTuple[0]
    testingSet = myTuple[1]
    testdf = createDataFrame(testingSet)
    traindf = createDataFrame(trainingSet)
    trainCSV = traindf.to_csv(trainName,index = None, header = True)
    testCSV = testdf.to_csv(testName, index = None, header = True)
    return

if __name__ == '__main__':
    writeCSV("TestingData.csv", "TrainingData.csv")
    


