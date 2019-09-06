'''
Created on Jun 19, 2019

@author: anshul
'''

import pandas as pd
from sklearn.preprocessing import scale
import numpy as np


def get_testing():
    test_features = []
    test_results = []
    values = []
    df = pd.read_csv("data/TestingData.csv")
    for row in df.itertuples():
        test_features.append([row.distance_ten_seconds, row.distance_total_game, row.velocity , row.distance_closest_def, row.angle_closest_def, row.distance_second_def,
                            row.angle_second_def,  row.angle_closest_team, row.distance_closest_team, row.shot_distance , row.shot_angle, row.offense_hull, 
                            row.defense_hull, row.shot_clock, row.catch_and_shoot])
        
        test_results.append(row.result)
        values.append(row.value)
    return np.asmatrix(test_features), np.asarray(test_results), values

def get_training():
    features = []
    results = []
    df = pd.read_csv("data/TrainingData.csv")
    for row in df.itertuples():
        features.append([row.distance_ten_seconds, row.distance_total_game, row.velocity , row.distance_closest_def, row.angle_closest_def, row.distance_second_def,
                        row.angle_second_def, row.angle_closest_team, row.distance_closest_team, row.shot_distance , row.shot_angle, row.offense_hull, 
                        row.defense_hull, row.shot_clock, row.catch_and_shoot])
        results.append(row.result)
    return np.asmatrix(features), np.asarray(results)



if __name__ == '__main__':
    get_testing()

    get_training()