'''
Created on Jun 18, 2019

@author: anshul
'''
import math as m
import pandas as pd
from collections import OrderedDict
import operator
import scipy.spatial as sp
import numpy as np
import SVM

eventDict = {}
eventDescriptors = OrderedDict()
gameDict = {}

class Coordinate:
    def __init__(self, x, y):
        self.x = x 
        self.y = y 

class Player:
    # In the case of no name being provided, the name is unknown + the player's ID.
    # Unlike a name, a player ID is required to manipulate the SportVU Data
    def __init__(self, playerID):
        self.id=playerID
        self.positions = {}
        return
    # High Level Structure: Player = (playerID, Games) -> Games = {gameID:Locs} -> Loc[ation]s = {timestamp:Coordinates}
    # This method updates the positions dictionary storing the game ID as a key and 
    # the gameLocs dictionary storing the player locations
    def addGame(self, gameID):
        gameLocs = OrderedDict()
        self.positions.update({gameID : gameLocs})
        return    
    
    # This method returns the gameLocs dictionary given a game ID. 
    def getGame(self, gameID):
        return self.positions.get(gameID)
    
    def addLocs(self, t, coord, gameID):
        self.getGame(gameID).update({t:coord})
        return
    
    def getLoc(self, t, gameID):
        return self.getGame(gameID).get(t)
    
    
    # Returns distance travelled from t_0 to t_f as a sum of Euclidean distances.
    def get_distance(self, gameID, t_end, t_start):#, continuous = False):
        locs = self.getGame(gameID)
        dist=0
        tList = list(locs.keys())
        tList[:] = [t for t in tList if(t <= t_start and t >= t_end)]

        if(len(tList) > 0):
            for n in range(-1, -(len(tList)), -1):
                if(eventDict[gameID][tList[n - 1]] in [8,11,15]):
                    break
                playDist = distance(locs.get(tList[n]), locs.get(tList[n-1]))
#                dist0 = distanceTo(self.positions.get(t_0),self.positions.get(tList[0]))
#                dist = dist + dist0
            for n in range(len(tList) - 1):
                if(eventDict.get(tList[n]) in [8,11,15]):
                    continue
                playDist = distance(locs.get(tList[n]), locs.get(tList[n+1]))
                dist = playDist + dist
        return dist
    
    # Finds speed between 2 times t1 and t2, assuming velocity is constant
    # If t1 is not provided as an argument, it defaults to calculating instantaneous velocity
    def get_velocity(self, gameID):
        times = list(self.positions[gameID].keys())
        p1 = self.positions.get(gameID).get(times[-1])
        p2 = self.positions.get(gameID).get(times[-2])
        
        dist=distance(p1, p2)
        time= times[-2] - times[-1]

        return abs(dist / time)


def get_convex_hull(locations):
    array = []
    for i in locations:
        array.append([i.x, i.y])
    hullArray = np.array(array)
    hull = sp.ConvexHull(hullArray)
    return hull.volume

def distance(coord_1, coord_2):
    #calculate distance between two points
    dist = m.sqrt((coord_1.x - coord_2.x)**2 + (coord_1.y - coord_2.y)**2)
    return dist

def calculate_angle(shooter_location, player_to_shooterDist, player_location):
    hoop = hoopChooser(shooter_location)
    x = player_to_shooterDist
    y = distance(player_location, hoop)
    z = distance(shooter_location, hoop)
    if x == 0:
        angle = 0
    else: 
        angle = m.acos((x**2 + z**2 - y**2) / (2 * x * z))
    return (angle, player_to_shooterDist, z)
    
def closest_teammate(shooter, offenseList):
    ret = []
    for each in offenseList:
        
        temp = distance(shooter, each)
        ret.append((temp, each))
    ret = sorted(ret, key = operator.itemgetter(0))
    closest_teammate_angle, closest_teammate_distance, shot_distance = calculate_angle(shooter, ret[1][0], ret[1][1])
    return closest_teammate_angle, closest_teammate_distance
    
def defensive_pressure(shooter, defenseList):
    ret = []
    for defender in defenseList:
        dist = distance(shooter, defender)
        ret.append((dist, defender))
    ret = sorted(ret, key = operator.itemgetter(0))
    (closest_def_angle, closest_def_distance, shot_distance) = calculate_angle(shooter, ret[0][0], ret[0][1])
    (second_def_angle, second_def_distance, blah) = calculate_angle(shooter, ret[1][0], ret[1][1])
    return closest_def_angle, closest_def_distance, second_def_angle, second_def_distance, shot_distance


def get_angle(coord):
    hoop = hoopChooser(coord)
    deltaX = coord.x - hoop.x
    deltaY = coord.y - hoop.y
    return m.atan(deltaY/deltaX)

def get_catch_and_shoot(touch):
    if touch:
        return 1
    return 0

def get_second_chance(off_reb):
    if off_reb:
        return 1
    return 0

def get_shot_clock(time_left):
    if time_left >= 27:
        return 1
    if time_left >= 5:
        return 0
    return -1

def hoopChooser(playerCoord):  
    #since there's two hoops, we need to decide which hoop a player is shooting at, so we need a function to decide 
    hoopDown = Coordinate(25, 5.25)
    hoopUp = Coordinate(25, 88.75)
    if playerCoord.y > 47:
        return hoopUp
    return hoopDown 


class Shot:
    def __init__(self, player):
        self.shooter = player
        self.gameID = 0
        self.distance_ten_seconds = 0 #DONE
        self.distance_total_game = 0 #DONE
        self.velocity = 0 #DONE
        self.distance_closest_def = 0 #DONE
        self.angle_closest_def = 0  #DONE
        self.distance_second_def = 0
        self.angle_second_def = 0
        self.shot_distance = 0 #DONE
        self.shot_angle = 0 #DONE
        self.angle_closest_teammate = 0 #DONE
        self.distance_closest_teammate = 0 #DONE
        self.offense_convex_hull = 0
        self.defense_convex_hull = 0
        self.shot_clock = 0
        self.catch_and_shoot = 0 #DONE
        self.result = 0 #DONE
        self.value = 2
        self.x = 0
        self.y = 0


def get_data():
    df = pd.read_csv("data/all_games.csv")
    df.columns = df.columns.str.replace('.', '_')
    shot_dict = {}
    touch = False
    off_reb = False
    playerDict = {}
    marker = 0
    count = 0
    total = 0
    lst = []
    for row in df.itertuples():
        if row.game_id not in [20150125, 20150117]:
            half = row.half
            time = round(row.game_clock + (1200 * (2 - half)),2)
            gameID = row.game_id
            eventID = row.event_id
            if gameID not in shot_dict:
                shot_dict[gameID] = {}
            if gameID not in eventDict:
                eventDict[gameID] = {}
            eventDict[gameID][time] = row.event_id
            
            duke_players = [row.p1_global_id, row.p2_global_id, row.p3_global_id, row.p4_global_id, row.p5_global_id]
            duke_location = [Coordinate(row.p1_x, row.p1_y), Coordinate(row.p2_x, row.p2_y), Coordinate(row.p3_x, row.p3_y), Coordinate(row.p4_x, row.p4_y), Coordinate(row.p5_x, row.p5_y)]
            
            for i in range(5):
                if duke_players[i] not in playerDict:
                    playerDict[duke_players[i]] = Player(duke_players[i])
                    
                player = playerDict[duke_players[i]]
                
                if gameID not in player.positions:
                    player.addGame(gameID)
                player.addLocs(time, duke_location[i], gameID)                    
            
            if eventID == 23:
                touch = True
            if eventID == 21:
                touch = False
            if eventID == 5:
                off_reb = True
            if eventID in [7, 8, 11]:
                off_reb = False
            if eventID in [3,4] and row.home == "yes":
                opponent_location = [Coordinate(row.p6_x, row.p6_y), Coordinate(row.p7_x, row.p7_y), Coordinate(row.p8_x, row.p8_y), Coordinate(row.p9_x, row.p9_y), Coordinate(row.p10_x, row.p10_y)]
                shooter = row.global_player_id
                
                if time not in shot_dict[gameID]:
                    shot_dict[gameID][time] = None
                
                shooterLocation = duke_location[row.p_poss - 1]
                shot = Shot(shooter)
                shot.gameID = gameID
                shot.x = shooterLocation.x
                shot.y = shooterLocation.y
                shot.distance_ten_seconds = playerDict[shooter].get_distance(gameID, time, time + 10)
                
                shot.distance_total_game = playerDict[shooter].get_distance(gameID, time, 2400)
                
                shot.velocity = playerDict[shooter].get_velocity(gameID)
                
                shot.angle_closest_def, shot.distance_closest_def, shot.angle_second_def, shot.distance_second_def, shot.shot_distance = defensive_pressure(shooterLocation, opponent_location)
                if shot.shot_distance > 20.75:
                    shot.value = 3
                
                shot.shot_angle = get_angle(shooterLocation)
                
                shot.angle_closest_teammate, shot.distance_closest_teammate = closest_teammate(shooterLocation, duke_location)
                
                shot.catch_and_shoot = get_catch_and_shoot(touch)
                
                shot.second_chance = get_second_chance(off_reb)
                
                shot_clock = row.shot_clock
                
                shot.offense_convex_hull = get_convex_hull(duke_location)
                
                shot.defense_convex_hull = get_convex_hull(opponent_location)
                if(pd.isnull(shot_clock) and row.game_clock < 35):
                    shot_clock = row.game_clock
                if(pd.isnull(shot_clock) and row.game_clock > 35):
                    shot_clock = marker - time
                    if shot_clock < 0:
                        shot_clock = 2400 + shot_clock
                    if shot_clock > 35:
                        shot_clock = 35
                shot.shot_clock = shot_clock
                lst.append(shot_clock)
                if(row.event_id == 3):
                    shot.result = 1
                shot_dict[gameID][time] = shot
                #shot_dict[shooter].append(shot)
                off_reb = False
            if eventID in [1, 2, 3, 4, 5, 6, 7, 8]:
                marker = time
    #print(count, total)
    return shot_dict


def create_dataframe():
    d = get_data()
    data = []
    table = []
    shot_num = 0
    for game, timeDict in d.items():
        for time, shot in timeDict.items():
            data.append([shot.distance_ten_seconds, shot.distance_total_game, shot.velocity , shot.distance_closest_def, shot.angle_closest_def, shot.distance_second_def,
                            shot.angle_second_def, shot.angle_closest_teammate, shot.distance_closest_teammate, shot.shot_distance , shot.shot_angle,  shot.offense_convex_hull, 
                            shot.defense_convex_hull, shot.shot_clock, shot.catch_and_shoot])
            table.append([game, time, shot_num, shot.value, shot.result])
            shot_num += 1
            """data.append([shot.shooter, shot.gameID, shot.distance_ten_seconds, shot.distance_total_game, shot.velocity , shot.distance_closest_def, shot.angle_closest_def, shot.distance_second_def,
                            shot.angle_second_def, shot.shot_distance , shot.shot_angle, shot.angle_closest_teammate, shot.distance_closest_teammate, shot.offense_convex_hull, 
                            shot.defense_convex_hull, shot.shot_clock, shot.catch_and_shoot, shot.result, shot.x, shot.y, probability])"""
    #df = pd.DataFrame(data, columns = ["shooter", "gameID", "value", "x", "y", "probability"])
    """df = pd.DataFrame(data, columns = ["shooterID", "gameID", "distance_ten_seconds", "distance_game", "velocity", "distance_closest_def", "angle_closest_def", "distance_second_def", "angle_second_def", 
                                       "angle_closest_teammate", "distance_closest_teammate", "shot_dist", "shot_angle", 
                                        "offense_hull", "defense_hull", "shot_clock", "catch_shoot"])"""
    
    return(data, table)

    
if __name__ == '__main__':
    #create_dataframe()
    print(get_data())