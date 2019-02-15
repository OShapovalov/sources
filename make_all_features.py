import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from sklearn.metrics import mean_squared_error as mse
import math
import os
from itertools import repeat
import pandas as pd
import swifter

pd.options.mode.chained_assignment = None


SIMPLE_FEATURE_COLUMNS = ['ncl[0]', 'ncl[1]', 'ncl[2]', 'ncl[3]', 'avg_cs[0]',
       'avg_cs[1]', 'avg_cs[2]', 'avg_cs[3]', 'ndof', 'MatchedHit_TYPE[0]',
       'MatchedHit_TYPE[1]', 'MatchedHit_TYPE[2]', 'MatchedHit_TYPE[3]',
       'MatchedHit_X[0]', 'MatchedHit_X[1]', 'MatchedHit_X[2]',
       'MatchedHit_X[3]', 'MatchedHit_Y[0]', 'MatchedHit_Y[1]',
       'MatchedHit_Y[2]', 'MatchedHit_Y[3]', 'MatchedHit_Z[0]',
       'MatchedHit_Z[1]', 'MatchedHit_Z[2]', 'MatchedHit_Z[3]',
       'MatchedHit_DX[0]', 'MatchedHit_DX[1]', 'MatchedHit_DX[2]',
       'MatchedHit_DX[3]', 'MatchedHit_DY[0]', 'MatchedHit_DY[1]',
       'MatchedHit_DY[2]', 'MatchedHit_DY[3]', 'MatchedHit_DZ[0]',
       'MatchedHit_DZ[1]', 'MatchedHit_DZ[2]', 'MatchedHit_DZ[3]',
       'MatchedHit_T[0]', 'MatchedHit_T[1]', 'MatchedHit_T[2]',
       'MatchedHit_T[3]', 'MatchedHit_DT[0]', 'MatchedHit_DT[1]',
       'MatchedHit_DT[2]', 'MatchedHit_DT[3]', 'Lextra_X[0]', 'Lextra_X[1]',
       'Lextra_X[2]', 'Lextra_X[3]', 'Lextra_Y[0]', 'Lextra_Y[1]',
       'Lextra_Y[2]', 'Lextra_Y[3]', 'NShared', 'Mextra_DX2[0]',
       'Mextra_DX2[1]', 'Mextra_DX2[2]', 'Mextra_DX2[3]', 'Mextra_DY2[0]',
       'Mextra_DY2[1]', 'Mextra_DY2[2]', 'Mextra_DY2[3]', 'FOI_hits_N', 'PT', 'P']

TRAIN_COLUMNS = ["label", "weight"]

FOI_COLUMNS = ["FOI_hits_X", "FOI_hits_Y", "FOI_hits_T",
               "FOI_hits_Z", "FOI_hits_DX", "FOI_hits_DY", "FOI_hits_S"]

ID_COLUMN = "id"

N_STATIONS = 4
FEATURES_PER_STATION = 6
N_FOI_FEATURES = N_STATIONS*FEATURES_PER_STATION
# The value to use for stations with missing hits
# when computing FOI features
EMPTY_FILLER = 1000


def parse_array(line, dtype=np.float32):
    return np.fromstring(line[1:-1], sep=" ", dtype=dtype)
    

def make_is_muon_features(df):
    p_items = ['P']
    m_MomRangeIsMuon = [3000., 6000., 10000.]

    stInStations_1 = df['MatchedHit_TYPE[0]'] == 2
    stInStations_2 = df['MatchedHit_TYPE[1]'] == 2
    stInStations_3 = df['MatchedHit_TYPE[2]'] == 2
    stInStations_4 = df['MatchedHit_TYPE[3]'] == 2

    is_muon = np.zeros(len(df))
    is_muon_loose = np.zeros(len(df))
    is_muon_dist = np.zeros(len(df))
    is_muon_loose_dist = np.zeros(len(df))

    i = 0
    for item, st_1, st_2, st_3, st_4 in tqdm(zip(df[p_items].values, stInStations_1, stInStations_2, stInStations_3, stInStations_4)):
        if item < m_MomRangeIsMuon[0]:
            is_muon[i] = 0.0
            is_muon_loose[i] = 0.0

            is_muon_loose_dist[i] = item - m_MomRangeIsMuon[0]
            is_muon_dist[i] = item - m_MomRangeIsMuon[0]
        else:
            if item < m_MomRangeIsMuon[1]:
                is_muon[i] = 1.0
                is_muon_loose[i] = 1.0

                is_muon_loose_dist[i] = max(item - m_MomRangeIsMuon[0], m_MomRangeIsMuon[1] - item)
                is_muon_dist[i] = max(item - m_MomRangeIsMuon[0], m_MomRangeIsMuon[1] - item)
            else:
                is_muon_loose[i] = 1.0 if st_3 or st_4 else 0.0

                if st_3 or st_4:
                    is_muon_loose_dist[i] = item - m_MomRangeIsMuon[0]
                else:
                    is_muon_loose_dist[i] = m_MomRangeIsMuon[1] - item

                if item < m_MomRangeIsMuon[2]:
                    is_muon[i] = 1.0 if st_3 or st_4 else 0.0

                    if st_3 or st_4:
                        is_muon_dist[i] = item - m_MomRangeIsMuon[0]
                    else:
                        is_muon_dist[i] = m_MomRangeIsMuon[1] - item

                else:
                    is_muon[i] = 1.0 if st_3 and st_4 else 0.0

                    if st_3 and st_4:
                        is_muon_dist[i] = item - m_MomRangeIsMuon[0]
                    else:
                        if st_3 or st_4:
                            is_muon_dist[i] = m_MomRangeIsMuon[2] - item
                        else:
                            is_muon_dist[i] = m_MomRangeIsMuon[1] - item

        i += 1

    df['is_muon'] = is_muon
    df['is_muon_loose'] = is_muon_loose
    df['is_muon_dist'] = is_muon_dist
    df['is_muon_loose_dist'] = is_muon_loose_dist


def make_p_features(df):
    p_items = ['P', 'PT']
    third = np.zeros(len(df))
    alfa = np.zeros(len(df))
    beta = np.zeros(len(df))
    alfa_sin = np.zeros(len(df))
    beta_sin = np.zeros(len(df))
    i = 0
    for item in tqdm(df[p_items].values):
        third[i] = np.sqrt(item[0]**2 - item[1]**2)
        alfa_sin[i] = item[1] / item[0]
        beta_sin[i] = third[i] / item[0]
        alfa[i] = math.atan2(item[1], third[i])
        beta[i] = math.atan2(third[i], item[1])
        i += 1

    df['P_third'] = third
    df['alfa'] = alfa
    df['beta'] = beta
    df['alfa_sin'] = alfa_sin
    df['beta_sin'] = beta_sin


def make_k_and_rmse_features(df):
    coords = ['MatchedHit_X[0]', 'MatchedHit_X[1]', 'MatchedHit_X[2]',
       'MatchedHit_X[3]', 'MatchedHit_Y[0]', 'MatchedHit_Y[1]',
       'MatchedHit_Y[2]', 'MatchedHit_Y[3]', 'MatchedHit_Z[0]',
       'MatchedHit_Z[1]', 'MatchedHit_Z[2]', 'MatchedHit_Z[3]', #'label'
       ]

    ksx = np.zeros(len(df))
    ksy = np.zeros(len(df))
    rmsex = np.zeros(len(df))
    rmsey = np.zeros(len(df))

    rmsex1 = np.zeros(len(df))
    rmsey1 = np.zeros(len(df))
    rmsex2 = np.zeros(len(df))
    rmsey2 = np.zeros(len(df))
    rmsex3 = np.zeros(len(df))
    rmsey3 = np.zeros(len(df))

    i = 0
    for item in tqdm(df[coords].values):
        x = np.array([item[0],item[1],item[2],item[3]]) - item[0]
        y = np.array([item[4],item[5],item[6],item[7]]) - item[4]
        z = np.array([item[8],item[9],item[10],item[11]]) - item[8]

        A = np.vstack([x, np.zeros(len(x))]).T
        res = np.linalg.lstsq(A, z, rcond=None)[0][0]
        ksx[i] = res
        rmsex[i] = np.sqrt(mse(z+item[8], x*res+item[8]))
        A = np.vstack([y, np.zeros(len(y))]).T
        res = np.linalg.lstsq(A, z, rcond=None)[0][0]
        ksy[i] = res
        rmsey[i] = np.sqrt(mse(z+item[8], y*res+item[8]))

        rmsex1[i] = np.abs(z[1] - x[1] * res)
        rmsex2[i] = np.abs(z[2] - x[2] * res)
        rmsex3[i] = np.abs(z[3] - x[3] * res)
        rmsey1[i] = np.abs(z[1] - y[1] * res)
        rmsey2[i] = np.abs(z[2] - y[2] * res)
        rmsey3[i] = np.abs(z[3] - y[3] * res)

        i+=1

    df['kx'] = ksx
    df['ky'] = ksy
    df['rmsex'] = rmsex
    df['rmsey'] = rmsey

    df['rmsex1'] = rmsex1
    df['rmsey1'] = rmsey1
    df['rmsex2'] = rmsex2
    df['rmsey2'] = rmsey2
    df['rmsex3'] = rmsex3
    df['rmsey3'] = rmsey3


def make_rectangle_features(df):
    coords = ['MatchedHit_X[0]', 'MatchedHit_X[1]', 'MatchedHit_X[2]',
              'MatchedHit_X[3]', 'MatchedHit_Y[0]', 'MatchedHit_Y[1]',
              'MatchedHit_Y[2]', 'MatchedHit_Y[3]', 'MatchedHit_Z[0]',
              'MatchedHit_Z[1]', 'MatchedHit_Z[2]', 'MatchedHit_Z[3]']
    station_1_rect = np.zeros(len(df))
    station_2_rect = np.zeros(len(df))
    station_3_rect = np.zeros(len(df))
    station_4_rect = np.zeros(len(df))
    i = 0
    for item in df[coords].values:
        if np.abs(item[0]) < 625 and np.abs(item[4]) < 500:
            station_1_rect[i] = 1
        elif np.abs(item[0]) < 1250 and np.abs(item[4]) < 1000:
            station_1_rect[i] = 2
        elif np.abs(item[0]) < 2500 and np.abs(item[4]) < 2000:
            station_1_rect[i] = 3
        else:
            station_1_rect[i] = 4

        if np.abs(item[1]) < 650 and np.abs(item[5]) < 520:
            station_2_rect[i] = 1
        elif np.abs(item[1]) < 1300 and np.abs(item[5]) < 1040:
            station_2_rect[i] = 2
        elif np.abs(item[1]) < 2600 and np.abs(item[5]) < 2080:
            station_2_rect[i] = 3
        else:
            station_2_rect[i] = 4

        if np.abs(item[2]) < 700 and np.abs(item[6]) < 560:
            station_3_rect[i] = 1
        elif np.abs(item[2]) < 1400 and np.abs(item[6]) < 1120:
            station_3_rect[i] = 2
        elif np.abs(item[2]) < 2800 and np.abs(item[6]) < 2240:
            station_3_rect[i] = 3
        else:
            station_3_rect[i] = 4

        if np.abs(item[3]) < 750 and np.abs(item[7]) < 600:
            station_4_rect[i] = 1
        elif np.abs(item[3]) < 1500 and np.abs(item[7]) < 1200:
            station_4_rect[i] = 2
        elif np.abs(item[3]) < 3000 and np.abs(item[7]) < 2400:
            station_4_rect[i] = 3
        else:
            station_4_rect[i] = 4

        i += 1

    df['station_1_rect'] = station_1_rect
    df['station_2_rect'] = station_2_rect
    df['station_3_rect'] = station_3_rect
    df['station_4_rect'] = station_4_rect

    
def load_csv(path, is_train):
    converters = dict(zip(FOI_COLUMNS, repeat(parse_array)))
    if is_train:
        types = dict(zip(SIMPLE_FEATURE_COLUMNS + TRAIN_COLUMNS, repeat(np.float32)))
        train = pd.read_csv(path,
                           index_col="id", converters=converters,
                           dtype=types,
                           usecols=[ID_COLUMN]+SIMPLE_FEATURE_COLUMNS+FOI_COLUMNS+TRAIN_COLUMNS)
        return train
    else:
        types = dict(zip(SIMPLE_FEATURE_COLUMNS, repeat(np.float32)))
        test = pd.read_csv(path,
                           index_col="id", converters=converters,
                           dtype=types,
                           usecols=[ID_COLUMN]+SIMPLE_FEATURE_COLUMNS+FOI_COLUMNS)
        return test
  
  
def find_closest_hit_per_station(row):
    result = np.empty(N_FOI_FEATURES+3*4, dtype=np.float32)
    closest_x_per_station = result[0:4]
    closest_y_per_station = result[4:8]
    closest_T_per_station = result[8:12]
    closest_z_per_station = result[12:16]
    closest_dx_per_station = result[16:20]
    closest_dy_per_station = result[20:24]

    closest_x = result[24:28]
    closest_y = result[28:32]
    closest_z = result[32:36]
    
    for station in range(4):
        hits = (row["FOI_hits_S"] == station)
        if not hits.any():
            closest_x_per_station[station] = EMPTY_FILLER
            closest_y_per_station[station] = EMPTY_FILLER
            closest_T_per_station[station] = EMPTY_FILLER
            closest_z_per_station[station] = EMPTY_FILLER
            closest_dx_per_station[station] = EMPTY_FILLER
            closest_dy_per_station[station] = EMPTY_FILLER
        else:
            x_distances_2 = (row["Lextra_X[%i]" % station] - row["FOI_hits_X"][hits])**2
            y_distances_2 = (row["Lextra_Y[%i]" % station] - row["FOI_hits_Y"][hits])**2
            distances_2 = x_distances_2 + y_distances_2

            closest_hit = np.argmin(distances_2)

            closest_x_per_station[station] = x_distances_2[closest_hit]
            closest_y_per_station[station] = y_distances_2[closest_hit]
            closest_T_per_station[station] = row["FOI_hits_T"][hits][closest_hit]
            closest_z_per_station[station] = row["FOI_hits_Z"][hits][closest_hit]
            closest_dx_per_station[station] = row["FOI_hits_DX"][hits][closest_hit]
            closest_dy_per_station[station] = row["FOI_hits_DY"][hits][closest_hit]
            closest_x[station] = row["FOI_hits_X"][hits][closest_hit]
            closest_y[station] = row["FOI_hits_Y"][hits][closest_hit]
            closest_z[station] = row["FOI_hits_Z"][hits][closest_hit]

    return result
  

def load_train_hdf(path):
    return pd.concat([
        pd.read_hdf(os.path.join(path, "train_part_%i_v2.hdf" % i))
            for i in (1, 2)], axis=0, ignore_index=True)
  
    
def make_all_features(inp_file, is_train, result_file, is_csv):   
    if is_csv:
        data = load_csv(inp_file, is_train)
    else:
	    if is_train:
	        data = load_train_hdf(inp_file)
	    else:
	        data = pd.read_hdf(inp_file, axis=0, ignore_index=True)
		
    print('Total rows: ', len(data))
	
    fails2 = data['MatchedHit_X[2]'] == -9999
    fails3 = data['MatchedHit_X[3]'] == -9999

    data['MatchedHit_X[2]'][fails2] = data['Lextra_X[2]'][fails2]
    data['MatchedHit_Y[2]'][fails2] = data['Lextra_Y[2]'][fails2]
    data['MatchedHit_Z[2]'][fails2] = 17621

    data['MatchedHit_X[3]'][fails3] = data['Lextra_X[3]'][fails3]
    data['MatchedHit_Y[3]'][fails3] = data['Lextra_Y[3]'][fails3]
    data['MatchedHit_Z[3]'][fails3] = 18854

    make_is_muon_features(data)
    make_p_features(data)
    make_k_and_rmse_features(data)
    make_rectangle_features(data)
    
    # This will take a while... We welcome your solutions on fast processing of jagged arrays
    closest_hits_features = data.swifter.apply(find_closest_hit_per_station, result_type="expand", axis=1)

    NEW_FEATURES = ["kx", "ky",
                    'rmsex', 'rmsey',
                    'rmsex1', 'rmsey1','rmsex2', 'rmsey2','rmsex3', 'rmsey3',
                    'station_1_rect', 'station_2_rect', 'station_3_rect' ,'station_4_rect',
                    'P_third', 'alfa', 'beta', 'alfa_sin', 'beta_sin',
                    'is_muon','is_muon_loose', 'is_muon_dist', 'is_muon_loose_dist'
                    ]
    if is_train:
        data = pd.concat(
            [data.loc[:, SIMPLE_FEATURE_COLUMNS + TRAIN_COLUMNS],
             closest_hits_features, data.loc[:, NEW_FEATURES]], axis=1)
    else:
        data = pd.concat(
            [data.loc[:, SIMPLE_FEATURE_COLUMNS],
             closest_hits_features, data.loc[:, NEW_FEATURES]], axis=1)

    data.to_hdf(result_file, 'data')
        
        
make_all_features('../IDAO-MuID/test_private_v2_track_1.hdf', False, 'test_private.h5', False)
#make_all_features('../IDAO-MuID/test_private_v3_track_1.csv.gz', False, 'test_private_2.h5', True)
#make_all_features('../IDAO-MuID', True, 'train_exp.h5')