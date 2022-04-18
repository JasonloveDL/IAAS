import datetime
import os.path
import pickle

import numpy as np
import pandas as pd
import torch

from .logger import get_logger

logger = get_logger('data preprocess')


def _preprocess_data(origin_data: pd.DataFrame, csv_result_path: str):
    if os.path.exists(csv_result_path):
        return
    origin_data.loc[:, 'HourEnding'].apply(lambda x: x)
    start_time = datetime.datetime.strptime(origin_data.loc[0, 'HourEnding'], '%Y/%m/%d %H:%M')
    for index in origin_data.loc[:, 'HourEnding'].index:
        cur_time = datetime.timedelta(hours=index) + start_time
        origin_data.loc[index, 'HourEnding'] = cur_time
        origin_data.loc[index, 'DayOfWeek'] = cur_time.weekday()
        origin_data.loc[index, 'HourOfDay'] = cur_time.hour
    origin_data.to_csv(csv_result_path)


def _generate_feature_value_load_scale(time_length: int, csv_result_path: str, pkl_data_path: str):
    data = pd.read_csv(csv_result_path)
    data.loc[:, 'RT_Demand'] = data.loc[:, 'RT_Demand'].apply(lambda x: float(x.replace(',', '') if type(x)==str else x))
    # data.loc[:, 'RT_Demand'] = (data.loc[:, 'RT_Demand'] - data.loc[:, 'RT_Demand'].min()) / \  # normalize to range (0,1)
    #                            ((data.loc[:, 'RT_Demand'].max() - data.loc[:, 'RT_Demand'].min()))
    predict_length = 24
    X, y = [], []
    for i in range(time_length, data.shape[0] - predict_length):
        X.append(data.loc[i - time_length:i - 1, ['RT_Demand', 'DayOfWeek', 'HourOfDay']].to_numpy())
        y.append(data.loc[i + predict_length - 1, 'RT_Demand'])
    X = np.array(X)
    y = np.array(y)
    with open(pkl_data_path, 'wb') as f:
        pickle.dump({'X': X, 'y': y}, f)


def _generate_feature_value_wind_power(origin_data, time_length: int, csv_result_path: str, pkl_data_path: str):
    origin_data.loc[:, '时间'] = origin_data.loc[:, '时间'].apply(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d %H:%M'))
    # change data into one point each hour
    origin_data = origin_data.loc[origin_data.loc[:, '时间'].apply(lambda x: x.minute) == 0]
    origin_data.loc[:, 'HourOfDay'] = origin_data.loc[:, '时间'].apply(lambda x:x.hour)
    origin_data.loc[:,'短期预测风向sin'] = np.sin(origin_data.loc[:, '短期预测风向'] * np.pi / 180)
    origin_data.loc[:,'短期预测风向cos'] = np.cos(origin_data.loc[:, '短期预测风向'] * np.pi / 180)
    # for col in ['短期预测湿度','短期预测风速','短期预测温度','短期预测气压','实际功率','实际风速']:
    #     origin_data.loc[:,col] = (origin_data.loc[:,col] - origin_data.loc[:,col].min()) / \
    #                              (origin_data.loc[:, col].max() - origin_data.loc[:, col].min())

    origin_data = origin_data.dropna()
    origin_data.to_csv(csv_result_path)
    seasons = ['spring', 'summer', 'autumn', 'winter']
    save_dict = {}
    for season_idx in range(1, 5):
        data = origin_data.loc[origin_data.loc[:, '时间'].apply(lambda x: x.quarter) == season_idx]
        data = data.reset_index()
        predict_length = 24  # 24 hour prediction
        X, y = [], []
        for i in range(time_length, data.shape[0] - predict_length):
            x_real = data.loc[i - time_length:i - 1,
                     ['实际功率','实际风速','HourOfDay']]\
                .to_numpy()
            x_nwp = data.loc[i - time_length + predict_length:i + predict_length - 1,
                     ['短期预测湿度','短期预测风速','短期预测温度','短期预测风向sin','短期预测风向cos','短期预测气压']]\
                .to_numpy()
            X.append(np.concatenate([x_real, x_nwp], axis=1))
            y.append(data.loc[i + predict_length - 1, '实际功率'])
        X = np.array(X)
        y = np.array(y)
        save_dict.update({seasons[season_idx-1]: (X, y)})
    with open(pkl_data_path, 'wb') as f:
        pickle.dump(save_dict, f)


def _load_feature_value(pkl_data_path: str):
    with open(pkl_data_path, 'rb') as f:
        data = pickle.load(f)
        X = data['X']
        y = data['y']
        return X, y


def _load_feature_value_wind_power(pkl_data_path: str):
    with open(pkl_data_path, 'rb') as f:
        data = pickle.load(f)
        spring = data['spring']
        summer = data['summer']
        autumn = data['autumn']
        winter = data['winter']
        spring_X = torch.Tensor(spring[0]).permute(0, 2, 1)
        summer_X = torch.Tensor(summer[0]).permute(0, 2, 1)
        autumn_X = torch.Tensor(autumn[0]).permute(0, 2, 1)
        winter_X = torch.Tensor(winter[0]).permute(0, 2, 1)
        test_length = 24 * 5
        spring = train_test_fix_test_length_spilt(spring_X, torch.Tensor(spring[1]), test_length)
        summer = train_test_fix_test_length_spilt(summer_X, torch.Tensor(summer[1]), test_length)
        autumn = train_test_fix_test_length_spilt(autumn_X, torch.Tensor(autumn[1]), test_length)
        winter = train_test_fix_test_length_spilt(winter_X, torch.Tensor(winter[1]), test_length)
        return spring, summer, autumn, winter


def get_data(data_root, place, time_length: int):
    """
    preprocessing data from original data and return the numpy result
    if data is preprocessed before, return the previously stored result
    :param data_root:
    :param place:
    :param time_length: length of time in feature
    :return: X,y is feature, target
    """
    pkl_data_path = os.path.join(data_root, f'load_{place}_{time_length}.pkl')
    original_data_path = os.path.join(data_root, f'load_original_{place}.csv')
    if os.path.exists(pkl_data_path):
        return _load_feature_value(pkl_data_path)
    csv_result_path = os.path.join(data_root, f'load_{place}_{time_length}.csv')
    origin_data = pd.read_csv(os.path.join(data_root, original_data_path))
    _preprocess_data(origin_data, csv_result_path)
    _generate_feature_value_load_scale(time_length, csv_result_path, pkl_data_path)
    return _load_feature_value(pkl_data_path)


def get_scaled_data(data_root, place, time_length: int):
    """
    preprocessing data from original data and return the numpy result
    if data is preprocessed before, return the previously stored result
    :param data_root:
    :param place:
    :param time_length: length of time in feature
    :return: X,y is feature, target
    """
    pkl_data_path = os.path.join(data_root, f'load_{place}_{time_length}_scaled.pkl')
    original_data_path = os.path.join(data_root, f'load_original_{place}.csv')
    if os.path.exists(pkl_data_path):
        return _load_feature_value(pkl_data_path)
    csv_result_path = os.path.join(data_root, f'load_{place}_{time_length}_scaled.csv')
    origin_data = pd.read_csv(os.path.join(data_root, original_data_path))
    _preprocess_data(origin_data, csv_result_path)
    _generate_feature_value_load_scale(time_length, csv_result_path, pkl_data_path)
    return _load_feature_value(pkl_data_path)


def get_scaled_data_wind_power(data_root, place, time_length: int):
    """
    preprocessing data from original data and return the numpy result
    if data is preprocessed before, return the previously stored result
    :param data_root:
    :param place:
    :param time_length: length of time in feature
    :return: X,y is feature, target
    """
    pkl_data_path = os.path.join(data_root, f'{place}_{time_length}.pkl')
    original_data_path = os.path.join(data_root, f'{place}.csv')
    if os.path.exists(pkl_data_path):
        return _load_feature_value_wind_power(pkl_data_path)
    csv_result_path = os.path.join(data_root, f'{place}_{time_length}.csv')
    origin_data = pd.read_csv(os.path.join(data_root, original_data_path))
    _generate_feature_value_wind_power(origin_data, time_length, csv_result_path, pkl_data_path)
    return _load_feature_value_wind_power(pkl_data_path)


def train_test_split(X: torch.Tensor, y: torch.Tensor, test_ratio):
    length = X.shape[0]
    train_length = int(length * (1 - test_ratio))
    X_train, y_train, X_test, y_test = X[:train_length], y[:train_length], X[train_length:], y[train_length:]
    return X_train, y_train, X_test, y_test


def train_test_fix_test_length_spilt(X: torch.Tensor, y: torch.Tensor, test_length):
    length = X.shape[0]
    train_length = length - test_length
    X_train, y_train, X_test, y_test = X[:train_length], y[:train_length], X[train_length:], y[train_length:]
    return X_train, y_train, X_test, y_test


def four_season_train_test_split(X: torch.Tensor, y: torch.Tensor):
    spring_end =24 * (31 + 29 + 31)
    summer_end =24 * (30 + 31 + 30) + spring_end
    autumn_end =24 * (31 + 31 + 30) + summer_end
    test_length = 24 * 5
    spring = train_test_fix_test_length_spilt(X[:spring_end], y[:spring_end], test_length)
    summer = train_test_fix_test_length_spilt(X[spring_end: summer_end], y[spring_end: summer_end], test_length)
    autumn = train_test_fix_test_length_spilt(X[summer_end: autumn_end], y[summer_end: autumn_end], test_length)
    winter = train_test_fix_test_length_spilt(X[autumn_end:], y[autumn_end:] , test_length)
    return spring, summer, autumn, winter

