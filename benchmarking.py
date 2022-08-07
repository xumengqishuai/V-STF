import numpy as np
import os
import pickle
import argparse

import torch
from benchmarkutils import *
from ha import HistoricalAverage
from rfr import RandomForestRegressor
from svr import SupportVectorRegressor
from ann import TrafficANN
from gnn import TrafficGNN
from linear import LinearRegressor
from tqdm import tqdm


def load_model(model_str, sensor, feature_dimension, history):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_str == 'ha':
        model_state = HistoricalAverage(sensor)
        model_full = HistoricalAverage(sensor)
    elif model_str == 'rfr':
        model_state = RandomForestRegressor(n_estimators=50, mode='state')
        model_full = RandomForestRegressor(n_estimators=50, mode='full')
    elif model_str == 'svr':
        model_state = SupportVectorRegressor(kernel='rbf', mode='state')
        model_full = SupportVectorRegressor(kernel='rbf', mode='full')
    elif model_str == 'fnn':
        model_state = TrafficANN(feature_dimension, 200, 100, mode='state').to(device)
        model_full = TrafficANN(feature_dimension, 200, 100, mode='full').to(device)
    elif model_str == 'gnn':
        model_state = TrafficGNN(feature_dimension, 200, 100, mode='state').to(device)
        model_full = TrafficGNN(feature_dimension, 200, 100, mode='full').to(device)
    elif model_str == 'V-STF':
        model_state = TrafficGNN(feature_dimension, 200, 100, mode='state').to(device)
        model_full = TrafficGNN(feature_dimension, 200, 100, mode='full').to(device)
    elif model_str == 'linear':
        model_state = LinearRegressor(estimator='linear', mode='state')
        model_full = LinearRegressor(estimator='linear', mode='full')
    elif model_str == 'linear-ridge':
        model_state = LinearRegressor(estimator='ridge', alpha=10, mode='state')
        model_full = LinearRegressor(estimator='ridge', alpha=10, mode='full')
    else:
        model_state = LinearRegressor(estimator='lasso', alpha=10, mode='state')
        model_full = LinearRegressor(estimator='lasso', alpha=10, mode='full')

    return model_state, model_full

def main():
    print("====start====")
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, type=str, help='string to identify model')
    args = parser.parse_args()


    community = 'gurnee'
    sensors = [28]

    model_str = args.model
    if model_str not in ['V-STF','ha', 'rfr', 'svr', 'fnn', 'gnn', 'linear', 'linear-lasso', 'linear-ridge']:
        raise ValueError('Select ha, rfr, svr, ann, gnn, linear, linear-lasso, or linear-ridge')


    graph_path = os.path.join('graphs', community, '{}-graph.json'.format(community))

    traffic_data_path_train = os.path.join('trafficdata', '{}-traffic-dictionary.json'.format(community))
    traffic_data_path_test = os.path.join('trafficdata', '{}-traffic-dictionary_原始数据.json'.format(community))

    training_dates = ['2019-6-5', '2019-6-6', '2019-6-7',
                      '2019-6-8', '2019-6-9', '2019-6-10', '2019-6-11', '2019-6-12',
                      '2019-6-13']
    testing_dates = ['2019-6-14']

    A, D, S = load_traffic_graph(graph_path)

    T = 5

    n_trials = 10

    results_dict = {}
    traffic_flow = {}
    for s in sensors:
        for test_date in testing_dates:
            for K in [2]:
                neighborhood_dict, neighbors = load_K_hop_neighborhood(A, s, S, K, skip_outbound=False)
                traffic_data_train = load_traffic_data(traffic_data_path_train, training_dates + testing_dates)
                traffic_data_test = load_traffic_data(traffic_data_path_test, training_dates + testing_dates)
                neighborhood_data_train = load_neighborhood_data(traffic_data_train, neighbors)
                neighborhood_data_test = load_neighborhood_data(traffic_data_test, neighbors)
                interpolated_data_train, n_samples = interpolate_traffic_data(neighborhood_data_train, T)
                interpolated_data_test, n_samples = interpolate_traffic_data(neighborhood_data_test, T)
                for history in [6]:
                    for horizon in [3, 6, 9]:
                        for use_state in [True]:
                            # 结果字典吗？
                            results_dict[(s, test_date, K, history, horizon, use_state)] = {'state': [], 'full': []}
                            traffic_flow[(horizon, use_state)] = {'state': {}, 'full': {}, 'ground_truth': {}}
                            # 载入训练数据 normalize是正则化
                            X_train, y_data_train, y_state_train, max_dict = load_train_data(training_dates,
                                                                                             interpolated_data_train,
                                                                                             neighborhood_dict,
                                                                                             neighbors,
                                                                                             history,
                                                                                             horizon,
                                                                                             n_samples,
                                                                                             use_state=use_state,
                                                                                             normalize_max=False)
                            # 载入测试数据
                            X_test, y_data_test, y_state_test, sample_indices = load_test_data(test_date,
                                                                                               interpolated_data_test,
                                                                                               neighborhood_dict,
                                                                                               neighbors,
                                                                                               history,
                                                                                               horizon,
                                                                                               n_samples,
                                                                                               max_dict,
                                                                                               use_state=use_state)
                            # 载入模型
                            model_state, model_full = load_model(model_str, s, X_train.shape[1], history)
                            for n in range(n_trials):
                                if model_str == 'ha':
                                    model_state.fit(interpolated_data_train, training_dates)
                                    model_full.fit(interpolated_data_train, training_dates)
                                    preds_state = model_state.predict(y_state_test, sample_indices)
                                    preds_full = model_full.predict(y_state_test, sample_indices, mode='full')
                                else:
                                    model_state.fit_state(X_train, y_data_train, y_state_train)
                                    model_full.fit_full(X_train, y_data_train)

                                    preds_state = model_state.predict(X_test, y_state_test)
                                    # print("state: ", preds_state)
                                    preds_full = model_full.predict(X_test, y_state_test)

                                traffic_flow[(horizon, use_state)]['state'][n] = preds_state
                                traffic_flow[(horizon, use_state)]['full'][n] = preds_full
                                traffic_flow[(horizon, use_state)]['ground_truth'][n] = y_data_test

                                state_result_MAE = mae(preds_state, y_data_test)
                                full_result_MAE = mae(preds_full, y_data_test)

                                state_result_MAPE = mape(preds_state, y_data_test)
                                full_result_MAPE = mape(preds_full, y_data_test)

                                state_result_RMSE = rmse(preds_state, y_data_test)
                                full_result_RMSE = rmse(preds_full, y_data_test)

                                results_dict[(s, test_date, K, history, horizon, use_state)]['state'].append(
                                    state_result_MAE)
                                results_dict[(s, test_date, K, history, horizon, use_state)]['full'].append(
                                    full_result_MAE)

                                results_dict[(s, test_date, K, history, horizon, use_state)]['state'].append(
                                    state_result_MAPE)
                                results_dict[(s, test_date, K, history, horizon, use_state)]['full'].append(
                                    full_result_MAPE)

                                results_dict[(s, test_date, K, history, horizon, use_state)]['state'].append(
                                    state_result_RMSE)
                                results_dict[(s, test_date, K, history, horizon, use_state)]['full'].append(
                                    full_result_RMSE)
                            print("{}-{}-{}-use_state:{} 完成计算".format(K, history, horizon, use_state))
            print("测试日期 {} 已经完成计算".format(test_date))
        print("传感器 {} 已经完成计算".format(s))

    with open('{}-results.pkl'.format(model_str), 'wb') as f:
        pickle.dump(results_dict, f)
    with open('{}-trafficflow.pkl'.format(model_str), 'wb') as f:
        pickle.dump(traffic_flow, f)


if __name__ == '__main__':
    main()
