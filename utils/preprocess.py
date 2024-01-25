import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

import torch
from torch.utils.data import TensorDataset, DataLoader


def disruptsubjects(data, num_groups=10, mean_shift_inc = 1.5, std_dev_shift_inc = 0.1): #if the data is HAR, we use this. Introduce Gaussian Noise
    subjects = list(data.subject.unique())

    # Calculate the approximate size of each group
    group_size = len(subjects) // num_groups

    # Group the subjects into separate lists
    grouped_lists = [subjects[i:i + group_size] for i in range(0, len(subjects), group_size)]

    # If there are remaining subjects, distribute them among the groups
    remaining_subjects = subjects[group_size * num_groups:]
    for i, subject in enumerate(remaining_subjects):
        grouped_lists[i].append(subject)
    print(grouped_lists)

    # shift the data of each groups
    mean_shift = 0.0
    std_dev_shift = 0.0
    np.random.seed(42)
    print(f'including column {data.iloc[:, :-2].columns}')
    for group in grouped_lists:   
        print(f'shifting distribution of group {group}')
        for subject in group:
            shift_on = data[data['subject'] == subject].index
            noise = np.random.normal(mean_shift, std_dev_shift, size = (len(shift_on), len(data.iloc[:, :-2].columns)))
            data.iloc[shift_on, :-2] += noise
        mean_shift += mean_shift_inc
        std_dev_shift += std_dev_shift_inc
    return data

def HAR_preprocessing_WISDM(data, window_size, step_size):
    x_list = data['x-axis'].values
    y_list = data['y-axis'].values
    z_list = data['z-axis'].values
    train_labels = data['activity'].values

    x = []
    y = []

    for i in range(0, len(data) - window_size + 1, step_size):
        x_window = x_list[i:i + window_size]
        y_window = y_list[i:i + window_size]
        z_window = z_list[i:i + window_size]

        x.append([
            np.mean(x_window), np.mean(y_window), np.mean(z_window),
            np.std(x_window), np.std(y_window), np.std(z_window),
            np.mean(np.abs(x_window - np.mean(x_window))),
            np.mean(np.abs(y_window - np.mean(y_window))),
            np.mean(np.abs(z_window - np.mean(z_window))),
            np.min(x_window), np.min(y_window), np.min(z_window),
            np.max(x_window), np.max(y_window), np.max(z_window),
            np.max(x_window) - np.min(x_window),
            np.max(y_window) - np.min(y_window),
            np.max(z_window) - np.min(z_window),
            np.median(x_window), np.median(y_window), np.median(z_window),
            np.median(np.abs(x_window - np.median(x_window))),
            np.median(np.abs(y_window - np.median(y_window))),
            np.median(np.abs(z_window - np.median(z_window))),
            np.percentile(x_window, 75) - np.percentile(x_window, 25),
            np.percentile(y_window, 75) - np.percentile(y_window, 25),
            np.percentile(z_window, 75) - np.percentile(z_window, 25),
            np.sum(x_window < 0), np.sum(y_window < 0), np.sum(z_window < 0),
            np.sum(x_window > 0), np.sum(y_window > 0), np.sum(z_window > 0),
            np.sum(x_window > np.mean(x_window)),
            np.sum(y_window > np.mean(y_window)),
            np.sum(z_window > np.mean(z_window)),
            len(find_peaks(x_window)[0]),
            len(find_peaks(y_window)[0]),
            len(find_peaks(z_window)[0]),
            stats.skew(x_window), stats.skew(y_window), stats.skew(z_window),
            stats.kurtosis(x_window), stats.kurtosis(y_window), stats.kurtosis(z_window),
            np.sum(x_window**2) / window_size,
            np.sum(y_window**2) / window_size,
            np.sum(z_window**2) / window_size,
            np.mean((x_window**2 + y_window**2 + z_window**2)**0.5),
            np.sum(np.abs(x_window)) / window_size +
            np.sum(np.abs(y_window)) / window_size +
            np.sum(np.abs(z_window)) / window_size
        ])

        y.append(train_labels[i:i + window_size][-1])

    x = pd.DataFrame(x, columns=[
        'x_mean', 'y_mean', 'z_mean',
        'x_std', 'y_std', 'z_std',
        'x_aad', 'y_aad', 'z_aad',
        'x_min', 'y_min', 'z_min',
        'x_max', 'y_max', 'z_max',
        'x_maxmin_diff', 'y_maxmin_diff', 'z_maxmin_diff',
        'x_median', 'y_median', 'z_median',
        'x_mad', 'y_mad', 'z_mad',
        'x_IQR', 'y_IQR', 'z_IQR',
        'x_neg_count', 'y_neg_count', 'z_neg_count',
        'x_pos_count', 'y_pos_count', 'z_pos_count',
        'x_above_mean', 'y_above_mean', 'z_above_mean',
        'x_peak_count', 'y_peak_count', 'z_peak_count',
        'x_skewness', 'y_skewness', 'z_skewness',
        'x_kurtosis', 'y_kurtosis', 'z_kurtosis',
        'x_energy', 'y_energy', 'z_energy',
        'avg_result_accl', 'sma'
    ])

    return x, y


def HAR_preprocessing_USC(data, window_size, step_size):
    accx = data['acc_x'].values
    accy = data['acc_y'].values
    accz = data['acc_z'].values
    gyrox = data['gyro_x'].values
    gyroy = data['gyro_y'].values
    gyroz = data['gyro_z'].values
    train_labels = data['activity'].values

    x = []
    y = []

    for i in range(0, len(data) - window_size + 1, step_size):
        accx_window = accx[i:i + window_size]
        accy_window = accy[i:i + window_size]
        accz_window = accz[i:i + window_size]
        gyrox_window = gyrox[i:i + window_size]
        gyroy_window = gyroy[i:i + window_size]
        gyroz_window = gyroz[i:i + window_size]

        x.append([
            np.mean(accx_window), np.mean(accy_window), np.mean(accz_window),
            np.mean(gyrox_window), np.mean(gyroy_window), np.mean(gyroz_window),
            np.std(accx_window), np.std(accy_window), np.std(accz_window),
            np.std(gyrox_window), np.std(gyroy_window), np.std(gyroz_window),
            np.mean(np.abs(accx_window - np.mean(accx_window))),
            np.mean(np.abs(accy_window - np.mean(accy_window))),
            np.mean(np.abs(accz_window - np.mean(accz_window))),
            np.mean(np.abs(gyrox_window - np.mean(gyrox_window))),
            np.mean(np.abs(gyroy_window - np.mean(gyroy_window))),
            np.mean(np.abs(gyroz_window - np.mean(gyroz_window))),
            np.min(accx_window), np.min(accy_window), np.min(accz_window),
            np.min(gyrox_window), np.min(gyroy_window), np.min(gyroz_window),
            np.max(accx_window), np.max(accy_window), np.max(accz_window),
            np.max(gyrox_window), np.max(gyroy_window), np.max(gyroz_window),
            np.max(accx_window) - np.min(accx_window),
            np.max(accy_window) - np.min(accy_window),
            np.max(accz_window) - np.min(accz_window),
            np.max(gyrox_window) - np.min(gyrox_window),
            np.max(gyroy_window) - np.min(gyroy_window),
            np.max(gyroz_window) - np.min(gyroz_window),
            np.median(accx_window), np.median(accy_window), np.median(accz_window),
            np.median(gyrox_window), np.median(gyroy_window), np.median(gyroz_window),
            np.median(np.abs(accx_window - np.median(accx_window))),
            np.median(np.abs(accy_window - np.median(accy_window))),
            np.median(np.abs(accz_window - np.median(accz_window))),
            np.median(np.abs(gyrox_window - np.median(gyrox_window))),
            np.median(np.abs(gyroy_window - np.median(gyroy_window))),
            np.median(np.abs(gyroz_window - np.median(gyroz_window))),
            np.percentile(accx_window, 75) - np.percentile(accx_window, 25),
            np.percentile(accy_window, 75) - np.percentile(accy_window, 25),
            np.percentile(accz_window, 75) - np.percentile(accz_window, 25),
            np.percentile(gyrox_window, 75) - np.percentile(gyrox_window, 25),
            np.percentile(gyroy_window, 75) - np.percentile(gyroy_window, 25),
            np.percentile(gyroz_window, 75) - np.percentile(gyroz_window, 25),
            np.sum(accx_window < 0), np.sum(accy_window < 0), np.sum(accz_window < 0),
            np.sum(gyrox_window < 0), np.sum(gyroy_window < 0), np.sum(gyroz_window < 0),
            np.sum(accx_window > 0), np.sum(accy_window > 0), np.sum(accz_window > 0),
            np.sum(gyrox_window > 0), np.sum(gyroy_window > 0), np.sum(gyroz_window > 0),
            np.sum(accx_window > np.mean(accx_window)),
            np.sum(accy_window > np.mean(accy_window)),
            np.sum(accz_window > np.mean(accz_window)),
            np.sum(gyrox_window > np.mean(gyrox_window)),
            np.sum(gyroy_window > np.mean(gyroy_window)),
            np.sum(gyroz_window > np.mean(gyroz_window)),
            len(find_peaks(accx_window)[0]),
            len(find_peaks(accy_window)[0]),
            len(find_peaks(accz_window)[0]),
            len(find_peaks(gyrox_window)[0]),
            len(find_peaks(gyroy_window)[0]),
            len(find_peaks(gyroz_window)[0]),
            stats.skew(accx_window), stats.skew(accy_window), stats.skew(accz_window),
            stats.skew(gyrox_window), stats.skew(gyroy_window), stats.skew(gyroz_window),
            stats.kurtosis(accx_window), stats.kurtosis(accy_window), stats.kurtosis(accz_window),
            stats.kurtosis(gyrox_window), stats.kurtosis(gyroy_window), stats.kurtosis(gyroz_window),
            np.sum(accx_window**2) / window_size,
            np.sum(accy_window**2) / window_size,
            np.sum(accz_window**2) / window_size,
            np.sum(gyrox_window**2) / window_size,
            np.sum(gyroy_window**2) / window_size,
            np.sum(gyroz_window**2) / window_size,
            np.mean((accx_window**2 + accy_window**2 + accz_window**2)**0.5),
            np.mean((gyrox_window**2 + gyroy_window**2 + gyroz_window**2)**0.5),
            np.sum(np.abs(accx_window)) / window_size +
            np.sum(np.abs(accy_window)) / window_size +
            np.sum(np.abs(accz_window)) / window_size,
            np.sum(np.abs(gyrox_window)) / window_size +
            np.sum(np.abs(gyroy_window)) / window_size +
            np.sum(np.abs(gyroz_window)) / window_size
        ])

        y.append(train_labels[i:i + window_size][-1])

    x = pd.DataFrame(x, columns=[
        'accx_mean', 'accy_mean', 'accz_mean',
        'gyrox_mean', 'gyroy_mean', 'gyroz_mean',
        'accx_std', 'accy_std', 'accz_std',
        'gyrox_std', 'gyroy_std', 'gyroz_std',
        'accx_aad', 'accy_aad', 'accz_aad',
        'gyrox_aad', 'gyroy_aad', 'gyroz_aad',
        'accx_min', 'accy_min', 'accz_min',
        'gyrox_min', 'gyroy_min', 'gyroz_min',
        'accx_max', 'accy_max', 'accz_max',
        'gyrox_max', 'gyroy_max', 'gyroz_max',
        'accx_maxmin_diff', 'accy_maxmin_diff', 'accz_maxmin_diff',
        'gyrox_maxmin_diff', 'gyroy_maxmin_diff', 'gyroz_maxmin_diff',
        'accx_median', 'accy_median', 'accz_median',
        'gyrox_median', 'gyroy_median', 'gyroz_median',
        'accx_mad', 'accy_mad', 'accz_mad',
        'gyrox_mad', 'gyroy_mad', 'gyroz_mad',
        'accx_IQR', 'accy_IQR', 'accz_IQR',
        'gyrox_IQR', 'gyroy_IQR', 'gyroz_IQR',
        'accx_neg_count', 'accy_neg_count', 'accz_neg_count',
        'gyrox_neg_count', 'gyroy_neg_count', 'gyroz_neg_count',
        'accx_pos_count', 'accy_pos_count', 'accz_pos_count',
        'gyrox_pos_count', 'gyroy_pos_count', 'gyroz_pos_count',
        'accx_above_mean', 'accy_above_mean', 'accz_above_mean',
        'gyrox_above_mean', 'gyroy_above_mean', 'gyroz_above_mean',
        'accx_peak_count', 'accy_peak_count', 'accz_peak_count',
        'gyrox_peak_count', 'gyroy_peak_count', 'gyroz_peak_count',
        'accx_skewness', 'accy_skewness', 'accz_skewness',
        'gyrox_skewness', 'gyroy_skewness', 'gyroz_skewness',
        'accx_kurtosis', 'accy_kurtosis', 'accz_kurtosis',
        'gyrox_kurtosis', 'gyroy_kurtosis', 'gyroz_kurtosis',
        'accx_energy', 'accy_energy', 'accz_energy',
        'gyrox_energy', 'gyroy_energy', 'gyroz_energy',
        'avg_result_accl', 'avg_result_gyro',
        'accsma', 'gyrosma'
    ])

    return x, y


def combine_dataloader(allloaders, taskobserved): #this function is specific for this project
    #featureslist
    featurestrain = []
    featurestest = []
    featuresval = []

    #labelslist
    labelstrain = []
    labelstest = []
    labelsval = []
    
    batch_size = allloaders[1]['train'].batch_size #batch size follows previous dataloader batchsize
    for task in taskobserved:
        print(f'appending task {task}!')
        #train set
        for batch, (x, y) in enumerate(allloaders[task]['train']):
            if x.size(0) != batch_size:
                continue
            for idx in range(len(x)):
                featurestrain.append(x[idx])
                labelstrain.append(y[idx])
                
        #test set
        for batch, (x, y) in enumerate(allloaders[task]['test']):
            if x.size(0) != batch_size:
                continue
            for idx in range(len(x)):
                featurestest.append(x[idx])
                labelstest.append(y[idx])
        
        #val set
        for batch, (x, y) in enumerate(allloaders[task]['val']):
            if x.size(0) != batch_size:
                continue
            for idx in range(len(x)):
                featuresval.append(x[idx])
                labelsval.append(y[idx])
    ########################################Train Dataloader##############################################
    featurestraintensor = torch.stack(featurestrain)
    labelstraintensor = torch.stack(labelstrain)

    traindataset = TensorDataset(featurestraintensor, labelstraintensor)
    # Create a new DataLoader with the specified batch size
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    ########################################Test Dataloader##############################################
    featurestesttensor = torch.stack(featurestest)
    labelstesttensor = torch.stack(labelstest)

    testdataset = TensorDataset(featurestesttensor, labelstesttensor)
    # Create a new DataLoader with the specified batch size
    testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True)
    #########################################Val Dataloader##############################################
    featuresvaltensor = torch.stack(featuresval)
    labelsvaltensor = torch.stack(labelsval)

    valdataset = TensorDataset(featuresvaltensor, labelsvaltensor)
    # Create a new DataLoader with the specified batch size
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True)
    return trainloader, testloader, valloader
