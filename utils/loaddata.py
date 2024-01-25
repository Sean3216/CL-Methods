import pandas as pd
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from utils.preprocess import disruptsubjects, HAR_preprocessing_WISDM, HAR_preprocessing_USC


def SplitUCIHAR(dir: str = './data', numtask: int = 10):
    #Loading feature names
    with open(dir+'/UCI_HAR/features.txt') as f:
        features = [line.split()[1] for line in f.readlines()]
    #deleting duplicate features
    seen = set()
    uniq_features = []
    for idx, x in enumerate(features):
        if x not in seen:
            uniq_features.append(x)
            seen.add(x)
        elif x + 'n' not in seen:
            uniq_features.append(x + 'n')
            seen.add(x + 'n')
        else:
            uniq_features.append(x + 'nn')
            seen.add(x + 'nn')

    #Loading train data
    train = pd.read_csv(dir+'/UCI_HAR/train/X_train.txt', delim_whitespace=True, header=None, names=uniq_features)
    train['subject'] = pd.read_csv(dir+'/UCI_HAR/train/subject_train.txt', header=None)
    train['activity'] = pd.read_csv(dir+'/UCI_HAR/train/y_train.txt', header=None)

    #Loading test data
    test = pd.read_csv(dir+'/UCI_HAR/test/X_test.txt', delim_whitespace=True, header=None, names=uniq_features)
    test['subject'] = pd.read_csv(dir+'/UCI_HAR/test/subject_test.txt', header=None)
    test['activity'] = pd.read_csv(dir+'/UCI_HAR/test/y_test.txt', header=None)

    #combining data
    combined = pd.concat([train, test], axis = 0, ignore_index=True).reset_index(drop=True)
    combined['activity'] = combined['activity'] - 1
    print(combined.head())
    combined = disruptsubjects(combined, num_groups=10)

    #Check duplicates
    print('No of duplicates in train: {}'.format(sum(train.duplicated())))
    print('No of duplicates in test : {}'.format(sum(test.duplicated())))

    #Check missing values(NaN)
    print('We have {} NaN/Null values in train'.format(train.isnull().values.sum()))
    print('We have {} NaN/Null values in test'.format(test.isnull().values.sum()))

    task = {} #initialized empty task dictionary
    
    #initiating tasks
    for i in range(numtask):
        task[i+1] = {'features':[], 'detailedclass':[]}
    
    #divide by subjects
    subjects = list(combined.subject.unique())
    numsubjects = len(subjects)
    if numtask == numsubjects:
        subjectpertask = 1
    else:
        subjectpertask = int(numsubjects//numtask)

    np.random.seed(42)
    for t in range(numtask):
        selected_subjects = np.random.choice(subjects, size = subjectpertask, replace = False)
        subjects = [x for x in subjects if x not in selected_subjects]
        print(f'Slicing for task {t+1}')
        print(f"select subjects {selected_subjects} for task {t+1}")
        features = combined[combined['subject'].isin(selected_subjects)].copy().reset_index(drop=True)
        labels = features['activity'].copy()
        features.drop(['subject','activity'], axis = 1, inplace = True)  
        #add to task as features and labels but turn them to torch tensor  
        for i in range(len(features)):
            task[t+1]['features'].append(torch.tensor(features.iloc[i,:].values))
            task[t+1]['detailedclass'].append(torch.tensor(labels.loc[i]))
    return task

### Split-WISDM
def SplitWISDM(dir: str = './data',numtask: int = 10):
    columns = ['subject','activity','timestamp','x-axis','y-axis','z-axis']
    data = pd.read_csv(dir+'/WISDM_AR/WISDM_ar_v1.1_raw.txt', header = None, names = columns, index_col = False, on_bad_lines = 'skip')
    data = data.dropna()
    data['z-axis'] = data['z-axis'].str.replace(';', '')
    data['z-axis'] = data['z-axis'].astype(float)

    #drop the rows where timestamp is 0
    data = data[data['timestamp'] != 0]

    #arrange data in ascending order of the user and timestamp
    data = data.sort_values(by = ['subject','timestamp'], ignore_index = True)
    label = {"Walking": 0, "Jogging": 1,
             "Upstairs": 2, "Downstairs": 3,
             "Sitting": 4,"Standing":5}
    data['activity'] = data['activity'].map(label)
    data = data[['x-axis','y-axis','z-axis','subject','activity']]
    data = disruptsubjects(data, num_groups=6)
    print(data.head())

    task = {} #initialized empty task dictionary
    
    #initiating tasks
    for i in range(numtask):
        task[i+1] = {'features':[], 'detailedclass':[]}
    
    #divide by subjects
    subjects = list(data.subject.unique())
    numsubjects = len(subjects)
    if numtask == numsubjects:
        subjectpertask = 1
    else:
        subjectpertask = int(numsubjects//numtask)

    np.random.seed(42)
    for t in range(numtask):
        selected_subjects = np.random.choice(subjects, size = subjectpertask, replace = False)
        subjects = [x for x in subjects if x not in selected_subjects]
        print(f'Slicing for task {t+1}')
        print(f"select subjects {selected_subjects} for task {t+1}")
        preFE = data[data['subject'].isin(selected_subjects)].copy().reset_index(drop=True)
        features, labels = HAR_preprocessing_WISDM(preFE, window_size = 100, step_size = 20)
        #add to task as features and labels but turn them to torch tensor  
        for i in range(len(features)):
            task[t+1]['features'].append(torch.tensor(features.iloc[i,:].values))
            task[t+1]['detailedclass'].append(torch.tensor(labels[i]))
    return task

### Inc-USCHAD
def IncUSCHAD(dir: str = './data',numtask: int = 10):
    #data sampled at 100Hz
    #create empty dictionary
    data_dict = {'acc_x': [], 'acc_y': [], 'acc_z': [],
                 'gyro_x': [], 'gyro_y': [], 'gyro_z': [],
                 'subject': [], 'activity': []}    
    #append per subject
    subjects = list(range(1,15))
    for subject in subjects:
        activities = range(1, 13)
        for activity in activities:
            for trial in range(1, 6):
                print(f"loading subject {subject}, activity {activity}, trial {trial}")
                data = sio.loadmat("%s/USC_HAD/Subject%d/a%dt%d.mat" % (dir, subject, activity, trial))
                data = np.array(data['sensor_readings'])
                for i in range(0, data.shape[0]):
                    #append value to dictionary
                    data_dict['acc_x'].append(data[i, 0])
                    data_dict['acc_y'].append(data[i, 1])
                    data_dict['acc_z'].append(data[i, 2])
                    data_dict['gyro_x'].append(data[i, 3])
                    data_dict['gyro_y'].append(data[i, 4])
                    data_dict['gyro_z'].append(data[i, 5])
                    data_dict['subject'].append(subject)
                    data_dict['activity'].append(activity)
    df = pd.DataFrame(data_dict)
    print(df.head())

    task = {} #initialized empty task dictionary
    
    #initiating tasks
    for i in range(numtask):
        task[i+1] = {'features':[], 'detailedclass':[]}
    
    #task are divided by the number of classes
    classes = list(df.activity.unique())
    numclasses = len(classes)
    if numtask == numclasses:
        classespertask = 1
    else:
        classespertask = int(numclasses//numtask)

    np.random.seed(42)
    classmin = 0
    classmax = 0
    for t in range(numtask):
        selected_classes = np.random.choice(classes, size = classespertask, replace = False)
        classes = [x for x in classes if x not in selected_classes]
        print(f'Slicing for task {t+1}')
        print(f"select labels {selected_classes} for task {t+1}")
        preFE = df[df['activity'].isin(selected_classes)].copy().reset_index(drop=True)
        features, labels = HAR_preprocessing_USC(preFE, window_size = 500, step_size = 100)
        #encoding labels
        classmax += classespertask
        newlabel = list(range(classmin, classmax))
        dictformap = {label: value for label, value in zip(selected_classes, newlabel)}
        print(f"original labels encoded to what value: \n{dictformap}")
        labels = [dictformap[label] for label in labels]
        #add to task as features and labels but turn them to torch tensor  
        for i in range(len(features)):
            task[t+1]['features'].append(torch.tensor(features.iloc[i,:].values))
            task[t+1]['detailedclass'].append(torch.tensor(labels[i]))
        classmin += classespertask
    return task

def loaddata(tasks, tasknum: int = 1, batch_size: int = 32): #returns the resepective data as the dataloader
    #first, make to tensor dataset
    xarray = np.stack(tasks[tasknum]['features'])
    X = torch.tensor(xarray, dtype = torch.float32)
    y = torch.tensor(np.array(tasks[tasknum]['detailedclass']), dtype = torch.long)
    dataset = TensorDataset(X,y)

    #second, split to train and eval for this task
    num_data = len(dataset)
    indices = list(range(num_data))
    np.random.shuffle(indices)
    splitval = int(np.floor(0.2 * num_data)) #0.2 eval size
    train_idx, eval_idx = indices[splitval:], indices[:splitval]

    #third, split to train and test from train data
    num_train = len(train_idx)
    np.random.shuffle(train_idx)
    splittest = int(np.floor(0.2* num_train)) #0.2 test size
    train_idx, test_idx = train_idx[splittest:], train_idx[:splittest]

    #### Check datacount
    #trainsample = []
    #for i in range(len(dataset)):
    #    if i in train_idx:
    #        trainsample.append(dataset[i])
    #pickedlabels = np.unique(y)
    #num1 = 0
    #num2 = 0
    #for i in range(len(trainsample)):
    #    if trainsample[i][1] == pickedlabels[0]:
    #        num1 += 1
    #    if trainsample[i][1] == pickedlabels[1]:
    #        num2 += 1
    #print(f"Class {pickedlabels[0]} count: {num1}")
    #print(f"Class {pickedlabels[1]} count: {num2}")

    #Define sampler
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    eval_sampler = SubsetRandomSampler(eval_idx)

    #load as dataloader
    train_loader = DataLoader(dataset,
                              batch_size = batch_size,
                              sampler = train_sampler)
    test_loader = DataLoader(dataset,
                             batch_size = batch_size,
                             sampler = test_sampler)
    valid_loader = DataLoader(dataset,
                              batch_size = batch_size,
                              sampler = eval_sampler)
    
    return train_loader, test_loader, valid_loader
