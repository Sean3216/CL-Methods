import torch

import copy

from utils.preprocess import combine_dataloader
from utils.generalinference import test, train


class ForgettingMeasure():
    # Forgetting is defined as the difference between the accuracy of the current task and the highest accuracy of the previous tasks
    # The Forgetting Measure at k-th task is the average forgetting of all old tasks
    def __init__(self):
        self.taskaccuracy = {} 
        self.forgetting = []
        self.tasksobserved = list(self.taskaccuracy.keys()) #list of tasks observed, including the current task
    
    def addacc(self, acc, test_on, train_on):
        if train_on not in list(self.taskaccuracy.keys()):
            self.tasksobserved.append(train_on)
            self.taskaccuracy[train_on] = {}
        self.taskaccuracy[train_on][test_on] = acc

    def calculateforgetting(self):
        print('=========Calculating Forgetting Measure!=========')
        if (len(self.tasksobserved)-1) == 0:
            print(f'Forgetting measure at task {self.tasksobserved[-1]} is {0}')
            return 0
        else:
            for train_on in self.tasksobserved[:-1]: #on every task recorded except the current task,
                #####################Calculate forgeting for each task on current task#####################
                accdiffs = []
                for test_on in self.taskaccuracy[train_on].keys(): #compare every test result on that specific task (train_on) with the current task
                    accuracy_diff = self.taskaccuracy[train_on][test_on] - self.taskaccuracy[self.tasksobserved[-1]][test_on]
                    print(f'Forgetting measure between model when trained on task {train_on} and tested for task {test_on} compared to model when train on task {self.tasksobserved[-1]} and tested for task {test_on} is {accuracy_diff}')
                    accdiffs.append(accuracy_diff)
                #show the maximum difference between the current task and that specific task (train_on)
                print(f'Forgetting measure on task {train_on} after train on task {self.tasksobserved[-1]} is {max(accdiffs)}')
                self.forgetting.append(max(accdiffs))
                ###########################################################################################
            AvgFM = sum(self.forgetting)/(len(self.tasksobserved)-1)
            print(f'Average Forgetting measure at task {self.tasksobserved[-1]} is {AvgFM}')
            self.forgetting = []
            return round(AvgFM, 4)

############################# Learning Plasticity #############################
class IntransienceMeasure():
    # Intranscience Measure is calculated by the difference between the classification accuracy of a randomly-initalized reference model jointly trained with all tasks and the classification accuracy of the current model with continual learning    def __init__(self):
    def __init__(self, vanillamod, alldata):
        self.vanilla = vanillamod
        self.alldata = alldata
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def calculateintransience(self, clmodacc, currenttask, epochs : int = 5, lr : float = 0.01, optimizer = torch.optim.SGD):
        print('=========Calculating Intransience Measure!=========')
        vanillamodel = copy.deepcopy(self.vanilla)
        taskobserved = list(self.alldata.keys())[:currenttask]
        print(f"Tasks considered: {taskobserved}")
        #stack all data until current task
        trainloader, testloader, valloader = combine_dataloader(self.alldata, taskobserved)
        #train vanillamodel with alltasks
        _ = train(vanillamodel, 0.01, optimizer, epochs, self.device, trainloader, testloader)
        allloss, allacc = test(vanillamodel, valloader, self.device)

        #calculate Intranscience Measure
        print(f'Classification accuracy with vanilla model on all tasks is {allacc}')
        print(f'Classification accuracy with CL model is {clmodacc}')
        intransiencemeasure = allacc - clmodacc
        print(f'Intransience Measure at task {taskobserved[-1]} is {intransiencemeasure}')
        return round(intransiencemeasure, 4)
    
    def new_classes_to_vanilla(self,numnewclasses):
        self.vanilla.add_new_neurons(numnewclasses)

############################# Average Accuracy #############################
class AverageAccuracy():
    def __init__(self):
        self.accuracies = []
    
    def calculate(self, task):
        if task == 0:
            avgacc = self.accuracies[0]
            self.accuracies = []
        else:
            avgacc = sum(self.accuracies)/len(self.accuracies)
            self.accuracies = []
        return avgacc