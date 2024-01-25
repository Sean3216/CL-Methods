import torch 
import torch.nn as nn

import numpy as np
import copy
import sys
import random

from utils.metrics import ForgettingMeasure, IntransienceMeasure, AverageAccuracy
from utils.utilfunc import get_unique_labels
from utils.generalinference import train, test

from ER.train import ERtrain
from DER import *

class CL_Class():
    '''
    All continual learning methods and base comparison are implemented with the assumption that we know how many number of classes might exist in the future.
    Thus, number of neurons in the final layer of the model is fixed.
    '''
    def __init__(self,
                 alltaskdata,
                 taskslist,
                 model: nn.Module,
                 epochs: int = 5,
                 random_state: int = 42):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Use device: {} as device".format(self.device))
        self.modelcl = copy.deepcopy(model)
        self.originalmodel = copy.deepcopy(model)
        self.data = alltaskdata
        self.taskslist = taskslist

        self.epochs = epochs
        self.random_state = random_state

        #Continual Learning Metrics
        self.FM = ForgettingMeasure()
        self.IM = IntransienceMeasure(self.originalmodel, self.data)
        self.AA = AverageAccuracy()

        self.allFM = []
        self.allIM = []
        self.allAA = []

    def finetuning(self): #Vanilla neural network finetuned(trained continuously over tasks)
        for task in self.taskslist: 
            print('=====================================================================================')
            print('Current task: ', task)
            print('=====================================================================================')
            #training
            trainresult = train(self.modelcl, 0.01, torch.optim.SGD, self.epochs, self.device, self.data[task]['train'], self.data[task]['test'])
            loss, currentacc = test(self.modelcl, self.data[task]['val'], self.device)
            print(f'Training result for task {task}:\nLoss: {loss}\nAccuracy: {round(currentacc, 4)}')
            #Memory Stability#
            self.FM.addacc(currentacc, task, task)
            #Average Accuracy#
            self.AA.accuracies.append(currentacc)
            if task > self.taskslist[0]:
                for previoustask in range(self.taskslist[0], task):
                    print(f"Testing model against data for task {previoustask} after training on task {task}")
                    loss, acc = test(self.modelcl, self.data[previoustask]['val'], device = self.device)
                    print(f"Loss: {loss}\nAccuracy: {round(acc, 4)}")
                    #Memory Stability#
                    self.FM.addacc(acc, test_on = previoustask, train_on = task)
                    #Average Accuracy#
                    self.AA.accuracies.append(acc)
            #Memory Stability#
            self.allFM.append(self.FM.calculateforgetting())
            #Learning Plasticity#
            self.allIM.append(self.IM.calculateintransience(currentacc, task, self.epochs, 0.01, optimizer=torch.optim.SGD))
            #Average Accuracy & Average Incremental Accuracy of current task
            self.allAA.append(self.AA.calculate(task)) #calculate current AA first
            print(f'Average Incremental Accuracy at task {task} is {np.mean(self.allAA)}')
        print(f'=====================SUMMARY=====================')
        print(f'All Forgetting Measure: {self.allFM}')
        print(f'All Intransience Measure: {self.allIM}')
        print(f'All Average Accuracy: {self.allAA}')
        print(f'Average Forgetting Measure: {round(np.mean(self.allFM[1:]),4)}')
        print(f'Average Intransience Measure: {round(np.mean(self.allIM),4)}')
        print(f'Average Incremental Accuracy: {round(np.mean(self.allAA),4)}')
        print(f'Standard Deviation of Forgetting Measure every task: {round(np.std(self.allFM[1:], ddof = 1),4)}')
        print(f'Standard Deviation of Intransience Measure every task: {round(np.std(self.allIM, ddof = 1),4)}')
        print(f'Standard Deviation of Average Accuracy: {round(np.std(self.allAA, ddof = 1),4)}')

    def replay(self, sampling_type: str = "reservoir", n_memory: int = 500): #train data and test data are in dataloader
        currentmemory = {'data': [], 'labels': []}
        observedclass = []
        random.seed(self.random_state)
        for task in self.taskslist: #in this implementation, tasks are in number (integer)
            print('=====================================================================================')
            print('Current task: ', task)
            print('=====================================================================================')
            if task == self.taskslist[0]:
                observedclass += get_unique_labels(self.data[task]['train'])
            if any(element not in observedclass for element in get_unique_labels(self.data[task]['train'])):
                difference_list = [element for element in get_unique_labels(self.data[task]['train']) if element not in observedclass]
                #adding new neurons to the classification layer
                #self.modelcl.add_new_neurons(len(difference_list))
                #self.IM.new_classes_to_vanilla(len(difference_list))
                #adding new classes to the observed class list
                observedclass += difference_list       
            #training
            currentmemory, trainresult = ERtrain(self.modelcl, 0.01, torch.optim.SGD,
                                                 self.epochs, self.device,
                                                 self.data[task]['train'], self.data[task]['test'],
                                                 currentmemory, n_memory, sampling= sampling_type)
            
            loss, currentacc = test(self.modelcl, self.data[task]['val'], self.device)
            print(f'Training result for task {task}:\nLoss: {loss}\nAccuracy: {round(currentacc, 4)}')
            #Memory Stability#
            self.FM.addacc(currentacc, task, task)
            #Average Accuracy#
            self.AA.accuracies.append(currentacc)
            if task > self.taskslist[0]:
                for previoustask in range(self.taskslist[0], task):
                    print(f"Testing model against data for task {previoustask} after training on task {task}")
                    loss, acc = test(self.modelcl, self.data[previoustask]['val'], device = self.device)
                    print(f"Loss: {loss}\nAccuracy: {round(acc, 4)}")
                    #Memory Stability#
                    self.FM.addacc(acc, test_on = previoustask, train_on = task)
                    #Average Accuracy#
                    self.AA.accuracies.append(acc)
            #Memory Stability#
            self.allFM.append(self.FM.calculateforgetting())
            #Learning Plasticity#
            self.allIM.append(self.IM.calculateintransience(currentacc, task, self.epochs, 0.01, optimizer=torch.optim.SGD))
            #Average Accuracy & Average Incremental Accuracy of current task
            self.allAA.append(self.AA.calculate(task)) #calculate current AA first
            print(f'Average Incremental Accuracy at task {task} is {np.mean(self.allAA)}')
        print(f'=====================SUMMARY=====================')
        print(f'All Forgetting Measure: {self.allFM}')
        print(f'All Intransience Measure: {self.allIM}')
        print(f'All Average Accuracy: {self.allAA}')
        print(f'Average Forgetting Measure: {round(np.mean(self.allFM[1:]),4)}')
        print(f'Average Intransience Measure: {round(np.mean(self.allIM),4)}')
        print(f'Average Incremental Accuracy: {round(np.mean(self.allAA),4)}')
        print(f'Standard Deviation of Forgetting Measure every task: {round(np.std(self.allFM[1:], ddof = 1),4)}')
        print(f'Standard Deviation of Intransience Measure every task: {round(np.std(self.allIM, ddof = 1),4)}')
        print(f'Standard Deviation of Average Accuracy: {round(np.std(self.allAA, ddof = 1),4)}')
    