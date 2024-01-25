from utils.loaddata import SplitUCIHAR, SplitWISDM, IncUSCHAD, loaddata #, SplitUCIHAR_raw
from utils.models import NetUCIHAR, NetWISDM, NetUSCHAD, NetUCIHAR_TMC, NetWISDM_TMC, NetUSCHAD_TMC
from CL import ContinualLearning
import torch
import argparse


parser = argparse.ArgumentParser(description = 'Choosing which experiment scenario to do')
group = parser.add_mutually_exclusive_group()
group.add_argument('--replay_reservoir', help = 'Experience Replay with Reservoir Sampling', action= argparse.BooleanOptionalAction)
group.add_argument('--replay_ring', help = 'Experience Replay with Ring Buffer', action= argparse.BooleanOptionalAction)
group.add_argument('--der', help = 'DER', action= argparse.BooleanOptionalAction)
group.add_argument('--derpp', help = 'DER++', action= argparse.BooleanOptionalAction)

group.add_argument('--finetune', help = 'Vanilla neural network finetuned', action= argparse.BooleanOptionalAction)
group.add_argument('--TMC', help = 'Tangent Model Composition', action= argparse.BooleanOptionalAction)

parser.add_argument('--dataset', help = 'Choose dataset', default = 'UCIHAR', choices = ['UCIHAR', 'WISDM', 'USCHAD'])

opt = parser.parse_args()

def main():
    if opt.dataset == 'UCIHAR':
        alltask = SplitUCIHAR(numtask=30)
        if opt.TMC:
            model = NetUCIHAR_TMC()
        else:
            model = NetUCIHAR()
        batch_sz = 10
        num_mem = 12
    elif opt.dataset == 'WISDM':
        alltask = SplitWISDM(numtask=18)
        if opt.TMC:
            model = NetWISDM_TMC()
        else:
            model = NetWISDM()
        batch_sz = 10
        num_mem = 12
    elif opt.dataset == 'USCHAD':
        alltask = IncUSCHAD(numtask=6)
        if opt.TMC:
            model = NetUSCHAD_TMC()
        else:
            model = NetUSCHAD()
        batch_sz = 32
        num_mem = 500
    
    if opt.der or opt.der_greedy or opt.der_uniform or opt.der_c_soup or opt.der_c_soup_cosine:
        model.der = True
        alpha = 0.3
    if opt.derpp or opt.derpp_greedy or opt.derpp_uniform or opt.derpp_c_soup or opt.derpp_c_soup_cosine:
        model.der = True
        alpha = 0.2
        beta = 0.5

    #load data of all task in a single dictionary
    alltaskdataloader = {}
    for task in alltask.keys():
        alltaskdataloader[task] = {}
        alltaskdataloader[task]['train'], alltaskdataloader[task]['test'], alltaskdataloader[task]['val'] = loaddata(alltask, tasknum = task, batch_size= batch_sz)
    tasknumbers = list(alltask.keys())
    del alltask

    cl = ContinualLearning()

if __name__ == "__main__":
    main()
