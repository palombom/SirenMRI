import argparse
import numpy as np
import pickle
import sys
from mat73 import loadmat
from numpy.core.records import fromarrays
from scipy.io import savemat

parser = argparse.ArgumentParser(description='Going from MAT trajectories to pickle and viceversa.')
parser.add_argument("-i", "--input", type=str, help="Input MAT/pickle file.", default="")
parser.add_argument("-o", "--output", type=str, help="Output pickle/MAT file.", default="")

args = parser.parse_args()
if not args.input:
    sys.exit("Please provide an input file.")

input = args.input
in_ext = input.split('.')[-1]
if args.output:
    output = args.output
else:
    output = input.split('.')[0]
    if in_ext == 'mat':
        output = output + '.pickle'
    else:
        output = output + '.mat'
print(f"Going from {input} to {output}")

if in_ext == 'mat':
    in_data = loadmat(input)
    init = in_data['traj']['data'][:,0,:]
    series = in_data['traj']['data'][:,1:,:]
    with open(output, 'wb') as handle:
        pickle.dump({'series': series, 'init': init}, handle)
else:
    with open(input, 'rb') as handle:
        trj=pickle.load(handle)
    trj_s_i = np.zeros((trj['series'].shape[0], trj['series'].shape[1] + 1, trj['series'].shape[2]))
    trj_s_i[:,0,:] = trj['init']
    trj_s_i[:,1:,:] = trj['series']
    traj = {'duration': trj['series'].shape[1]/10, 'noOfWalkers': trj['series'].shape[0], 'steps': trj['series'].shape[1], 'data': trj_s_i}
    savemat(output, {'traj': traj})
