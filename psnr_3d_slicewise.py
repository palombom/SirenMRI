import torch
import numpy as np
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
import pickle
import argparse
import util

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--predicted", help="Predicted volume")
parser.add_argument("-g", "--ground_truth", help="Ground truth")
parser.add_argument("-o", "--output", help="Output folder")

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

nii_g = nib.load(args.ground_truth)
img_g = nii_g.get_fdata()
sx, sy, sz, vols = img_g.shape[0], img_g.shape[1], img_g.shape[2], img_g.shape[3]

nii_p = nib.load(args.predicted)
img_p = nii_p.get_fdata()

psnr = []
for i in range(sz):
    img = img_g[: ,: ,i ,:]
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img.astype(np.float32))

    coordinates, features = util.to_coordinates_and_features_2D(img)

    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    scaler.fit(features.numpy())
    features = scaler.transform(features.numpy())
    features = torch.from_numpy(features.astype(np.float32))

    coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)

    img = img_p[: ,: ,i ,:]
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img.astype(np.float32))

    coordinates, predicted = util.to_coordinates_and_features_2D(img)

    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    scaler.fit(predicted.numpy())
    predicted = scaler.transform(predicted.numpy())
    predicted = torch.from_numpy(predicted.astype(np.float32))

    coordinates, predicted = coordinates.to(device, dtype), predicted.to(device, dtype)

    psnr.append(util.get_clamped_psnr(predicted, features))

with open(args.output + '/psnr_slicewise.pickle', 'wb') as handle:
    pickle.dump(psnr, handle)