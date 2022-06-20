import scipy.io
import scipy
import argparse
import getpass
import imageio
import json
import os
import random
import sys
import torch
import util
import numpy as np
import nibabel as nib
from siren import ModulatedSiren
from torchvision import transforms
from torchvision.utils import save_image
from training import Trainer
from sklearn.preprocessing import MinMaxScaler
import pickle
import py7zr


parser = argparse.ArgumentParser()
parser.add_argument("-ld", "--logdir", help="Path to save output", default=f"/tmp/{getpass.getuser()}")
parser.add_argument("-lm", "--log_measures", help="Save measures for each epoch", action='store_true')
parser.add_argument("-ni", "--num_iters", help="Number of iterations to train for", type=int, default=50000)
parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=2e-4)
parser.add_argument("-se", "--seed", help="Random seed", type=int, default=random.randint(1, int(1e6)))
parser.add_argument("-img", "--image", help="Image to train on", default=f"/tmp/{getpass.getuser()}")
parser.add_argument("-lss", "--layer_size", help="Layer sizes as list of ints", type=int, default=28)
parser.add_argument("-nl", "--num_layers", help="Number of layers", type=int, default=10)
parser.add_argument("-w0", "--w0", help="w0 parameter for SIREN model.", type=float, default=30.0)
parser.add_argument("-w0i", "--w0_initial", help="w0 parameter for first layer of SIREN model.", type=float, default=30.0)
parser.add_argument("-b", "--batch_size", help="Size of mini-batches (-1 for one single batch per epoch).", type=int, default=-1)
parser.add_argument("-fa", "--final_activation", help="Final activation function.", default="identity")

args = parser.parse_args()
mlp_activation = {'relu': torch.nn.ReLU(), 'tanh': torch.nn.Tanh()}
final_activation = {'identity': torch.nn.Identity(), 'relu': torch.nn.ReLU(), 'tanh': torch.nn.Tanh()}

# Set up torch and cuda
deviceinuse = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

# Set random seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Dictionary to register mean values (both full precision and half precision)
results = {'fp_bpp': [], 'hp_bpp': [], 'fp_psnr': [], 'hp_psnr': []}

# Create directory to store the training info
if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)
if args.log_measures and not os.path.exists(args.logdir + '/logs'):
    os.makedirs(args.logdir + '/logs')
# Create directory to store the training info
compression_folder = args.logdir + "/SirenCompression"
if not os.path.exists(compression_folder):
    os.makedirs(compression_folder)

img_to_compress = args.image

# Fit images
print("")
print(f'Compressing Image {img_to_compress}')

# Load image
nii = nib.load(img_to_compress)
img_tmp = nii.get_fdata()
sx, sy, sz, vols = img_tmp.shape[0], img_tmp.shape[1], img_tmp.shape[2], img_tmp.shape[3]

# Save the input for the decoding
indata = {'input': np.array([sx, sy, sz, vols])}
scipy.io.savemat(compression_folder + f'/input_to_best_model.mat', indata)

# Prepare to save the decompressed image
new_header = nii.header.copy()
img_decompressed = img_tmp

print(f'Image size: {sx} x {sy} x {sz} x {vols}')

func_rep = ModulatedSiren(
    dim_in=2,
    dim_hidden=args.layer_size,
    dim_latent=1,
    dim_out=vols,
    num_layers=args.num_layers
)
if torch.cuda.device_count() > 1:
    func_rep = torch.nn.DataParallel(func_rep)
func_rep.to(device)

# Set up training
trainer = Trainer(func_rep, lr=args.learning_rate)

img = np.transpose(img_tmp, (3, 0, 1, 2))
img = torch.from_numpy(img.astype(np.float32))

coordinates, latent, features = util.to_coordinates_latent_and_features(img)

scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
scaler.fit(features.numpy())
features = scaler.transform(features.numpy())
features = torch.from_numpy(features.astype(np.float32))

coordinates, latent, features = coordinates.to(device, dtype), latent.to(device, dtype), features.to(device, dtype)

# Train model in full precision
trainer.train(coordinates, features, num_iters=args.num_iters, batch_size=args.batch_size, latent=latent)
print(f'Best training psnr: {trainer.best_vals["psnr"]:.2f}')
print(f'Best training loss: {trainer.best_vals["loss"]:.2f}')
if args.log_measures:
    with open(args.logdir + '/log.pickle', 'wb') as handle:
        pickle.dump(trainer.logs, handle)

# Save best model
torch.save(trainer.best_model, compression_folder + f'/best_model.pt')

# Update current model to be best model
func_rep.load_state_dict(trainer.best_model)

# Save full precision image reconstruction
with torch.no_grad():
    PredParams = func_rep(coordinates, latent)
PredParams = scaler.inverse_transform(PredParams.cpu().numpy())
img_decompressed = np.reshape(PredParams,
                              (img_tmp.shape[0], img_tmp.shape[1], img_tmp.shape[2], img_tmp.shape[3]))


# Save the prediction in NIFTI
print("Saving the decompressed NIFTI")
new_img = nib.nifti1.Nifti1Image(img_decompressed, None, header=new_header)
nib.save(new_img, args.logdir + f'/dwi_decompressed.nii.gz')

# Compress the output
with py7zr.SevenZipFile(compression_folder + '.7z', 'w') as archive:
    archive.writeall(compression_folder)