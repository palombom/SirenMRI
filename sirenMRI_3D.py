import scipy
import argparse
import getpass
import os
import random
import torch
import util
import numpy as np
import nibabel as nib
from siren import Siren
from training import Trainer
from sklearn.preprocessing import MinMaxScaler
import pickle


parser = argparse.ArgumentParser()
parser.add_argument("-ld", "--logdir", help="Path to save logs", default=f"/tmp/{getpass.getuser()}")
parser.add_argument("-lm", "--log_measures", help="Save measures for each epoch", action='store_true')
parser.add_argument("-ni", "--num_iters", help="Number of iterations to train for", type=int, default=50000)
parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=2e-4)
parser.add_argument("-se", "--seed", help="Random seed", type=int, default=random.randint(1, int(1e6)))
parser.add_argument("-fd", "--full_dataset", help="Whether to use full dataset", action='store_true')
parser.add_argument("-img", "--image", help="Image to train on", default=f"/tmp/{getpass.getuser()}")
parser.add_argument("-lss", "--layer_size", help="Layer sizes as list of ints", type=int, default=28)
parser.add_argument("-nl", "--num_layers", help="Number of layers", type=int, default=10)
parser.add_argument("-w0", "--w0", help="w0 parameter for SIREN model.", type=float, default=30.0)
parser.add_argument("-w0i", "--w0_initial", help="w0 parameter for first layer of SIREN model.", type=float, default=30.0)

args = parser.parse_args()

# Set up torch and cuda
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

# Set random seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Dictionary to register mean values (both full precision and half precision)
results = {'fp_bpp': [], 'hp_bpp': [], 'fp_psnr': [], 'hp_psnr': []}

# Create directory to store experiments
if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)

img_to_compress = args.image

# Fit images

print(f'Image {img_to_compress}')

# Load image

img_tmp = nib.load(img_to_compress)
new_header = img_tmp.header.copy()
img_tmp = img_tmp.get_fdata()
img = img_tmp[:,:,:,:]
img = np.transpose(img, (3, 0, 1, 2))
img = torch.from_numpy(img.astype(np.float32))
    
# Setup model
func_rep = Siren(
    dim_in=3,
    dim_hidden=args.layer_size,
    dim_out=69,
    num_layers=args.num_layers,
    final_activation=torch.nn.Identity(),
    w0_initial=args.w0_initial,
    w0=args.w0
)
# If more than one GPU is available, use them all
if torch.cuda.device_count() > 1:
    func_rep = torch.nn.DataParallel(func_rep)
func_rep.to(device)

# Set up training
trainer = Trainer(func_rep, lr=args.learning_rate)
coordinates, features = util.to_coordinates_and_features_3D(img)

scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
scaler.fit(features.numpy())
features = scaler.transform(features.numpy())
features = torch.from_numpy(features.astype(np.float32))

coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)
    
print(coordinates.shape)
print(features.shape)
    
# Calculate model size. Divide by 8000 to go from bits to kB
model_size = util.model_size_in_bits(func_rep) / 8000.
print(f'Model size: {model_size:.1f}kB')
fp_bpp = util.bpp(model=func_rep, image=img)
print(f'Full precision bpp: {fp_bpp:.2f}')

# Train model in full precision
trainer.train(coordinates, features, num_iters=args.num_iters)
print(f'Best training psnr: {trainer.best_vals["psnr"]:.2f}')
if args.log_measures:
    with open(args.logdir + '/log.pickle', 'wb') as handle:
        pickle.dump(trainer.logs, handle)

# Log full precision results
results['fp_bpp'].append(fp_bpp)
results['fp_psnr'].append(trainer.best_vals['psnr'])

# Save best model
torch.save(trainer.best_model, args.logdir + f'/best_model.pt')
    
# Save the input for the decoding
indata = {}
indata['input'] = coordinates.cpu().numpy()
scipy.io.savemat(args.logdir + f'/input_to_best_model.mat',indata)

# Update current model to be best model
func_rep.load_state_dict(trainer.best_model)

# Save full precision image reconstruction
with torch.no_grad():
    PredParams = func_rep(coordinates)

PredParams = scaler.inverse_transform(PredParams.cpu().numpy())
img_decompressed = np.reshape(PredParams, (img_tmp.shape[0], img_tmp.shape[1], img_tmp.shape[2], img_tmp.shape[3]))

# Save the prediction in NIFTI
print("Saving the decompressed NIFTI")
new_img = nib.nifti1.Nifti1Image(img_decompressed, None, header=new_header)
nib.save(new_img, args.logdir + f'/dwi_decompressed.nii.gz')


# Calculate model size. Divide by 8000 to go from bits to kB
model_size = util.model_size_in_bits(func_rep) / 8000.
print(f'Model size: {model_size:.1f}kB')
fp_bpp = util.bpp(model=func_rep, image=img)
print(f'Full precision bpp: {fp_bpp:.2f}')
