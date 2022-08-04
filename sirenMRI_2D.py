import argparse
import getpass
import os
import random
import sys
import torch
import util
import numpy as np
import nibabel as nib
from siren import Siren
from siren import MLP
from training import Trainer
from sklearn.preprocessing import MinMaxScaler
import pickle
import py7zr


parser = argparse.ArgumentParser()
parser.add_argument("-ld", "--logdir", help="Path to save output", default=f"/tmp/{getpass.getuser()}")
parser.add_argument("-lm", "--log_measures", help="Save measures for each epoch", action='store_true')
parser.add_argument("-ni", "--num_iters", help="Number of iterations to train for", type=int, default=2000)
parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=3e-4)
parser.add_argument("-se", "--seed", help="Random seed", type=int, default=random.randint(1, int(1e6)))
parser.add_argument("-img", "--image", help="Image to train on", default=f"/tmp/{getpass.getuser()}")
parser.add_argument("-lss", "--layer_size", help="Layer sizes as list of ints", type=int, default=256)
parser.add_argument("-nl", "--num_layers", help="Number of layers", type=int, default=3)
parser.add_argument("-w0", "--w0", help="w0 parameter for SIREN model.", type=float, default=30.0)
parser.add_argument("-w0i", "--w0_initial", help="w0 parameter for first layer of SIREN model.", type=float, default=30.0)
parser.add_argument("-m", "--model", help="Model to use. Implemented are siren or mlp.", default="siren")
parser.add_argument("-fa", "--final_activation", help="Final activation function (SIREN model-only).", default="identity")
parser.add_argument("-a", "--activation", help="Activation function to use with mlp (relu or tanh).", default="relu")
parser.add_argument("-ssr", "--single_siren_start", help="Substitute first layer with SIREN (MLP model-only).", action='store_true')
parser.add_argument("-sse", "--single_siren_end", help="Substitute last layer with SIREN (MLP model-only).", action='store_true')
parser.add_argument("-op", "--operation", help="Operation to perform (train+decompress, train, decompress).", default="train+decompress")
parser.add_argument("-x", "--archive", help="7zip archive to decompress (for decompression only).", default="")

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

if 'train' in args.operation:
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

    # Prepare to save the decompressed image
    new_header = nii.header.copy()
    img_decompressed = img_tmp
    scaler = []
else:
    with py7zr.SevenZipFile(args.archive, 'r') as archive:
        archive.extractall(path=".")
    compression_folder = "SirenCompression"
    with open(compression_folder + f'/input_to_best_model.pickle', 'rb') as handle:
        indata = pickle.load(handle)
    img_decompressed = np.zeros(indata['input'])
    sx, sy, sz, vols = indata['input']
    new_header = indata['header']
    scaler = indata['scaler']

print(f'Image size: {sx} x {sy} x {sz} x {vols}')

for i in range(sz):

    slice_to_process = i

    # Setup model
    if args.model == 'siren':
        func_rep = Siren(
            dim_in=2,
            dim_hidden=args.layer_size,
            dim_out=vols,
            num_layers=args.num_layers,
            final_activation=final_activation[args.final_activation],
            w0_initial=args.w0_initial,
            w0=args.w0
        ).to(device)
    elif args.model == 'mlp':
        func_rep = MLP(
            dim_in=2,
            dim_hidden=args.layer_size,
            dim_out=vols,
            num_layers=args.num_layers,
            activation=mlp_activation[args.activation],
            siren_start=args.single_siren_start,
            siren_end=args.single_siren_end
        ).to(device)
    else:
        print(f'Unknown model: {args.model}')
        sys.exit(1)

    if 'train' in args.operation:
        print("")
        print(f'Compressing slice: {slice_to_process}')

        # Set up training
        trainer = Trainer(func_rep, lr=args.learning_rate)

        img = img_tmp[:,:,slice_to_process,:]
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img.astype(np.float32))

        coordinates, features = util.to_coordinates_and_features_2D(img)

        slice_scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
        slice_scaler.fit(features.numpy())
        features = slice_scaler.transform(features.numpy())
        features = torch.from_numpy(features.astype(np.float32))
        scaler.append(slice_scaler)

        coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)

        # Train model in full precision
        trainer.train(coordinates, features, num_iters=args.num_iters)
        print(f'Best training psnr: {trainer.best_vals["psnr"]:.2f}')
        print(f'Best training loss: {trainer.best_vals["loss"]:.2f}')
        if args.log_measures:
            with open(args.logdir + '/logs/log_slice_' + str(i) + '.pickle', 'wb') as handle:
                pickle.dump(trainer.logs, handle)

        # Save best model
        torch.save(trainer.best_model, compression_folder + f'/best_model_slice_{slice_to_process}.pt')

        # Update current model to be best model
        func_rep.load_state_dict(trainer.best_model)
    else:
        print("")
        print(f'Decompressing slice: {slice_to_process}')

        img = img_decompressed[:,:,slice_to_process,:]
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img.astype(np.float32))

        coordinates = util.to_coordinates_and_features_2D(img)
        coordinates = coordinates[0].to(device, dtype)

        func_rep.load_state_dict(torch.load(compression_folder + f'/best_model_slice_{slice_to_process}.pt'))

    if 'decompress' in args.operation:
        # Save full precision image reconstruction
        with torch.no_grad():
            PredParams = func_rep(coordinates)
        PredParams = scaler[slice_to_process].inverse_transform(PredParams.cpu().numpy())
        img_decompressed[:, :, slice_to_process, :] = np.reshape(PredParams,
                                                                (img_decompressed.shape[0],
                                                                 img_decompressed.shape[1],
                                                                 img_decompressed.shape[3]))

if 'decompress' in args.operation:
    # Save the prediction in NIFTI
    print("Saving the decompressed NIFTI")
    new_img = nib.nifti1.Nifti1Image(img_decompressed, None, header=new_header)
    nib.save(new_img, args.logdir + f'/dwi_decompressed.nii.gz')

if 'train' in args.operation:
    # Save the input for the decoding
    indata = {'input': np.array([sx, sy, sz, vols]), 'header': new_header, 'scaler': scaler}
    with open(compression_folder + f'/input_to_best_model.pickle', 'wb') as handle:
        pickle.dump(indata, handle)
    # Compress the output
    with py7zr.SevenZipFile(compression_folder + '.7z', 'w') as archive:
        archive.writeall(compression_folder, "SirenCompression")