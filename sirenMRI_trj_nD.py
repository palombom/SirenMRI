import argparse
import getpass
import os
import random
import sys
import torch
import util
import numpy as np
from siren import Siren
from siren import MLP
from training import Trainer
from sklearn.preprocessing import MinMaxScaler
import pickle
#import py7zr


parser = argparse.ArgumentParser()
parser.add_argument("-ld", "--logdir", help="Path to save output", default=f"/tmp/{getpass.getuser()}")
parser.add_argument("-lm", "--log_measures", help="Save measures for each epoch", action='store_true')
parser.add_argument("-ni", "--num_iters", help="Number of iterations to train for", type=int, default=2000)
parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=3e-4)
parser.add_argument("-se", "--seed", help="Random seed", type=int, default=random.randint(1, int(1e6)))
parser.add_argument("-img", "--image", help="Image to train on", default=f"/tmp/{getpass.getuser()}")
parser.add_argument("-bw", "--batch_w", help="Size of walkers batch.", type=int, default=1000)
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
    b_size = args.batch_w

    # Fit images
    print("")
    print(f'Compressing Image {img_to_compress}')

    # Load image
    with open(img_to_compress, 'rb') as handle:
        trj = pickle.load(handle)
    init_pos = trj['init']
    trj_series = trj['series']
    n_walkers = trj_series.shape[0]
    trj_len = trj_series.shape[1]
    coord = trj_series.shape[2]
    trj_decompressed = trj_series

    scaler = {c: [] for c in range(coord)}
else:
    with py7zr.SevenZipFile(args.archive, 'r') as archive:
        archive.extractall(path=".")
    compression_folder = "SirenCompression"
    with open(compression_folder + f'/input_to_best_model.pickle', 'rb') as handle:
        in_data = pickle.load(handle)
    n_walkers, trj_len, coord = in_data['input']
    init_pos = in_data['init']
    trj_decompressed = np.zeros(in_data['input'])
    scaler = in_data['scaler']
    b_size = in_data['batch_size']

print(f'Size: {n_walkers} x {trj_len} x {coord}')

for i in range(int(n_walkers/b_size)):

    # Setup model
    if args.model == 'siren':
        func_rep = Siren(
            dim_in=3,
            dim_hidden=args.layer_size,
            dim_out=trj_len*coord,
            num_layers=args.num_layers,
            final_activation=final_activation[args.final_activation],
            w0_initial=args.w0_initial,
            w0=args.w0
        ).to(device)
    elif args.model == 'mlp':
        func_rep = MLP(
            dim_in=3,
            dim_hidden=args.layer_size,
            dim_out=trj_len*coord,
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
        print(f'Compressing batch: {i}')

        # Set up training
        trainer = Trainer(func_rep, lr=args.learning_rate)

        init_batch = init_pos[i * b_size:(i + 1) * b_size, :]
        #for j in range(coord):
        trj_batch = trj_series[i*b_size:(i+1)*b_size, :, :]

        coordinates = torch.from_numpy(init_batch.astype(np.float32))
        features = torch.from_numpy(trj_batch.astype(np.float32))

        for j in range(coord):
            slice_scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
            slice_scaler.fit(features[:,:,j].numpy())
            features_np = slice_scaler.transform(features[:,:,j].numpy())
            features[:,:,j] = torch.from_numpy(features_np.astype(np.float32))
            scaler[j].append(slice_scaler)

        features = torch.reshape(features, (b_size, trj_len*coord))
        coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)

        # Train model in full precision
        trainer.train(coordinates, features, num_iters=args.num_iters)
        print(f'Best training psnr: {trainer.best_vals["psnr"]:.2f}')
        print(f'Best training loss: {trainer.best_vals["loss"]:.2f}')
        if args.log_measures:
            with open(args.logdir + '/logs/log_batch_' + str(i) + '_coord_' + str(j) + '.pickle', 'wb') as handle:
                pickle.dump(trainer.logs, handle)

        # Save best model
        torch.save(trainer.best_model, compression_folder + f'/best_model_batch_{i}.pt')

        # Update current model to be best model
        func_rep.load_state_dict(trainer.best_model)
    else:
        print("")
        print(f'Decompressing batch: {i}')

        init_batch = init_pos[i * b_size:(i + 1) * b_size, :]
        coordinates = torch.Tensor(init_batch)

        func_rep.load_state_dict(torch.load(compression_folder + f'/best_model_batch_{i}.pt'))

    if 'decompress' in args.operation:
        # Save full precision image reconstruction
        with torch.no_grad():
            PredParams = func_rep(coordinates)
        PredParams = PredParams.view(b_size, trj_len, coord).cpu().numpy()
        for j in range(coord):
            PredParams[:,:,j] = scaler[j][i].inverse_transform(PredParams[:,:,j])
            trj_decompressed[i * b_size:(i + 1) * b_size, :, j] = PredParams[:,:,j]

if 'decompress' in args.operation:
    # Save the prediction
    print("Saving the decompressed trajectories")
    with open(args.logdir + f'/trajectories_decompressed.pickle', 'wb') as handle:
        pickle.dump({'series': trj_decompressed, 'init': init_pos}, handle)

if 'train' in args.operation:
    # Save the input for the decoding
    indata = {'input': np.array([n_walkers, trj_len, coord]), 'scaler': scaler, 'batch_size': b_size, 'init': init_pos}
    with open(compression_folder + f'/input_to_best_model.pickle', 'wb') as handle:
        pickle.dump(indata, handle)
    # Compress the output
    with py7zr.SevenZipFile(compression_folder + '.7z', 'w') as archive:
        archive.writeall(compression_folder, "SirenCompression")