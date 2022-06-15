import torch
import tqdm
from collections import OrderedDict
from util import get_clamped_psnr


class Trainer():
    def __init__(self, representation, lr=1e-3, print_freq=1):
        """Model to learn a representation of a single datapoint.

        Args:
            representation (siren.Siren): Neural net representation of image to
                be trained.
            lr (float): Learning rate to be used in Adam optimizer.
            print_freq (int): Frequency with which to print losses.
        """
        self.representation = representation
        self.optimizer = torch.optim.Adam(self.representation.parameters(), lr=lr)
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.print_freq = print_freq
        self.steps = 0  # Number of steps taken in training
        self.loss_func = torch.nn.MSELoss()
        self.best_vals = {'psnr': 0.0, 'loss': 1e8}
        self.logs = {'psnr': [], 'loss': []}
        # Store parameters of best model (in terms of highest PSNR achieved)
        self.best_model = OrderedDict((k, v.detach().clone()) for k, v in self.representation.state_dict().items())

    def train(self, coordinates, latent, features, num_iters, batch_size=-1):
        """Fit neural net to image.

        Args:
            coordinates (torch.Tensor): Tensor of coordinates.
                Shape (num_points, coordinate_dim).
            features (torch.Tensor): Tensor of features. Shape (num_points, feature_dim).
            num_iters (int): Number of iterations to train for.
            batch_size (int): Size of mini-batches (-1 for one single batch per epoch)
        """
        if batch_size == -1:
            batch_size = coordinates.size()[0]

        with tqdm.trange(num_iters, ncols=150) as t:
            for i in t:
                permutation = torch.randperm(coordinates.size()[0])

                for b in range(0, coordinates.size()[0], batch_size):
                    # Update model
                    self.optimizer.zero_grad()

                    indices = permutation[i:i + batch_size]
                    batch_c, batch_l, batch_f = coordinates[indices, :], latent[indices, :], features[indices, :]

                    predicted = self.representation(batch_c, batch_l)
                    loss = self.loss_func(predicted, batch_f)
                    loss.backward()
                    self.optimizer.step()
                
                    # Calculate psnr
                    psnr = get_clamped_psnr(predicted, batch_f)

                    # Print results and update logs
                    log_dict = {'loss': loss.item(),
                                'psnr': psnr,
                                'best_psnr': self.best_vals['psnr'],
                                'loss_at_best_psnr': self.best_vals['loss']}
                    t.set_postfix(**log_dict)
                    for key in ['loss', 'psnr']:
                        self.logs[key].append(log_dict[key])

                    # Update best values
                    if psnr > self.best_vals['psnr']:
                        if loss.item() < self.best_vals['loss']:
                            self.best_vals['psnr'] = psnr
                            self.best_vals['loss'] = loss.item()
                            #self.scheduler.step(loss.item())
                            # If model achieves best PSNR and loss seen during training, update
                            # model
                            for k, v in self.representation.state_dict().items():
                                self.best_model[k].copy_(v)

