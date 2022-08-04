# Lossy compression of multidimensional medical images using sinusoidal activation networks: an evaluation study

[Pre-print on arXiv](https://arxiv.org/abs/2208.01602)

[Matteo Mancini](https://neurosnippets.com)<sup>1</sup>, Derek K. Jones<sup>1</sup>, [Marco Palombo](https://www.cardiff.ac.uk/people/view/2571014-palombo-marco)<sup>1,2,*</sup>

<sup>1</sup>Cardiff University Brain Research Imaging Centre (CUBRIC), Cardiff University, Cardiff, UK

<sup>2</sup>School of Computer Science and Informatics, Cardiff University, Cardiff, UK

<sup>*</sup>Corresponding author

This is the official implementation of the paper "Lossy compression of multidimensional medical images using sinusoidal activation networks: an evaluation study". The code allows to compress and decompress multidimensional medical imaging data (e.g. diffusion MRI) leveraging [implicit neural representation](https://github.com/vsitzmann/awesome-implicit-representations) and [period activation functions](https://arxiv.org/abs/2006.09661). The compression and decompression operations coincide respectively with the training and inference procedures.

The code is based on [SIREN](https://github.com/vsitzmann/siren) and [COIN](https://github.com/EmilienDupont/coin). It has been tested on `Ubuntu 18.04` with `CUDA 11.6`, `python 3.6.9` and a NVIDIA Titan XP GPU, as well as on `CentOS 7` with `CUDA 11.4`, `python 3.6.8` and up to four NVIDIA Tesla V100-SXM2-32GB.

## Get started

Once the repository has been cloned, we recommend to create a virtual environment and then install all the required packages:

    cd SirenMRI
    virtualenv venv
    source venv/bin/activate
    pip install -r requirements.txt

## Compression and decompression

To compress a given 4D dataset (here named `data.nii.gz`):

    python sirenMRI_2D.py -ld ~/compression_output -img data.nii.gz -op train -se 234

The option `-se` specifies the numerical value to use as random generator seed - the same number needs to be for the subsequent decompression.

To decompress the resulting `.7z` file:

    python sirenMRI_2D.py -x ~/compression_output/SirenCompression.7z -op decompress -se 234

## Reproducing the paper: getting the data

In the paper, we used the Adult Diffusion dataset from the MGH Human Connectome Project ([available here](https://db.humanconnectome.org/)). Specifically, we used the first five subjects (IDs: MGH1002,  MGH1001,  MGH1003,  MGH1004,  MGH1005). In the next section, it is assumed that the diffusion data to compress are specified using the environment variable `$datapath`.

## Reproducing the paper: running the experiments

To train the 2D network (performing compression and subsequent decompression):

    python sirenMRI_2D.py -ld ../2D_output -lm -img $datapath/diff_1k.nii.gz -se 321

In this way, the main parameters (number of layers, units per layer, etc.) are set to their default values, which reflect the optimal ones indicated in the paper. The `-lm` option allows to save, for each slice, the details about PSNR and training losses over the epochs.

To add a ReLU unit as the final activation (which makes the training more stable):

    python sirenMRI_2D.py -ld ../2D_w_relu_output -lm -img $datapath/diff_1k.nii.gz -se 321 -fa relu

To train the 3D network:

    python sirenMRI_3D.py -ld ../3D_output -lm -img $datapath/diff_1k.nii.gz -se 321

The 3D training procedure in most cases require more than one GPU. In specific environments (e.g. cluster nodes) it may be necessary to manually set the number of GPUs that are _visible_ to the Python script:

    # this example makes _visible_ the devices with IDs 0 and 1
    CUDA_VISIBLE_DEVICES=0,1 python sirenMRI_3D.py -ld ../3D_output -lm -img $datapath/diff_1k.nii.gz -se 321

Once the 3D training is done, one can compute the PSNR slice-wise (for comparison purposes with the 2D results) using the following script:

    python3 psnr_3d_slicewise.py -p ../3D_output/ -g $datapath/diff_1k.nii.gz -o ../3D_output

## Contact

If you have any issue/doubt/question, feel free to get in touch with the authors.