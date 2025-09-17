# DeepSES

[![Python](https://img.shields.io/badge/Python-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-orange.svg)](https://pytorch.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-blue.svg)](https://isocpp.org/)
[![OpenGL](https://img.shields.io/badge/OpenGL-lightgrey.svg)](https://www.opengl.org/)
[![CUDA](https://img.shields.io/badge/CUDA-red.svg)](https://developer.nvidia.com/cuda-zone)
[![TensorRT](https://img.shields.io/badge/TensorRT-red.svg)](https://developer.nvidia.com/tensorrt)

This repository contains the code accompanying the paper **DeepSES: Learning solvent-excluded surfaces via neural signed distance fields** ([link to paper]).   
It can be used to train a neural network for predicting the signed distance field (SDF) of the solvent-excluded surface (SES) of a molecule from the SDF of the van-der-Waals (vdW) surface and render the SES at interactive frame rates.

---

## Table of Contents

- [Overview](#overview)  
- [Paper](#paper)  
- [Training](#training)  
  - [Environment Setup](#environment-setup)  
  - [Data](#data)  
  - [Training Instructions](#training-instructions)  
- [Execution](#execution)  
  - [Singularity Container](#singularity-container)  
  - [Building and Running](#building-and-running)   
- [Citing this work](#citing-this-work)  
- [License](#license)  

---

## Overview

This project implements the methods described in our paper and contains two main components:  

1. **Training code** in Python: We use Pytorch to train a 3D convolutional neural network (CNN) to predict the SDF of the SES from the SDF of the vdW surface for molecules of different sizes.     
The model works on patches of size 64³ from the whole SDF for which we set a default resolution of 512³, but it can also be used for larger resolutions.    
Each 64³ patch is randomly sampled from a set of molecules and from the whole 512³ SDF. Additionally, we apply a random augmentation to each patch (mirroring or flipping axes).    
The training data can be found [here](https://doi.org/10.5281/zenodo.17086718).

2. **Execution code** in C++: We use OpenGL, CUDA and TensorRT to compute and render the SDF of the SES.    
The trained Pytorch model is saved as an .onnx file and used to create TensorRT engine for inference, which is optimized for the used hardware.     
The pipeline works roughly as follows:  
    - A compute shader in OpenGL computes the vdW SDF (3D texture). 
    - This texture is then mapped to a CUDA buffer and passed through the TensorRT engine to compute the SES SDF (simplified, see below).
    - The buffer is then unmapped from CUDA to let OpenGL use it to render the SES via raymarching.  

    To speed up the inference (and save some GPU memory) we only compute the SES SDF for the patches which are visible from the current camera position and where it differs from the vdW SDF. Those patches are determined by raymarching the vdW SDF and filtered before copying them into a CUDA buffer for inference.

---

## Paper

This repository accompanies the following paper:

**Title:** *DeepSES: Learning solvent-excluded surfaces via neural signed distance fields*  
**Authors:** Niklas Merk, Anna Sterzik, Kai Lawonn  
**Published in:** *Computers & Graphics (C&G), VCBM 2025*  
**Paper link:** [View Paper](INSERT_PAPER_LINK_HERE)

---

## Training

### Environment Setup

Create the Conda environment:

```bash
cd train
conda env create --file=environment.yml
conda activate deepses
````

### Data

Download and extract the training data into the training directory:

```bash
wget https://doi.org/10.5281/zenodo.17086719/files/sdf_data.zip
unzip sdf_data.zip
```

### Training Instructions

In the file [learn.py](learn.py) you can adjust the training parameters (choose the model, batch size, number of epochs, how much ram to use, etc.)     
Run training with:

```bash
python learn.py
```

---

## Execution

### Singularity Container

The C++ execution code is designed to run inside a Singularity container. The `.def` file is included for building the container.

### Building and Running

1. Build the container:

```bash
cd run
sudo singularity build deepses.sif deepses.def
```

2. Run the container:

```bash
singularity exec --nv deepses.sif bash
```

3. Build the C++ code inside the container:

```bash
mkdir build && cd build
cmake ..
make
```
4. Create a TensorRT engine and run deepses:
```bash
cd apps
./create_engine # this may take some time
#NOTE: you can run other pre-trained models by adjusting the paths in create_engine.cc and interactive-deepses.cc

# execute deepses for a molecule file
./interactive-deepses <path_to_this_repo>/deepses/run/data/pdb/vcbm_eval/<some_molecule_file>.cif.gz 8
#NOTE: the second argument is the number of patches (size 64) to use per dimension, this means 8 patches will result in a texture of size (8 * 64 = 512)³.

# You can also run deepses with ambient occlusion
./interactive-deepses_with_ao <path_to_this_repo>/deepses/run/data/pdb/vcbm_eval/<some_molecule_file>.cif.gz 8
```

5. You can also run a reimplementation of the [Hermosilla method](https://doi.org/10.1007/s00371-017-1397-2) or render just the van-der-Waals surface
```bash
# Hermosilla method
./interactive-hermosilla-method <path_to_this_repo>/deepses/run/data/pdb/vcbm_eval/<some_molecule_file>.cif.gz 8

# vdW surface
./interactive-vdw <path_to_this_repo>/deepses/run/data/pdb/vcbm_eval/<some_molecule_file>.cif.gz 8
```

---

## Downloading other .cif.gz / .pdb files form the PDB

This repository also includes a bash script to download random files for molecules from the [PDB](https://www.rcsb.org/). You can adjust how many files should be downloaded and in which range the number of atoms of the molecules should be.

```bash
cd run
chmod +x download_random_pdb_or_cif.sh
./download_random_pdb_or_cif.sh <num_files> <save_folder> <atom_min> <atom_max>
```

---

## Citing this work

If you use this code in your research, please cite our paper:

```
@article{your2025paper,
  title={DeepSES: Learning Solvent-Excluded Surfaces via Neural Signed Distance Fields},
  author={Merk, Niklas and Sterzik, Anna and Lawonn, Kai},
  journal={Computers & Graphics (C&G), VCBM},
  year={2025},
  url={INSERT_PAPER_LINK_HERE}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).