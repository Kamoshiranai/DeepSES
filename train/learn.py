import torch
import os
import h5py
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from datetime import datetime
import time
from einops import rearrange
from tqdm import tqdm
import warnings
from typing import Tuple, List

from models import UNet_no_attn, My_UNet_3x

# training parameters

if __name__ == "__main__":
    print("finished imports...")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # NOTE: gpu version of torch not installed on local machine...
SEED = 12345
BATCH_SIZE = 32 # NOTE adjust based on gpu memory
EPOCH_SIZE = None # if None, "whole" training dataset is used for one epoch
START_LEARNING_RATE = 1e-4 # after warm-up
END_LEARNING_RATE = 1e-5
BETA_1 = 0.9
BETA_2 = 0.999
print_loss_per_epoch = False
print_loss_per_batch = False
print_status_info = True
BATCHES_2_AVERAGE_LOSS = 16 # average loss over BATCHES_2_AVERAGE_LOSS and return to tb on each validation
EPOCHS = 100
VALIDATE_EVERY_EPOCH = True #NOTE: if False, validates only after each pass through whole dataset
save_best_model = True

# model parameters
NUM_INNER_CHANNELS = 4
NORM_GROUPS = 1
CHANNEL_MULTS = [1,2,4]
RES_BLOCKS = 1
DROP_OUT = 0
USE_ND_DROP_OUT = False

training_data_ses_path = 'train/sdf_data/ses'
training_data_vdw_path = 'train/sdf_data/vdw'

torch.manual_seed(SEED)
torch.set_default_dtype(torch.float32)
numpy_rng = np.random.default_rng(SEED)

def trivial_collate_fn(x):
    return x

class HDF5_Dataset(torch.utils.data.Dataset):
    """
    Creates a dataset of the ses and vdw files. 
    It can be iterated over the 3d patches into which each vdw/ses file is implictly split.
    Basically two large torch tensors storing part of the dataset into RAM where the rest is only loaded when needed.

    Parameters:
    -----------
    path_vdw: str,
        path to the vdw files (.hdf).
    path_ses: str,
        path to the ses files (.hdf).
    file_names: List[str],
        explicit file names to use from the above paths.
    number_of_files_ram: int,
        How many of those files to store in RAM, the rest is only loaded when needed.
    patch_size: int = 64,
        Size of the patches into which each ses/vdw file is partitioned. We used signed distance fields of size 512^3 and partitioned them into patches of size 64^3.
    """
    def __init__(self, path_vdw: str, path_ses: str, file_names: List[str], number_of_files_ram: int, patch_size: int = 64):
        super().__init__()
    
        self.path_vdw = path_vdw
        self.path_ses = path_ses
        self.file_names = file_names # tells which files belong to this data set
        self.number_of_files = len(file_names)
        self.number_of_files_ram = min(number_of_files_ram, self.number_of_files)
        self.patch_size = patch_size
        # get number of patches from first file
        with h5py.File(os.path.join(path_vdw, file_names[0]), 'r') as f:
            self.number_of_patches = f['texture'].shape[0] // patch_size

        # NOTE: load only "num_files_ram" into RAM
        self.vdw_data_ram = torch.zeros((self.number_of_files_ram, self.number_of_patches*self.patch_size, self.number_of_patches*self.patch_size, self.number_of_patches*self.patch_size))
        self.ses_data_ram = torch.zeros((self.number_of_files_ram, self.number_of_patches*self.patch_size, self.number_of_patches*self.patch_size, self.number_of_patches*self.patch_size))

        for file_number, file_name in tqdm(enumerate(file_names[:number_of_files_ram]), total = self.number_of_files_ram):
            with h5py.File(os.path.join(path_vdw, file_name), 'r') as f_vdw:
                self.vdw_data_ram[file_number] = torch.tensor(f_vdw['texture'][:], dtype = torch.float32)

            with h5py.File(os.path.join(path_ses, file_name), 'r') as f_ses:
                self.ses_data_ram[file_number] = torch.tensor(f_ses['texture'][:], dtype = torch.float32)

    def __getitems__(self, batch_indices: list[int]):

        batch_size = len(batch_indices)

        file_indices, patch_indices_x, patch_indices_y, patch_indices_z = np.unravel_index(batch_indices, (self.number_of_files, self.number_of_patches, self.number_of_patches, self.number_of_patches))

        patch_indices_x *= self.patch_size
        patch_indices_y *= self.patch_size
        patch_indices_z *= self.patch_size
        
        vdw_patch = [ self._get_vdw_patch( file_indices[i], (patch_indices_x[i], patch_indices_y[i], patch_indices_z[i]) ) for i in range(batch_size) ]
        ses_patch = [ self._get_ses_patch( file_indices[i], (patch_indices_x[i], patch_indices_y[i], patch_indices_z[i]) ) for i in range(batch_size) ]

        # add channel axis to get a shape of (batch_size, channel, 3d image)
        vdw_patch = rearrange(vdw_patch, "b d1 d2 d3 -> b 1 d1 d2 d3")
        ses_patch = rearrange(ses_patch, "b d1 d2 d3 -> b 1 d1 d2 d3")

        return vdw_patch, ses_patch

    def read_hdf5_data(self, file_index, patch_indices, identifier: str):
        """
        Safely reads specified patch of 'texture' dataset from an HDF5 file.
        """
        if identifier == "vdw":
            file_path = os.path.join(self.path_vdw, self.file_names[file_index])
        elif identifier == "ses":
            file_path = os.path.join(self.path_ses, self.file_names[file_index])
        else:
            raise ValueError(f"unknown identifier {identifier}, must be 'vdw' or 'ses'.")

        with h5py.File(file_path, 'r') as f:
            patch_data = torch.tensor(f['texture'][patch_indices[0]: patch_indices[0]+self.patch_size, patch_indices[1]: patch_indices[1]+self.patch_size, patch_indices[2]: patch_indices[2]+self.patch_size], dtype = torch.float32)
            return patch_data
        
    def _get_vdw_patch(self, file_index, patch_indices):
        """
        Read file from ram oder main memory depending on file index.
        """
        if file_index < self.number_of_files_ram:
            vdw_patch = self.vdw_data_ram[file_index, patch_indices[0]:patch_indices[0] + self.patch_size, patch_indices[1]:patch_indices[1] + self.patch_size, patch_indices[2]:patch_indices[2] + self.patch_size]
        elif file_index < self.number_of_files:
            vdw_patch = self.read_hdf5_data(file_index, patch_indices, "vdw")
        else:
            raise ValueError(f"file_index {file_index} exceeds number of files = {self.number_of_files}")
        
        return vdw_patch
    
    def _get_ses_patch(self, file_index, patch_indices):
        """
        Read file from ram oder main memory depending on file index.
        """
        if file_index < self.number_of_files_ram:
            ses_patch = self.ses_data_ram[file_index, patch_indices[0]:patch_indices[0] + self.patch_size, patch_indices[1]:patch_indices[1] + self.patch_size, patch_indices[2]:patch_indices[2] + self.patch_size]
        elif file_index < self.number_of_files:
            ses_patch = self.read_hdf5_data(file_index, patch_indices, "ses")
        else:
            raise ValueError(f"file_index {file_index} exceeds number of files = {self.number_of_files}")
        
        return ses_patch

    def __len__(self):
        return len(self.file_names) * self.number_of_patches**3
    
class HDF5_Dataset_sliding_window(HDF5_Dataset):
    """
    Instead of fixed patches samples the position of a 3d "sliding" window to load a patch. Additionally applies augmentation (see self.augment_tensor) to each patch.
    
    Parameters:
    -----------
    path_vdw: str,
        path to the vdw files (.hdf).
    path_ses: str,
        path to the ses files (.hdf).
    file_names: List[str],
        explicit file names to use from the above paths.
    epoch_size: int = None,
        Determines how many patches are sampled per epoch. If None then one epoch iterates over the whole dataset (same number of samples as it would be without random sliding window).
    number_of_files_ram: int,
        How many of those files to store in RAM, the rest is only loaded when needed.
    patch_size: int = 64,
        Size of the patches into which each ses/vdw file is partitioned. We used signed distance fields of size 512^3 and partitioned them into patches of size 64^3.
    """
    def __init__(self, path_vdw, path_ses, file_names: list[str], epoch_size: int = None, number_of_files_ram: int = 0, patch_size: int = 64):
        super().__init__(path_vdw, path_ses, file_names, number_of_files_ram, patch_size)

        # set up rng for sampling patches
        self.rng = np.random.default_rng() #NOTE: take care when spawning multiple workers in the Dataloader, as they will need different seeds!

        if epoch_size == None:
            self.epoch_size = len(self.file_names) * self.number_of_patches**3 # sample as many patches as we had before
        else:
            self.epoch_size = epoch_size

    def augment_tensor(self, tensor: torch.Tensor, augmentation_lists: Tuple[List[int]]) -> torch.Tensor:
        """
        Applies a transformation to a 3D tensor,
        which can include:
        1. Randomly flipping one or more axes (0, 1, 2).
        2. Randomly permuting the axes (0, 1, 2).

        Args:
            tensor: A 3D PyTorch tensor (e.g., shape (D, H, W)).

        Returns:
            The randomly transformed 3D tensor.

        Raises:
            ValueError: If the input tensor is not 3D.
        """
        if tensor.ndim != 3:
            raise ValueError(f"Input tensor must be 3D, but got {tensor.ndim} dimensions.")

        transformed_tensor = tensor.clone() # work on a copy

        axes_to_flip = augmentation_lists[0]
        permuted_axes_list = augmentation_lists[1]

        # Apply flipping if any axes were selected
        if axes_to_flip:
            transformed_tensor = torch.flip(transformed_tensor, dims=axes_to_flip)
        
        # Apply permutation only if it's different from the original order
        original_axes_list = [0, 1, 2]
        if permuted_axes_list != original_axes_list:
            transformed_tensor = transformed_tensor.permute(permuted_axes_list)

        return transformed_tensor

    def random_augmentation_ids(self) -> Tuple[List[int], List[int]]:

        # Random Flipping using NumPy RNG

        # Decide which axes to flip (generate 3 random booleans)
        flip_booleans = self.rng.choice([True, False], size=3)
        axes_to_flip = [i for i, flip in enumerate(flip_booleans) if flip]

        # Generate a random permutation of the axes (0, 1, 2)

        # rng.permutation returns a *new* shuffled array/sequence
        permuted_axes_np = self.rng.permutation(3)
        permuted_axes_list = permuted_axes_np.tolist() # Convert to list for torch.permute

        return (axes_to_flip, permuted_axes_list)

    def __getitems__(self, batch_indices: list[int]):
        
        batch_size = len(batch_indices)

        file_indices = self.rng.integers(0, self.number_of_files, size = batch_size)
        patch_indices = self.rng.integers(0, (self.number_of_patches-1)*self.patch_size + 1, size = (batch_size, 3))

        # determine augmentations
        augmentations = [self.random_augmentation_ids() for _ in range(batch_size)]

        # apply augmentations to chosen vdw and sdf patches
        vdw_patch = [ self.augment_tensor( self._get_vdw_patch(file_indices[i], patch_indices[i] ), augmentations[i] ) for i in range(batch_size) ]
        ses_patch = [ self.augment_tensor( self._get_ses_patch(file_indices[i], patch_indices[i] ), augmentations[i] ) for i in range(batch_size) ]

        # add channel axis to get a shape of (batch_size, channel, 3d image)
        vdw_patch = rearrange(vdw_patch, "b d1 d2 d3 -> b 1 d1 d2 d3")
        ses_patch = rearrange(ses_patch, "b d1 d2 d3 -> b 1 d1 d2 d3")

        return vdw_patch, ses_patch
    
    def __len__(self):
        return self.epoch_size
    
def seed_worker(worker_id):

    """Create a unique seed based on the worker id and some entropy used for sampling patch_indices (see HDF5_Dataset_sliding_window's _get_items function)"""

    seed = np.random.SeedSequence().entropy + worker_id
    # Set a worker-specific RNG in the dataset
    # Assuming dataset is a global or can be set in some shared way
    # Alternatively, you can reassign the generator in each worker's copy of the dataset:
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.rng = np.random.default_rng(seed) #NOTE: alternatively could try to set a new seed for each workers copy of the dataset

def fw_pass(model, loss_fn, X, y):

    # determine patches where at least one value is above 1.4 (probe radius)
    with torch.no_grad():
        mask = (X < 1.4).any(dim = (1, 2, 3, 4))

    predictions = model(X)

    loss = loss_fn(predictions, y)

    with torch.no_grad():
        raw_loss = loss.mean()

    # ignore masked patches in loss calculation
    masked_loss = loss[mask]
    modified_loss = masked_loss.sum() / (mask.sum()*X.shape[-1]**3 + 1e-9) # normalize with respect to mask
    
    return modified_loss, raw_loss
    
def train_one_epoch(training_loader, loss_fn, optimizer, scheduler, epoch_number, tensorboard_writer):

    running_loss = torch.zeros(1, device = DEVICE)
    running_raw_loss = torch.zeros(1, device = DEVICE)
    total_time = 0
    total_batches = 0

    # training loop
    for i, (X_batch, y_batch) in enumerate(training_loader):

        total_batches += 1
        tensorboard_time = epoch_number * len(training_loader) + i + 1
        tensorboard_writer.add_scalar('LR', scheduler.get_last_lr()[0], tensorboard_time)

        # Start recording gpu time
        start_batch = time.perf_counter()

        # move training batch to device
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        #NOTE: reduce memory usage by function scoping of python (discards predictions from last fw pass before computing next fw pass) -> slight speed up

        loss, raw_loss = fw_pass(model, loss_fn, X_batch, y_batch)

        elapsed_time = time.perf_counter() - start_batch

        total_time += elapsed_time

        # optionally deleting X_batch and y_batch to free up memory
        del X_batch
        del y_batch

        optimizer.zero_grad()
        loss.backward() # changes gradients      
        optimizer.step() # changes weights
        # Suppress warning of more workers than recommended
        warnings.filterwarnings("ignore", category=UserWarning, message="The epoch parameter in*")
        scheduler.step()

        running_loss += loss.detach() #NOTE loss.item() might force synchronization
        running_raw_loss += raw_loss.detach()

        if print_loss_per_batch:
            print(f'batch {i+1}, loss: {loss.item():.2e}')
        tensorboard_writer.add_scalar('Time/DPPS', BATCH_SIZE / elapsed_time, tensorboard_time)

    tensorboard_writer.add_scalar('Raw_Loss/train', raw_loss.item() / total_batches, epoch_number + 1)
            
    return running_loss.item() / total_batches
    
if __name__ == "__main__":

    if print_status_info:
        print("using device:", DEVICE)
        
    # set up model and optimizer

    model = UNet_no_attn(
        dims = 3,
        in_channel = 1,
        out_channel = 1,
        inner_channel = NUM_INNER_CHANNELS,
        norm_groups = NORM_GROUPS, 
        channel_mults = CHANNEL_MULTS,
        res_blocks = RES_BLOCKS,
        dropout = DROP_OUT,
        use_nd_dropout = USE_ND_DROP_OUT,
    )

    # model = My_UNet_3x(
    #     dims = 3,
    #     in_channel = 1,
    #     out_channel = 1,
    #     inner_channel = NUM_INNER_CHANNELS,
    #     norm_groups = NORM_GROUPS,
    #     dropout = DROP_OUT,
    #     use_nd_dropout = USE_ND_DROP_OUT,
    # )

    model.to(DEVICE) #NOTE: needs to be called before loading optimizer dict

    loss_fn = torch.nn.MSELoss(reduction = "none") #NOTE: this lets us exclude certain patches

    # PREPROCESS + PREPARE DATA

    # split sdf files into train, validate and test set
    if print_status_info:
        print("create train, validate and test split...")

    sdf_file_names = [name for name in os.listdir(training_data_vdw_path) if os.path.isfile(os.path.join(training_data_vdw_path, name))] 

    # test-validate/test split
    file_names_train, file_names_test = train_test_split(sdf_file_names, train_size=0.8, shuffle=True, random_state=SEED)

    # exclude eval files used later
    exclude_files = ["1YX7.h5", "8HDZ.h5", "6RUT.h5", "6W1N.h5", "5WIS.h5", "8TRZ.h5", "8G3D.h5", "8GLV.h5"]
    file_names_train = [name for name in file_names_train if name not in exclude_files]

    # test-validation split for evaluation of the chosen model
    file_names_test, file_names_validate = train_test_split(file_names_test, train_size=0.5, shuffle=True, random_state=SEED)

    # Initialize train, validate and test Datasets (custom) with index lists
    if print_status_info:
        print("initialize datasets...")

    # NOTE: here you can set the number of files from the train, validation and test set that should be stored in the RAM. One file (actually vdw + ses file) is ~1GB in total (can adjust for your available RAM)
    training_set = HDF5_Dataset_sliding_window(training_data_vdw_path, training_data_ses_path, file_names_train, number_of_files_ram = 32, epoch_size = EPOCH_SIZE)
    validation_set = HDF5_Dataset(training_data_vdw_path, training_data_ses_path, file_names_validate, number_of_files_ram = 8)
    test_set = HDF5_Dataset(training_data_vdw_path, training_data_ses_path, file_names_test, number_of_files_ram = 0)

    # prepare optimizer
    steps_per_epoch = len(training_set) / BATCH_SIZE
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = steps_per_epoch

    optimizer = torch.optim.Adam(model.parameters(), lr=START_LEARNING_RATE, betas = (BETA_1, BETA_2))
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = 1/warmup_steps, end_factor = 1.0, total_iters = warmup_steps - 1)
    annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps - warmup_steps, END_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, annealing_scheduler], [warmup_steps])

    # Initialize Dataloaders from those datasets
    if print_status_info:
        print("initialize dataloaders...")
        
    # NOTE: default collate function is neccessary to create batches, when __getitem__ is used. For __getitems__ use identity
    torch_rng_cpu = torch.Generator()
    torch_rng_cpu.manual_seed(SEED)

    # Suppress warning of more workers than recommended
    warnings.filterwarnings("ignore", category=UserWarning, message=".*DataLoader will create.*")

    training_loader = DataLoader(training_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = 8, generator = torch_rng_cpu, pin_memory = True, persistent_workers = True, worker_init_fn = seed_worker, collate_fn=trivial_collate_fn) #NOTE: if using HDF5_Dataset_sliding_window with multiple workers, we need to set a different seed for the rng for the sliding window of each worker. But we can also omit shuffling.
    validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, num_workers = 4, generator = torch_rng_cpu, pin_memory=True, collate_fn=lambda x: x)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers = 4, generator = torch_rng_cpu, pin_memory=True, collate_fn=lambda x: x)

    timestamp = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
    tb_writer = SummaryWriter(f"runs/{timestamp}")

    # write parameters to tensorboard

    parameters = {
        "model": str(type(model)),
        "number of inner channels": NUM_INNER_CHANNELS,
        "norm groups": NORM_GROUPS,
        "channel_mults" : CHANNEL_MULTS,
        "res_blocks" : RES_BLOCKS,
        "drop out": DROP_OUT,
        "use nd drop out": USE_ND_DROP_OUT,
        "learning rate": START_LEARNING_RATE, 
        "batch size": BATCH_SIZE, 
        "number of epochs": EPOCHS,
        "patch size": training_set.patch_size,
        "device": DEVICE,
        "seed": SEED,
    }
    tb_writer.add_text("Parameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in parameters.items()])),)
    
    # add train, validate and test file names to tb
    tb_writer.add_text("Train Files", "\n".join(file_names_train), global_step = 0)
    tb_writer.add_text("Validation Files", "\n".join(file_names_validate), global_step = 0)
    tb_writer.add_text("Test Files", "\n".join(file_names_test), global_step = 0)

    # TRAINING

    if print_status_info:
        print("start training")
    start_training = time.time()        
    best_validation_loss = np.inf

    for epoch in tqdm(range(EPOCHS)):
        if print_loss_per_epoch:
            print("Epoch {}:".format(epoch + 1))

        model.train(True)
        start_epoch = time.perf_counter()
        training_loss = train_one_epoch(training_loader, loss_fn, optimizer, scheduler, epoch, tb_writer)
        if print_loss_per_epoch:
            print(f"Loss training {training_loss:.2e}, duration {(time.perf_counter() - start_epoch) / 60:.2f} min")
        tb_writer.add_scalar('Time/Epoch_duration', (time.perf_counter() - start_epoch), epoch + 1)

        #NOTE: Validate only after one pass through dataset
        if (EPOCH_SIZE is None) or ( epoch >= 2 and ((epoch+1) * EPOCH_SIZE) % (len(file_names_train) * training_set.number_of_patches**3) <= EPOCH_SIZE ) or VALIDATE_EVERY_EPOCH:

            model.train(False) # no need to track gradients for validation
            with torch.no_grad():
                running_validation_loss = 0.0
                running_raw_validation_loss = 0.0
                for j, (X_validation, y_validation) in enumerate(validation_loader):

                    X_validation = X_validation.to(DEVICE)
                    y_validation = y_validation.to(DEVICE)
                    validation_loss, raw_validation_loss = fw_pass(model, loss_fn, X_validation, y_validation)
                    running_validation_loss += validation_loss
                    running_raw_validation_loss += raw_validation_loss

            average_validation_loss = running_validation_loss.item() / (j+1) # len(validation_loader)
            average_raw_validation_loss = running_raw_validation_loss.item() / (j+1) # len(validation_loader)
            if print_loss_per_epoch:
                print(f"Loss validation {average_validation_loss:.2e}.")

            # log to tb
            tb_writer.add_scalars("Loss", { "Training" : training_loss,  "Validation" : average_validation_loss}, epoch + 1)
            tb_writer.add_scalar('Raw_Loss/validation', average_raw_validation_loss, epoch + 1)

            # track best loss model
            if average_validation_loss < best_validation_loss:
                best_validation_loss = average_validation_loss
                checkpoint_dict_best_model = {
                'model_state_dict': model.state_dict(),
                "number of inner channels": NUM_INNER_CHANNELS,
                "norm groups": NORM_GROUPS,
                "channel_mults" : CHANNEL_MULTS,
                "res_blocks" : RES_BLOCKS,
                "drop out": DROP_OUT,
                "use nd drop out": USE_ND_DROP_OUT,
                'batch_size': BATCH_SIZE,
                "patch_size": training_set.patch_size,
                'seed': SEED,
                'optimizer_state_dict': optimizer.state_dict(),
                'learning_rate': START_LEARNING_RATE,
                'best_validation_loss': best_validation_loss,
                'file_names_train': file_names_train,
                'file_names_validate': file_names_validate,
                'file_names_test': file_names_test,
                }

                if save_best_model:
                    torch.save(checkpoint_dict_best_model, f"runs/{timestamp}/checkpoint_dict_best_model.tar")

    end_training = time.time()
    training_time = end_training-start_training
    time_per_epoch = training_time/EPOCHS
    print(f"finished training after {training_time/60:.2f} min")
    print(f"time per epoch: {time_per_epoch/60:.2f} min")

    tb_writer.add_text("Best Validation Loss", f"{best_validation_loss:.2e}")

    # NOTE: to load the model use:
    # checkpoint = torch.load(f"runs/{run_name}/checkpoint.tar")
    # config = checkpoint['config]
    # agent = Agent(envs)
    # agent.load_state_dict(checkpoint['agent_state_dict'])
    # learning_rate = checkpoint['learning_rate']
    # optimizer = optim.Adam(agent.parameters(), lr = learning_rate, eps=1e-5)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # global_step = checkpoint['global_step']
    # ...
        
    # TEST BEST MODEL
    if print_status_info:
        print("test best model...")

    best_model = UNet_no_attn(
        dims = 3,
        in_channel = 1,
        out_channel = 1,
        inner_channel = NUM_INNER_CHANNELS,
        norm_groups = NORM_GROUPS, 
        channel_mults = CHANNEL_MULTS,
        res_blocks = RES_BLOCKS,
        dropout = DROP_OUT,
        use_nd_dropout = USE_ND_DROP_OUT,
    ).to(DEVICE)

    # best_model = My_UNet_3x(
    #     dims = 3,
    #     in_channel = 1,
    #     out_channel = 1,
    #     inner_channel = NUM_INNER_CHANNELS,
    #     norm_groups = NORM_GROUPS,
    #     dropout = DROP_OUT,
    #     use_nd_dropout = USE_ND_DROP_OUT,
    # ).to(DEVICE)

    best_model.load_state_dict(checkpoint_dict_best_model['model_state_dict'])

    best_model.train(False) # no need to track gradients for testing
    with torch.no_grad():
        running_test_loss = 0.0
        running_raw_test_loss = 0.0
        for j, (X_test, y_test) in enumerate(test_loader):

            # Inference
            X_test = X_test.to(DEVICE)
            y_test = y_test.to(DEVICE)
            test_loss, raw_test_loss = fw_pass(best_model, loss_fn, X_test, y_test)
            running_test_loss += test_loss
            running_raw_test_loss += raw_test_loss
            
        average_test_loss = running_test_loss.item() / (j+1) # len(test_loader)
        average_raw_test_loss = running_raw_test_loss.item() / (j+1) # len(test_loader)
        print(f"Loss testing {average_test_loss}")

    tb_writer.add_text("Test loss", f"Loss testing {average_test_loss:.2e}")
    tb_writer.add_text("Raw test loss", f"Raw loss testing {average_raw_test_loss:.2e}")

    # # Save model as ONNX file
    # model.return_mask = False
    # if print_status_info:
    #     print("saving best model as ONNX model...")
    # # save model as onnx model
    # dummy_input = torch.randn(BATCH_SIZE, 1, test_set.patch_size, test_set.patch_size, test_set.patch_size, device = DEVICE)
    # torch.onnx.export(model, dummy_input, f"runs/{timestamp}/best_model.onnx", input_names=["input"], output_names=["output"], opset_version=11)
        
    tb_writer.close()


