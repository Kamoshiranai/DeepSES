import torch
import os
import h5py
from tqdm import tqdm
import numpy as np
# from scipy.ndimage import maximum_filter, minimum_filter
from einops import rearrange
import matplotlib.pyplot as plt

from models import My_UNet_3x, UNet_no_attn
#from learn import HDF5_Dataset

test_data_vdw_path = 'sdf_data/vdw_eval'
test_data_ses_path = 'sdf_data/ses_eval'
test_data_hermosilla_path = 'sdf_data/hermosilla_eval'

output_filename = "unet_4_ch_1-2-4_mults_10_06_2025/results.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

# load model checkpoint and parameters
model_path = "unet_4_ch_1-2-4_mults_10_06_2025/checkpoint_dict_best_model.tar" #NOTE: set this to the path of the model you want to test
checkpoint_dict_model = torch.load(model_path) #, map_location=torch.device("cpu"))

NUM_INNER_CHANNELS = checkpoint_dict_model["number of inner channels"]
NUM_LAYERS = 3 # checkpoint_dict_model["number of inner channels"]
NORM_GROUPS = checkpoint_dict_model["norm groups"]
# CHANNEL_MULTS = checkpoint_dict_model["channel_mults"]
# RES_BLOCKS = checkpoint_dict_model["res_blocks"]
DROP_OUT = checkpoint_dict_model["drop out"]
USE_ND_DROP_OUT = checkpoint_dict_model["use nd drop out"]
# NO_MID_ATTENTION = checkpoint_dict_model["no mid attention"]
BATCH_SIZE = checkpoint_dict_model["batch_size"]
PATCH_SIZE = checkpoint_dict_model["patch_size"]

PROBE_RADIUS = 1.4

filenames_test = [name for name in os.listdir(test_data_vdw_path) if os.path.isfile(os.path.join(test_data_vdw_path, name))]

print("testing on files: ", filenames_test)

# set up model
# NOTE: use the type of the model you want to test

# model = UNet_no_attn(
#     dims = 3,
#     in_channel = 1,
#     out_channel = 1,
#     inner_channel = NUM_INNER_CHANNELS,
#     norm_groups = NORM_GROUPS, 
#     channel_mults = CHANNEL_MULTS,
#     res_blocks = RES_BLOCKS,
#     dropout = DROP_OUT,
#     use_nd_dropout = USE_ND_DROP_OUT,
# ).to(DEVICE)

model = My_UNet_3x(
    dims = 3,
    in_channel = 1,
    out_channel = 1,
    inner_channel = NUM_INNER_CHANNELS,
    norm_groups = NORM_GROUPS,
    dropout = DROP_OUT,
    use_nd_dropout = USE_ND_DROP_OUT,
).to(DEVICE)

model.load_state_dict(checkpoint_dict_model['model_state_dict'])

model.train(False)

with open(output_filename, 'w') as outfile:

    # --- Write the header row (optional but recommended) ---
    pre_header = f"model = {model_path}\n"
    outfile.write(pre_header)
    # Using tab separation:
    header = "FileName\tNumAtoms\tGridSize\tResolution\tHermosillaMeanError\tHermosillaSTD\tHermosillaSTDofMean\tPredictedMeanError\tPredictedSTD\tPredictedSTDofMean\n"
    outfile.write(header)

    # -------------------------------------
    # Iterate over files and compare
    # -------------------------------------

    for file_idx, filename in tqdm(enumerate(filenames_test), total=len(filenames_test), desc = "compare on molecules"):

        # --------------------------------------
        # Generate predicted ses data
        # --------------------------------------

        with h5py.File(os.path.join(test_data_vdw_path, filename), 'r') as f:
            vdw_sdf = torch.tensor(f['texture'][:], dtype = torch.float32)
            sdf_grid_res = f["texture"].attrs["grid_res"]
            sdf_numAtoms = f["texture"].attrs["numAtoms"]

        with h5py.File(os.path.join(test_data_ses_path, filename), 'r') as f:
            ses_sdf = torch.tensor(f['texture'][:], dtype = torch.float32)

        with h5py.File(os.path.join(test_data_hermosilla_path, filename), 'r') as f:
            hermosilla_sdf = torch.tensor(f['texture'][:], dtype = torch.float32)

        # array for output
        predicted_ses = torch.zeros_like(vdw_sdf)
        sdf_grid_size = vdw_sdf.shape[0]
        number_of_patches = sdf_grid_size // PATCH_SIZE # number of patches per dim

        BATCH_SIZE = 64

        model.train(False) # no need to track gradients for testing
        model.return_mask = False
        with torch.no_grad():
            # reshape tensor to batched version
            X_batched = rearrange(vdw_sdf, "(b1 h) (b2 w) (b3 d) -> (b1 b2 b3) 1 h w d", b1 = number_of_patches, b2 = number_of_patches, b3 = number_of_patches)
            predicted_ses_batched = torch.zeros_like(X_batched)

            #NOTE: Remove patches that are not relevant
            # mask = ~(X_batched == rearrange(X_batched[:, 0, 0, 0, 0], "b -> b 1 1 1 1")).all(dim = (1, 2, 3, 4))
            mask = ~torch.tensor([(X_batched[i] >= PROBE_RADIUS).all() for i in range(number_of_patches**3)], dtype = bool) # filter out patches where vdw and ses do not differ
            # print(rearrange(mask, "(b1 b2 b3) -> b1 b2 b3", b1=8, b2=8, b3=8))
            # mask = torch.tensor([True for _ in range(number_of_patches**3)])
            X_batched_filtered = X_batched[mask]

            predicted_ses_batched_filtered = torch.zeros_like(X_batched_filtered)

            print(f"run inference on {len(X_batched_filtered)} patches...")

            for batch_idx in range((len(X_batched_filtered) + BATCH_SIZE - 1) // BATCH_SIZE):
                X_test = X_batched_filtered[batch_idx * BATCH_SIZE : batch_idx * BATCH_SIZE + min(BATCH_SIZE, len(X_batched_filtered) - batch_idx * BATCH_SIZE)]

                # Inference
                X_test = X_test.to(DEVICE)

                predicted_ses_batched_filtered[batch_idx * BATCH_SIZE : batch_idx * BATCH_SIZE + min(BATCH_SIZE, len(X_batched_filtered) - batch_idx * BATCH_SIZE)] = model(X_test).detach().cpu()
        
        #NOTE: combine processed and non processed patches
        predicted_ses_batched[mask] = predicted_ses_batched_filtered
        predicted_ses_batched[~mask] = X_batched[~mask]

        # for batch_idx in range(len(X_batched)):
        #     if mask[batch_idx]:
        #         X_test = rearrange(X_batched[batch_idx], "1 h w d -> 1 1 h w d").to(DEVICE)
        #         predicted_ses_batched[batch_idx] = model(X_test).detach().cpu()
        #     else:
        #         predicted_ses_batched[batch_idx] = X_batched[batch_idx]

        # reshape predicted ses back to 512^3
        predicted_ses = rearrange(predicted_ses_batched, "(b1 b2 b3) 1 h w d -> (b1 h) (b2 w) (b3 d)", b1 = number_of_patches, b2 = number_of_patches, b3 = number_of_patches)

        # --------------------------------------
        # compare predicted SES with SES
        # --------------------------------------

        ses_mask = (ses_sdf < PROBE_RADIUS) * (ses_sdf >= 0)
        
        # Handle case where mask might be empty
        print(f"considered pred ses voxels: {torch.sum(ses_mask) / sdf_grid_size**3 * 100:.2f} %")
        if torch.sum(ses_mask) > 0:
            predicted_ses_error = torch.abs(predicted_ses[ses_mask] - ses_sdf[ses_mask])
            predicted_ses_mean_error = torch.mean(predicted_ses_error).item()
            # predicted_ses_RMSE = torch.sqrt(torch.mean(predicted_ses_error**2)).item()
            predicted_ses_std = torch.std(predicted_ses_error).item()
            predicted_ses_std_of_mean = predicted_ses_std / np.sqrt(torch.numel(predicted_ses_error))
        else:
            predicted_ses_mean_error = float('nan') # Or None, or 0, depending on how you want to handle empty masks
            # predicted_ses_RMSE = float('nan')
            predicted_ses_std = float("nan")
            predicted_ses_std_of_mean = float("nan")
            print(f"Warning: Empty mask for predicted SES in file {filename}")

        # -------------------------------------------------
        # plot errors in histograms
        # -------------------------------------------------
        plt.figure(figsize=(8, 6))
        plt.hist(predicted_ses_error, bins='auto', color='skyblue', edgecolor='black')
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        plt.title("Errors Predicted SES")
        plt.tight_layout()
        plt.savefig(f"./error_predicted_ses_{filename}.png")
        plt.close()


        # --------------------------------------
        # compare hermosilla data with SES
        # --------------------------------------

        # Handle case where mask might be empty
        print(f"considered hermosilla voxels: {torch.sum(ses_mask) / sdf_grid_size**3 * 100:.2f} %")
        if torch.sum(ses_mask) > 0:
            hermosilla_error = torch.abs(hermosilla_sdf[ses_mask] - ses_sdf[ses_mask])
            hermosilla_mean_error = torch.mean(hermosilla_error).item()
            # hermosilla_RMSE = torch.sqrt(torch.mean(hermosilla_error**2)).item()
            hermosilla_std = torch.std(hermosilla_error).item()
            hermosilla_std_of_mean = hermosilla_std / np.sqrt(torch.numel(hermosilla_error))

        else:
            hermosilla_mean_error = float('nan') # Or None, or 0
            # hermosilla_RMSE = float('nan')
            hermosilla_std = float("nan")
            hermosilla_std_of_mean = float("nan")
            print(f"Warning: Empty mask for Hermosilla in file {filename}")

        # -------------------------------------------------
        # plot errors in histograms
        # -------------------------------------------------
        plt.figure(figsize=(8, 6))
        plt.hist(hermosilla_error, bins='auto', color='skyblue', edgecolor='black')
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        plt.title("Errors Hermosilla Method")
        plt.tight_layout()
        plt.savefig(f"./error_hermosilla_{filename}.png")
        plt.close()

        print(filename)
        print(sdf_numAtoms)
        print(f"predicted SES:\n mean error = {predicted_ses_mean_error:.2e}, std = {predicted_ses_std:.2e}, std of mean = {predicted_ses_std_of_mean:.2e}")
        print(f"Hermosilla method:\n mean error = {hermosilla_mean_error:.2e}, std = {hermosilla_std:.2e}, std of mean = {hermosilla_std_of_mean:.2e}")
        
        # write results in .txt

        # --- Format the output line ---
        # Using tab separation:
        output_line = f"{filename}\t{sdf_numAtoms}\t{sdf_grid_size}\t{sdf_grid_res:.4f}\t{hermosilla_mean_error:.5f}\t{hermosilla_std:.5f}\t{hermosilla_std_of_mean:.8f}\t{predicted_ses_mean_error:.5f}\t{predicted_ses_std:.5f}\t{predicted_ses_std_of_mean:.8f}\n"
        
        # --- Write the line to the file ---
        outfile.write(output_line)

print(f"\nResults successfully written to {output_filename}")