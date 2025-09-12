import torch
from models import UNet_no_attn, My_UNet_3x

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

# load model checkpoint and parameters
checkpoint_dict_model = torch.load(f"train/unet_4_ch_1-2-4_mults_10_06_2025/checkpoint_dict_best_model.tar") #, map_location=torch.device("cpu"))

NUM_INNER_CHANNELS = checkpoint_dict_model["number of inner channels"]
CHANNEL_MULTS = checkpoint_dict_model["channel_mults"]
RES_BLOCKS = checkpoint_dict_model["res_blocks"]
NORM_GROUPS = checkpoint_dict_model["norm groups"]
DROP_OUT = checkpoint_dict_model["drop out"]
USE_ND_DROP_OUT = checkpoint_dict_model["use nd drop out"]
BATCH_SIZE = checkpoint_dict_model["batch_size"]
PATCH_SIZE = checkpoint_dict_model["patch_size"]

# set up model

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
).to(DEVICE)

# model = My_UNet_3x(
#     dims = 3,
#     in_channel = 1,
#     out_channel = 1,
#     inner_channel = NUM_INNER_CHANNELS,
#     norm_groups = NORM_GROUPS,
#     dropout = DROP_OUT,
#     use_nd_dropout = USE_ND_DROP_OUT,
# ).to(DEVICE)

model.load_state_dict(checkpoint_dict_model['model_state_dict'])

model.train(False)
# model.return_mask = False

# save model as onnx model
#NOTE: if batch size is fixed, just set it here and remove dynamic axes in export below
dummy_input = torch.randn(1, 1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
torch.onnx.export(
    model, 
    dummy_input, 
    f"train/unet_4_ch_1-2-4_mults_10_06_2025/model_batch_size=dynamic.onnx", 
    input_names=["input"], 
    output_names=["output"],
    opset_version=20,
    export_params=True,
    do_constant_folding=True,
    dynamic_axes={
        "input": {0: 'batch_size'},  # Mark dimension 0 of 'input' as dynamic
        "output": {0: 'batch_size'} # Mark dimension 0 of 'output' as dynamic
                                       # Add entries for all inputs/outputs that have batch dim
    }
)