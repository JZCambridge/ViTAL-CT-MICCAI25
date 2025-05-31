import sys
import os
import csv
import copy
import logging
import argparse
from datetime import datetime
import time
from cgi import test
import math
import random
import pandas as pd

# Add the parent directory of src to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.config as config

seed = config.RANDOM_STATE
random.seed(seed)

# Torch and related libraries
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR


# Vision libraries
from torchvision import models
import timm
from tqdm import tqdm
import numpy as np
np.random.seed(seed)

# Custom utilities
from utils.lora import LoRA_ViT, LoRA_ViT_timm, LoRA_ViT_timm_posembed
from utils.dataloader import Dataloader_CTA, Dataloader_CTA_CV
from utils.result import ResultCLS, ResultMLS
from utils.utils import ckpt_path_update, get_incorrect_details, save_incorrect_files_details #, check_cfg, init, save, 
from utils.loss import CRPSLoss, Ca3Loss, CoronaryFocalLoss
from utils.benchmark import pneumonia_data, pneumonia_data_CV

# model
from model.vit_small_simplify import VisionTransformer
from model.res_vit import Res_ViT

weightInfo = {
    "vit_small_p16_224": "hf-hub:timm/vit_small_patch16_224.augreg_in21k_ft_in1k", #Params (M): 22.1
    "vit_base_p16_224": "timm/vit_base_patch16_224.augreg_in21k_ft_in1k", #Params (M): 86.6
    "vit_large_p16_224": 'timm/vit_large_patch16_224.augreg_in21k_ft_in1k' #Params (M): 304.3
}
previous_dir_name = None


def extractBackbone(state_dict, prefix: str = None) -> dict:
    """
    Extracts the backbone from a state dictionary by removing keys that start with the specified prefix.
    If no prefix is provided, it removes keys related to the fully connected (fc) layer.

    Args:
        state_dict (dict): The state dictionary from which to extract the backbone.
        prefix (str): The prefix to look for when extracting the backbone. If None, it defaults to removing 'fc' keys.

    Returns:
        dict: The modified state dictionary with the backbone extracted.
    """
    # If prefix is None, remove keys starting with 'fc'
    if prefix is None:
        for k in list(state_dict.keys()):
            if k.startswith("fc"):
                del state_dict[k]
        return state_dict

    # Create a new state_dict to avoid modifying the original one
    new_state_dict = {}

    # Extract backbone keys and remove the prefix
    for k in list(state_dict.keys()):
        if k.startswith(f"{prefix}.") and not k.startswith(f"{prefix}.fc"):
            # Remove prefix and add to new state_dict
            new_key = k[len(f"{prefix}."):]
            new_state_dict[new_key] = state_dict[k]

    return new_state_dict

def debug_bce_loss(pred, label, pos_weight, loss, total_loss=False):
    """Function to debug BCE loss calculation."""
    if pos_weight is None:
        pos_weight = torch.tensor([1.0] * label.size(1))
    pos_weight = pos_weight.to(device)
    pred_sig = torch.sigmoid(pred)
    total_loss = 0

    if total_loss:
        for i in range(len(pred)):
            bce_loss = pos_weight * label[i] * torch.log(pred_sig[i]) + (1 - label[i]) * torch.log(1 - pred_sig[i])
            bce_loss = -torch.mean(bce_loss)
            total_loss += bce_loss
        total_loss /= len(pred)
        print(f"total_loss: {total_loss}")
        tolerance = 1e-3
        assert math.isclose(total_loss, loss, rel_tol=tolerance), "total_loss and loss are not equal"
        return None
    else:
        bce_loss = pos_weight * label * torch.log(pred_sig) + (1 - label) * torch.log(1 - pred_sig)
        return bce_loss

def get_loss_function(loss_func_type, weights, device, ignore_index, num_classes=2):
    """Initialize the appropriate loss function based on the given type."""
    if loss_func_type == 'bce':
        return nn.BCEWithLogitsLoss().to(device) if num_classes == 1 else nn.BCEWithLogitsLoss(pos_weight=weights).to(device)
    elif loss_func_type == 'ce':
        if ignore_index == -100:
            return nn.CrossEntropyLoss(weight=weights).to(device)
        else:
            return nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weights).to(device)
    elif loss_func_type == 'crps':
        return CRPSLoss().to(device)
    elif loss_func_type == 'ca3':
        return Ca3Loss().to(device)
    elif loss_func_type == 'focal':
        return CoronaryFocalLoss(alpha=0.75, gamma=2.5, num_classes=num_classes).to(device)
    else:
        raise ValueError("Invalid loss function type")

def save_checkpoint(ckpt_path, epoch, net, optimizer, best_val_result):
    """Save the model checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val": best_val_result,
        },
        ckpt_path,
    )

def log_auc_results(result, hrpname):
    """Log the AUC results."""
    if result.test_mls_auc:
        title = "|"
        message = "|"
        for idx, auc in enumerate(result.test_mls_auc):
            title += f"{hrpname[idx]:^20}|"
            message += f"{auc:^20.4f}|"
        logging.info(title)
        logging.info(message)

def train(epoch: int, trainset, loss_type='bce', pos_weight=None, debug=False, num_classes=2) -> None:
    """
    Train the model for one epoch.

    Args:
        epoch (int): The current epoch number.
        trainset (DataLoader): The DataLoader for the training data.
        loss_type (str): The type of loss function to use ('bce', 'ce', 'ca3').
        pos_weight (torch.Tensor): The positive weight for BCE loss. Default is None.
        debug (bool): If True, enables debugging output. Default is False.

    Returns:
        None

    Example setup (ensure these are properly defined in your actual code)
        net = ...
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        loss_func = nn.CrossEntropyLoss()
        scaler = GradScaler()
        logging.basicConfig(level=logging.INFO)
    """
    # debugging setup
    if loss_type == 'bce' and num_classes == 1:
        # Adjust pos_weight for binary classification with a single output
        if pos_weight is not None:
            # print(f"Original Pos_weight: {pos_weight}")  # Debugging
            pos_weight = pos_weight[1].to(device)  # Use the weight for class 1 only
            # print(f"Adjusted Pos_weight: {pos_weight}")  # Should be a scalar
    else:
        if pos_weight is not None:
            pos_weight = pos_weight.to(device)

    running_loss = 0.0
    this_lr = optimizer.param_groups[0]['lr']

    net.train()
    result.init()
    
    if epoch == cfg.freezing_epoch:
        # for param in net.module.unet_proj.parameters():
        #     param.requires_grad = False
        #for param in net.module.unet_attn.parameters():
        #    param.requires_grad = False
        for param in net.module.residual.parameters():
            param.requires_grad = False

    for image, label, _, position, unetembed in tqdm(trainset, ncols=60, desc="train", unit="batch", leave=None):
        image, label, position, unetembed = image.to(device), label.to(device), position.to(device), unetembed.to(device)
        optimizer.zero_grad()

        # Use autocast for mixed precision training
        with autocast(enabled=True):
            pred = net(image,position,unetembed).to(device)
            # print(f'label.shape: {label.shape}')
            # print(f'pred.shape: {pred.shape}')
            
            # Calculate loss
            if loss_type == 'bce':
                # Adjust labels for binary classification
                if num_classes == 1:
                    label = label[:, 1]  # Use the second column as the target for class 1
                    label = label.float().unsqueeze(1).to(device)  # Reshape to (batch_size, 1)
                loss = loss_func(pred, label)
                if debug:
                    debug_bce_loss(pred, label, pos_weight, loss)
            elif loss_type == 'focal':
                # Adjust labels for binary classification
                if num_classes == 1:
                    label = label[:, 1]  # Use the second column as the target for class 1
                    label = label.float().unsqueeze(1).to(device)  # Reshape to (batch_size, 1)
                loss = loss_func(pred, label)

            elif loss_type == 'ce':
                label = ResultMLS.extract_indices_ca(array=label)
                loss = loss_func(pred, label)
            elif loss_type == 'ca3':
                loss = loss_func(pred, label)
            else:
                logging.error(f"Unknown loss function: {loss_type}")
                exit(1)

            result.eval(label, pred)
        
        scaler.scale(loss).backward()
        
        '''upated to use gradient clipping'''
        # === GRADIENT CLIPPING ADDITION ===
        # 1. Unscale gradients before clipping
        scaler.unscale_(optimizer)
        
        # 2. Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(
            net.parameters(),
            max_norm=1.0,  # Critical for CTA stability
            norm_type=2     # L2 norm clipping
        )
        # ==================================

        # Update weights
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        running_loss += loss.item()
    
    # === SCHEDULER INTEGRATION ===
    # Update learning rate using OneCycle policy
    scheduler.step() 
    
    # Log learning rate after update
    this_lr = optimizer.param_groups[0]['lr']
    '''upated to use gradient clipping'''

    # scheduler.step()

    epoch_loss = running_loss / len(trainset)
    logging.info(f"\n\nTraining EPOCH: {epoch}, LOSS: {epoch_loss:.3f}, LR: {this_lr:.2e}")
    result.print(epoch, 'train', debug=False)

    return epoch_loss

@torch.no_grad()
def eval(epoch: int, testset, datatype: str = "val", criteria='loss', save_incorrect_files=False, log_save_path=None, pos_weight=None, debug=False, update=True, loss_type=None, num_classes=2) -> None:
    """
    Evaluate the model on a test dataset.

    Args:
        epoch (int): The current epoch number.
        testset (DataLoader): The DataLoader for the test data.
        datatype (str): The type of data being evaluated (default is "val").
        criteria (str): The criteria for evaluation (default is 'loss').
        save_incorrect_files (bool): If True, saves incorrect files' paths to a CSV (default is False).
        pos_weight (torch.Tensor): The positive weight for BCE loss. Default is None.
        debug (bool): If True, enables debugging output. Default is False.

    Returns:
        None

    """
    global previous_dir_name
    running_loss = 0.0
    incorrect_details = []
    raw_preds = []
    raw_losses = []
    raw_labels = []

    # Ensure the log_save_path directory exists
    if log_save_path and not os.path.exists(log_save_path):
        os.makedirs(log_save_path)

    result.init()
    net.eval()

    for image, label, file_path, position, unetembed in tqdm(testset, ncols=60, desc=datatype, unit="batch", leave=None):
        image, label, position, unetembed = image.to(device), label.to(device), position.to(device), unetembed.to(device)

        with autocast(enabled=True):
            pred = net(image, position, unetembed)

            if (loss_type == 'bce' or loss_type == 'focal') and num_classes == 1:
                # Adjust labels for binary classification
                label = label[:, 1]  # Use the second column as the target for class 1
                label = label.float().unsqueeze(1).to(device)  # Reshape to (batch_size, 1)

            result.eval(label, pred)

            # Calculate loss
            loss = loss_func(pred, label)
            
            if debug:
                losses = debug_bce_loss(pred, label, pos_weight, loss)
                raw_preds.extend(pred.detach().cpu().numpy())
                raw_losses.extend(losses.detach().cpu().numpy())
                raw_labels.extend(label.detach().cpu().numpy())

            # Save incorrect predictions
            if save_incorrect_files:
                incorrect_details.extend(get_incorrect_details(pred, label, file_path))

        # Accumulate loss
        running_loss += loss.item()

    # Calculate the loss for the entire dataset
    epoch_loss = running_loss / len(testset)

    result.print(epoch, datatype, criteria=criteria, metrics=epoch_loss, debug=False, update=update) # debug=True to print all class results

    if debug:
        # Paths for saving files
        raw_preds_path = os.path.join(log_save_path, "raw_preds.csv")
        raw_losses_path = os.path.join(log_save_path, "raw_losses.csv")
        raw_labels_path = os.path.join(log_save_path, "raw_labels.csv")

        # Convert current epoch data to DataFrame
        current_preds = pd.DataFrame({f"epoch_{epoch}_preds": raw_preds})
        current_losses = pd.DataFrame({f"epoch_{epoch}_losses": raw_losses})
        current_labels = pd.DataFrame({f"epoch_{epoch}_labels": raw_labels})

        # Function to append along columns
        def append_columns(file_path, new_data):
            if os.path.exists(file_path):
                existing_data = pd.read_csv(file_path)
                updated_data = pd.concat([existing_data, new_data], axis=1)
            else:
                updated_data = new_data
            updated_data.to_csv(file_path, index=False)

        # Append along columns
        append_columns(raw_preds_path, current_preds)
        append_columns(raw_losses_path, current_losses)
        append_columns(raw_labels_path, current_labels)

    if save_incorrect_files:
        previous_dir_name = save_incorrect_files_details(epoch, incorrect_details, previous_dir_name, log_save_path)

    return epoch_loss

if __name__ == "__main__":
    # Command-Line Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-freezing_epoch", type=int, default=40, help="epoch for freezing resudual module")
    parser.add_argument("-save_checkpoint_epoch", type=int, default=1000, help="epoch for saving checkpoints")
    parser.add_argument("-event_name", type=str, default='conv_vit_f0_seed420_saveweight', help="name of the experiment")
    parser.add_argument("-vit_size", type=str, default='base', help="size of the ViT model: base, small, tiny, large")
    parser.add_argument("-bs", type=int, default=32, help="Batch size for training")
    parser.add_argument("-res_bs", type=int, default=32*196, help="Batch size for resnet training")
    parser.add_argument("-fold", type=int, default=0, help="Fold number for cross-validation")
    parser.add_argument("-data_path", type=str, default='/media/jay/Storage21/ScotHeart_Aug_ResVit', help="Path to the data directory, nifiti files: /media/jay/Storage2/ScotHeart/GoodToshibaXsections/, /media/jay/Storage2/ScotHeart_Clean/all/")
    parser.add_argument("-output_path", type=str, default='/home/tan_gong/Result/', help="Path to the output directory/")
    parser.add_argument("-data_info", type=str, default="HRP2", help="Data information type (e.g., HRP5, Ca3)")
    parser.add_argument("-annotation", type=str, default="/home/jay/Documents/VIT_CTA/Vit_CT_Repository/0_Raw_Data/Toshiba_HRP.csv", help="Path to the annotation file, evnets:/home/jay/Documents/VIT_CTA/Vit_CT_Repository/0_Raw_Data/Toshiba_HRP_Events2018.csv")
    parser.add_argument("-lr", type=float, default=1e-4, help="Learning rate") # TODO: fixed learning rate
    parser.add_argument("-epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("-num_workers", type=int, default=3, help="Number of data loading workers")
    parser.add_argument("-num_classes", "-nc", type=int, default=1, help="Number of classes")
    parser.add_argument("-backbone", type=str, default="vit_base_p16_224", help="Model backbone type")
    parser.add_argument("-train_type", "-tt", type=str, default="resvit", help="Training type: lora, full, resnet, net, resvit, res_check")
    parser.add_argument("-rank", "-r", type=int, default=64, help="Rank for LoRA: Vit_base for 64") 
    parser.add_argument("-alpha", "-a", type=int, default=64, help="Alpha for LoRA: Vit_base for 64")
    parser.add_argument("-loss_func", type=str, default='bce', help="Loss function type: bce, ce, ca3, crps, focal") #'''try new loss'''
    parser.add_argument("-ignore_index", type=int, default=2, help="Ignore index for cross-entropy loss")
    parser.add_argument("-weights_path", type=str, default=None, help="Path to trained weights file") # TODO: Changed path
    parser.add_argument("-init_weights_path", type=str, default='./weights/B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz') #base model
    parser.add_argument("-performance_measure", type=str, default='loss', help="Performance measure: loss, auc")
    parser.add_argument("-debug", type=bool, default=False, help="Debug mode")
    parser.add_argument("-log_table_head", type=str, nargs='+', default=['posremod', 'lowatten', 'napkin', 'spotty', 'punctate'])
    parser.add_argument("-split", type=str, default="none", help="stratified, random, none")
    parser.add_argument("-balance", type=bool, default=True, help="Stratified sampling")
    parser.add_argument("-debug_size", type=int, default=0, help="Debug size")
    parser.add_argument("-check_incorrect", type=bool, default=False, help="Check all files and save incorrect files")
    parser.add_argument("-benchmark", type=bool, default=False, help="Benchmark a know dataset")
    parser.add_argument("-inbalance", type=str, default='balanced', help='data inbalance dealing method: raw, weighted, balanced')
    parser.add_argument("-random_settings", type=dict, default={'normal_augmentation':False, 'torch_augmentation':False, 
                                        'column_flip':False, 'column_flip_settings':{'random_flip': True, 'flip_height': True, 'flip_width': True, 'flip_diagonal': True, 'flip_antidiagonal': True}, # Flip columns, False to disable
                                        'column_rotate':False, 'angle_range':90, # column-wise rotation, False to disable
                                        'column_shift':False, 'random_shift':True, 'shift_range':2, # Shift columns, False to disable
                                        'frame_swing': False, 'frame_swing_range': 4, # slice-wise shift, 0 to disable
                                        'white_noise_2D':False, 'noise_scale':40, # slice-wise noise, False to disable
                                        'gaussian_filter_2D':False, 'gaussian_random':True, 'sigma':1.2, # slice-wise gaussian, False to disabl
                                        }, help="Data augmentation")

    cfg = parser.parse_args()
    #TODO: Implement this function!!!
    # cfg = check_cfg(cfg)
    
    # Initialization
    scaler = GradScaler()

    # Logging
    if cfg.benchmark: # TODO: change the path for benchmark
        log_save_path = os.path.join(os.path.join(cfg.output_path, "Benchmark_Run"), f"{datetime.now().strftime('%m%d_%H%M')}_{random.randint(10, 99)}_{cfg.train_type}seed{seed}_fold{cfg.fold}_{cfg.inbalance}/")
    else:
        balance_ref = 'weighted'
        if cfg.balance: balance_ref = 'balanced'
        log_save_path = os.path.join(os.path.join(cfg.output_path, cfg.event_name), f"{datetime.now().strftime('%m%d_%H%M')}_{random.randint(10, 99)}_{cfg.train_type}seed{seed}_fold{cfg.fold}_{balance_ref}/")
    os.makedirs(log_save_path, exist_ok=True)
    print(f"Log save path: {log_save_path}")

    # Set up logging
    log_file_path = os.path.join(log_save_path, "log.log")
    logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s-%(levelname)s] %(message)s",
                    datefmt="%y-%m-%d %H:%M:%S",
                    handlers=[
                        logging.StreamHandler(),             # Console output
                        logging.FileHandler(log_file_path)   # File output
                    ])
    # datafmt = init(taskfolder=log_save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(cfg)
    
    
    # Model Setup
    if cfg.train_type == "lora":
        model = timm.create_model(weightInfo[cfg.backbone], pretrained=True)
        net = LoRA_ViT_timm(model, r=cfg.rank, alpha=cfg.alpha, num_classes=cfg.num_classes).to(device)
        num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        logging.info(f"Trainable parameters: {num_params / 2**20:.4f}M")
    elif cfg.train_type == "full":
        net = timm.create_model(weightInfo[cfg.backbone], pretrained=True, num_classes=cfg.num_classes).to(device)
    elif cfg.train_type == "resnet":
        net = timm.create_model('resnet50', pretrained=True, num_classes=cfg.num_classes).to(device) 
    elif cfg.train_type == "net":
        if cfg.vit_size == 'small':
            net = VisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6, num_classes=cfg.num_classes) # TODO: need to check whether this needed to be changed
        elif cfg.vit_size == 'tiny':
            net = VisionTransformer(patch_size=16, embed_dim=192, depth=12, num_heads=3, num_classes=cfg.num_classes)  # TODO: need to check whether this needed to be changed
        elif cfg.vit_size == 'base':
            net = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=cfg.num_classes)
        elif cfg.vit_size == 'large':
            net = VisionTransformer(patch_size=16, embed_dim=1024, depth=24, num_heads=16, num_classes=cfg.num_classes)

        if os.path.exists(cfg.init_weights_path):
            net.load_pretrained(cfg.init_weights_path)
        else:
            logging.error(f"Weight path Wrong: {cfg.init_weights_path} \nNo weight loaded, training from scratch!")
        
        net = LoRA_ViT_timm(net, r=cfg.rank, alpha=cfg.alpha, num_classes=cfg.num_classes).to(device)
    
    elif cfg.train_type == "resvit":
        if cfg.vit_size == 'small':
            net = Res_ViT(patch_size=16, embed_dim=384, depth=12, num_heads=6, res_bs = cfg.res_bs, num_classes=cfg.num_classes) # TODO: need to check whether this needed to be changed
        elif cfg.vit_size == 'tiny':
            net = Res_ViT(patch_size=16, embed_dim=192, depth=12, num_heads=3, res_bs = cfg.res_bs, num_classes=cfg.num_classes)  # TODO: need to check whether this needed to be changed
        elif cfg.vit_size == 'base':
            net = Res_ViT(patch_size=16, embed_dim=768, depth=12, num_heads=12, res_bs = cfg.res_bs, num_classes=cfg.num_classes)
        elif cfg.vit_size == 'large':
            net = Res_ViT(patch_size=16, embed_dim=1024, depth=24, num_heads=16, res_bs = cfg.res_bs, num_classes=cfg.num_classes)

        if os.path.exists(cfg.init_weights_path):
            net.vit.load_pretrained(cfg.init_weights_path)
        else:
            logging.error(f"Weight path Wrong: {cfg.init_weights_path} \nNo weight loaded, training from scratch!")
        
        if cfg.rank != None and cfg.alpha != None:
            print("LoRA")
            net.vit = LoRA_ViT_timm_posembed(net.vit, r=cfg.rank, alpha=cfg.alpha, num_classes=cfg.num_classes).to(device)
        else:
            print("No LoRA")
    
    elif cfg.train_type == "res_check":
        from model.resnet_check import ResNetStandalone
        net = ResNetStandalone(num_classes=cfg.num_classes, res_bs=cfg.res_bs).to(device)

    else:
        logging.error("Invalid training type")
        exit(1)
    
    net = torch.nn.DataParallel(net)
    
    # Load Pre-trained Weights
    '''
    if cfg.init_weights_path and os.path.isfile(cfg.init_weights_path):
        checkpoint = torch.load(cfg.init_weights_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith('module.lora_vit.head') or k.startswith('module.proj_3d'))}
        net.load_state_dict(state_dict, strict=False)
        logging.info(f"Loaded weights from {cfg.init_weights_path}")
    else:
        logging.error(f"No weight loaded, training from scratch!")
    '''
    # Cross-validation
    if cfg.fold == 0:
        # Data Loading
        if cfg.benchmark:
            trainset, valset, testset, weights = pneumonia_data('/home/jay/Documents/chest_pneumonia_xray/chest_xray', val_per=0.2, batch=32, 
                                                                date_setting=cfg.inbalance) # TODO: date_setting = 'raw', 'weighted', 'balanced'!! 
        else:
            trainset, valset, testset, weights = Dataloader_CTA(
                cfg, 
                filepath=cfg.data_path, 
                table_path=cfg.annotation,
                label=cfg.data_info, 
                train_proportion=0.6,
                val_proportion=0.2,
                tensor_output=True,
                slices_thres_dict={'RCA': 20, 'LAD': 20, 'LCX': 20},
                balance=cfg.balance,
                split=cfg.split,
                debug=False,
                input_data_type='tensor_combined', # tensor_combined, tensor_aug, tensor
                debug_size=0
            )
        logging.info(f'Training set size: {len(trainset)}')
        logging.info(f'Validation set size: {len(valset)}')
        logging.info(f'Test set size: {len(testset)}')
        logging.info(f'Class weights: {weights}')
        print(f'fold: {cfg.fold}')
        print(f'Training set size: {len(trainset)}')
        print(f'Validation set size: {len(valset)}')
        print(f'Test set size: {len(testset)}')
        print(f'Class weights: {weights}')
        
        # Loss Function Setup
        loss_func = get_loss_function(loss_func_type=cfg.loss_func, weights=weights, device=device, ignore_index=cfg.ignore_index, num_classes=cfg.num_classes)
        
        # records
        # Prepare the CSV file
        csv_file_path = os.path.join(log_save_path, 'result.csv')

        # If the CSV file exists, delete it
        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)

        # Create a new CSV file with the header
        result_df = pd.DataFrame(columns=['epoch', 'train', 'val', 'test'])
        result_df.to_csv(csv_file_path, index=False)

        '''upated to use new adam and scheduler'''
        # Optimizer, Scheduler and Result Handler 
        # optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
        optimizer = optim.AdamW(
                    net.parameters(),
                    lr=cfg.lr,          # ↑ from 1e-6
                    weight_decay=0.05,  # Regularization
                    betas=(0.9, 0.999),  # Adam's default values
                    )
        # scheduler = CosineAnnealingLR(optimizer, cfg.epochs, 1e-6)
        # Replace current scheduler initialization
        total_steps = cfg.epochs * len(trainset)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,          # Peak learning rate (1e-4 recommended)
            total_steps=total_steps,
            pct_start=0.1,          # 10% of steps for warmup
            anneal_strategy='linear'
        )
        '''upated to use new adam and scheduler'''
        result = ResultMLS(cfg.num_classes, auc_calc=True, class_prediction=cfg.data_info, debug=False)
        
        # Training and Evaluation Loop
        for epoch in range(1, cfg.epochs + 1):
            # Train
            train_result = train(epoch, trainset, loss_type=cfg.loss_func, pos_weight=weights, debug=cfg.debug, num_classes=cfg.num_classes)
            
            if epoch == 1 and cfg.performance_measure == 'loss':
                result.best_val_result = float('inf')

            # Validate
            val_result = eval(epoch, valset, datatype="val", criteria=cfg.performance_measure, pos_weight=weights, debug=cfg.debug, update=True, log_save_path=log_save_path, loss_type=cfg.loss_func, num_classes=cfg.num_classes)
            
            # Test and Save
            if result.best_epoch == result.epoch:
                # save all best epoch TODO: disabled for now
                # ckpt_path = ckpt_path_update(taskfolder=log_save_path, epoch=epoch)
                # save_checkpoint(ckpt_path, epoch, net, optimizer, result.best_val_result)
                test_result = eval(epoch, testset, datatype="test", criteria=cfg.performance_measure, pos_weight=weights, loss_type=cfg.loss_func, num_classes=cfg.num_classes)
                logging.info(f"BEST {cfg.performance_measure.upper()} VAL: {result.best_val_result:.6f}, TEST: {result.test_result:.6f}, EPOCH: {result.best_epoch:3}")
                log_auc_results(result, cfg.log_table_head)

                # Check all files and save incorrect files
                if cfg.check_incorrect:
                    combined_dataset = ConcatDataset([trainset.dataset, valset.dataset, testset.dataset])
                    full_dataloader = DataLoader(combined_dataset, batch_size=cfg.bs, shuffle=True, num_workers=cfg.num_workers)
                    eval(epoch, full_dataloader, datatype="Checking incorrect files", save_incorrect_files=True, log_save_path=log_save_path,pos_weight=weights, loss_type=cfg.loss_func, num_classes=cfg.num_classes)
            else:
                test_result = None

            # Test
            if epoch % save_checkpoint_epoch == 0:
                ckpt_path = ckpt_path_update(taskfolder=log_save_path, epoch=epoch)
                save_checkpoint(ckpt_path, epoch, net, optimizer, result.best_val_result)
            test_result = eval(epoch, testset, datatype="test", criteria=cfg.performance_measure, pos_weight=weights, loss_type=cfg.loss_func, num_classes=cfg.num_classes)
            logging.info(f"RESULT {cfg.performance_measure.upper()} VAL: {result.best_val_result:.6f}, TEST: {result.test_result:.6f}, EPOCH: {result.best_epoch:3}")
            log_auc_results(result, cfg.log_table_head)
            
            # Append results to CSV file
            epoch_result = {
                'epoch': epoch,
                'train': train_result,
                'val': val_result,
                'test': test_result if test_result is not None else 0
            }
            
            # Convert epoch result to DataFrame and append it to CSV
            epoch_df = pd.DataFrame([epoch_result])
            epoch_df.to_csv(csv_file_path, mode='a', header=False, index=False)

    else: 
        # TODO: corss validaiton in each epoch will lead to information leakage
        # later need to implement epoch inside the cross-validation

        # Data Loading for Cross-Validation & stratified sampling
        if cfg.benchmark:
            trainsets, valsets, testset, weights = pneumonia_data_CV('/home/jay/Documents/chest_pneumonia_xray/chest_xray', val_per=0.2, batch=32, 
                                                                    data_setting=cfg.inbalance) # TODO: change balance!!
        else:
            trainsets, valsets, testset, weights = Dataloader_CTA_CV(
                cfg, 
                filepath=cfg.data_path, 
                table_path=cfg.annotation,
                label=cfg.data_info, 
                external_proportion=0.2, 
                fold=cfg.fold,
                tensor_output=True,
                slices_thres_dict={'RCA': 20, 'LAD': 20, 'LCX': 20},
                balance=cfg.balance,
                split=cfg.split,
                debug=cfg.debug,
                input_data_type = 'tensor'
            )

        for i in range(len(trainsets)):
            tran = trainsets[i]
            val = valsets[i]
            print(i)
            print(f'Training set size: {len(tran)}')
            print(f'Validation set size: {len(val)}')

        # records
        # Prepare the CSV file
        csv_file_path = os.path.join(log_save_path, 'result.csv')

        # If the CSV file exists, delete it
        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)

        # Create a new CSV file with the header
        column_names = ['epoch', 'train_mean', 'val_mean', 'test', 'train_mean_fold', 'val_mean_fold']
        for i in range(cfg.fold):
            column_names.append(f'val_fold_{i}')
        for i in range(cfg.fold):
            column_names.append(f'train_fold_{i}')
        result_df = pd.DataFrame(columns=column_names)
        result_df.to_csv(csv_file_path, index=False)

        # Loss Function Setup
        print(f'weights: {weights}')
        loss_func = get_loss_function(loss_func_type=cfg.loss_func, weights=weights[0], device=device, ignore_index=cfg.ignore_index, num_classes=cfg.num_classes) # 1st weights are used for all folds
        
        # Optimizer, Scheduler and Result Handler
        optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
        scheduler = CosineAnnealingLR(optimizer, cfg.epochs, 1e-6)
        result = ResultMLS(cfg.num_classes, auc_calc=True, class_prediction=cfg.data_info, debug=False)

        # Initialize the best model weights & loss
        # best_model_weights = copy.deepcopy(net.state_dict())
        if cfg.performance_measure == 'loss': result.best_val_result = float('inf')

        # Training and Evaluation Loop
        for epoch in range(1, cfg.epochs + 1):
            # each fold performance
            fold_train_metrics = [None] * cfg.fold
            fold_metrics = [None] * cfg.fold
            fold_nets = [None] * cfg.fold

            for fold in range(cfg.fold):
                start_time = time.time()
                trainset = trainsets[fold]
                valset = valsets[fold]

                print(trainset)
                print(f'Training set size: {len(trainset)}')
                print(f'Validation set size: {len(valset)}')

                # # load the best model weights
                # net.load_state_dict(best_model_weights)

                # Train
                train_metrics = train(epoch, trainset, loss_type=cfg.loss_func, pos_weight=weights[fold], debug=cfg.debug) # , pos_weight=weights
                
                # Validate
                out_metrics = eval(epoch, valset, datatype="val", criteria=cfg.performance_measure, pos_weight=weights[fold], debug=cfg.debug, update=False) # , pos_weight=weights

                # Keep all fold results
                fold_train_metrics[fold] = train_metrics
                fold_metrics[fold] = out_metrics
                fold_nets[fold] = copy.deepcopy(net.state_dict())

                # logging
                fold_time = time.time() - start_time
                fold_time_hms = time.strftime("%H:%M:%S", time.gmtime(fold_time))
                logging.info(f'Fold {fold+1} {cfg.performance_measure.upper()} VAL: {out_metrics:.4f}; Time: {fold_time_hms}\n')

            # Selection of the best fold
            # each fold metrics mean # can be maximum/minimum!
            out_metrics_mean = np.mean(fold_metrics)
            out_train_metrics_mean = np.mean(fold_train_metrics)

            # find fold closest to mean
            fold_choice = np.argmin(np.abs(fold_metrics - out_metrics_mean))
            
            # load the best model weights
            net.load_state_dict(fold_nets[fold_choice])

            # Update results
            result.update(epoch, datatype="val", criteria=cfg.performance_measure, metrics=fold_metrics[fold_choice])

            # logging
            epoch_time = time.time() - start_time
            epoch_time_hms = time.strftime("%H:%M:%S", time.gmtime(epoch_time))
            logging.info(f"Epoch {epoch}: CV {cfg.performance_measure.upper()} Mean VAL: {fold_metrics[fold_choice]:.4f}; Time: {epoch_time_hms}\n")

            # Save the best model weights
            if result.best_epoch == result.epoch:
                # save all best epoch
                ckpt_path = ckpt_path_update(taskfolder=log_save_path, epoch=epoch)
                save_checkpoint(ckpt_path, epoch, net, optimizer, result.best_val_result)
                test_result = eval(epoch, testset, datatype="test", criteria=cfg.performance_measure)#, pos_weight=weights)
                logging.info(f"BEST {cfg.performance_measure.upper()} VAL: {result.best_val_result:.6f}, TEST: {result.test_result:.6f}, EPOCH: {result.best_epoch:3}")
                log_auc_results(result, cfg.log_table_head)

                # Check all files and save incorrect files
                if cfg.check_incorrect:
                    combined_dataset = ConcatDataset([trainset.dataset, valset.dataset, testset.dataset])
                    full_dataloader = DataLoader(combined_dataset, batch_size=cfg.bs, shuffle=True, num_workers=cfg.num_workers)
                    eval(epoch, full_dataloader, datatype="Checking incorrect files", save_incorrect_files=True, log_save_path=log_save_path)#, pos_weight=weights)
            else:
                test_result = None
        
            # Append results to CSV file
            epoch_result = {
                'epoch': epoch,
                'train_mean': out_train_metrics_mean,
                'val_mean': out_metrics_mean,
                'test': test_result if test_result is not None else 0,
                'train_mean_fold': fold_train_metrics[fold_choice],
                'val_mean_fold': fold_metrics[fold_choice]
            }
            for i in range(cfg.fold):
                epoch_result[f'val_fold_{i}'] = fold_metrics[i]
            for i in range(cfg.fold):
                epoch_result[f'train_fold_{i}'] = fold_train_metrics[i]
            
            # Convert epoch result to DataFrame and append it to CSV
            epoch_df = pd.DataFrame([epoch_result])
            epoch_df.to_csv(csv_file_path, mode='a', header=False, index=False)
            

