import logging
import os
import time
import csv
from datetime import datetime

import numpy as np
import torch
import torch as tc
from scipy.ndimage import rotate, shift, gaussian_filter

def initLogging(logFilename):
    """Init for logging"""
    logger = logging.getLogger("")

    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s-%(levelname)s] %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
            filename=logFilename,
            filemode="w",
        )
        console = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s-%(levelname)s] %(message)s")
        console.setFormatter(formatter)
        logger.addHandler(console)

def init(taskfolder = f"./results/", epoch = 0):

    if not os.path.exists(taskfolder):
        os.makedirs(taskfolder)
    datafmt = time.strftime("%Y%m%d_%H%M%S")

    log_dir = f"{taskfolder}/{datafmt}.log"
    initLogging(log_dir)
    ckpt_path = f"{taskfolder}/{datafmt}.pt"
    return ckpt_path

def ckpt_path_update(taskfolder = f"./results/", epoch = 0):
    if not os.path.exists(taskfolder):
        os.makedirs(taskfolder)
    datafmt = time.strftime("%Y%m%d_%H%M%S")

    ckpt_path = f"{taskfolder}/{datafmt}_ep{epoch}.pt"
    return ckpt_path


def save(result, net, ckpt_path):
    # Save best model
    
    torch.save(net.state_dict(), ckpt_path.replace(".pt", "_last.pt"))

    logging.info(f"BEST : {result.best_result:.3f}, EPOCH: {(result.best_epoch):3}")
    return

def get_incorrect_details(pred, label, file_path):
    """Function to get details of incorrect predictions."""
    incorrect_details = []
    pred_labels = torch.argmax(pred[:, :-1], dim=1)
    label_idx = torch.argmax(label[:, :-1], dim=1)
    incorrect_indices = pred_labels != label_idx
    for i in incorrect_indices.nonzero(as_tuple=True)[0]:
        incorrect_details.append((file_path[i], label_idx[i].item(), pred_labels[i].item()))
    return incorrect_details

def save_incorrect_files_details(epoch, incorrect_details, previous_dir_name=None, log_save_path=None):
    """Function to save incorrect files' details to a CSV file."""
    # Create the directory with the current date and time in its name
    current_time = datetime.now().strftime("%m%d_%H%M")
    dir_n = f"incorrect_hrps_bestepoch_{epoch}_{current_time}"
    dir_name = os.path.join(log_save_path, dir_n)
    # Delete the previously saved directory if it exists
    if previous_dir_name and os.path.exists(previous_dir_name):
        import shutil
        shutil.rmtree(previous_dir_name)
    
    os.makedirs(dir_name, exist_ok=True)

    with open(f'{dir_name}/incorrect_file_paths_epoch_{epoch}_{current_time}.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile)
        filewriter.writerow(['File Path', 'True Label', 'Predicted Label'])  # Write header
        for path, true_label, pred_label in incorrect_details:
            filewriter.writerow([path, true_label, pred_label])
    
    # Update the previous_dir_name to the current one
    previous_dir_name = dir_name
    return previous_dir_name

# TODO: Implement this function !!!
def check_cfg(cfg):
    if cfg.data_info == "HRP2":
        assert cfg.num_classes == 2, "Number of classes for HRP2 dataset should be 2."


