import logging
import os

# import os
import time

import numpy as np
import torch
import torch.nn.functional as F

# import yaml
from imblearn.metrics import sensitivity_score, specificity_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


def mkdirs(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    return


def get_one_hot(label, num_cls):
    label = label.reshape(-1)
    label = torch.eye(num_cls, device=label.device)[label]
    return label


class ResultCLS:
    def __init__(self, num_cls) -> None:
        self.epoch = 1
        self.best_epoch = 0
        self.best_val_result = 0.0
        self.test_acc = 0.0
        self.test_auc = 0.0
        self.test_f1 = 0.0
        self.test_sen = 0.0
        self.test_spe = 0.0
        self.test_pre = 0.0
        self.num_cls = num_cls

        return

    def eval(self, label, pred):
        self.pred.append(pred)
        self.true.append(label)
        return

    def init(self):
        self.st = time.time()
        self.pred = []
        self.true = []
        return

    @torch.no_grad()
    def stastic(self):
        num_cls = self.num_cls

        pred = torch.cat(self.pred, dim=0)
        true = torch.cat(self.true, dim=0)

        probe = torch.softmax(pred, dim=1).cpu().detach().numpy()
        true_one_hot = get_one_hot(true, num_cls).cpu().detach().numpy()
        true = true.cpu().detach().numpy()
        pred = torch.argmax(pred, dim=1).cpu().detach().numpy()

        self.acc = accuracy_score(true, pred)
        self.sen = sensitivity_score(true, pred, average="macro")
        self.spe = specificity_score(true, pred, average="macro")
        self.pre = precision_score(true, pred, average="macro")
        self.f1 = f1_score(true, pred, average="macro")
        self.auc = roc_auc_score(true_one_hot, probe, average="macro")
        self.cm = confusion_matrix(true, pred)
        self.time = np.round(time.time() - self.st, 1)

        self.pars = [self.acc, self.sen, self.spe, self.pre, self.f1, self.auc]
        return

    def print(self, epoch: int, datatype="test"):
        self.stastic()
        titles = ["dataset", "ACC", "SEN", "SPE", "PRE", "F1", "AUC"]
        items = [datatype.upper()] + self.pars
        forma_1 = "\n|{:^8}" + "|{:^5}" * (len(titles) - 1) + "|"
        forma_2 = "\n|{:^8}" + "|{:^.3f}" * (len(titles) - 1) + "|"
        # logging.info(f"ACC: {self.pars[1]:.3f}, TIME: {self.time:.1f}s")
        logging.info(f"ACC: {self.pars[0]:.3f}, TIME: {self.time:.1f}s")
        logging.info((forma_1 + forma_2).format(*titles, *items))
        logging.debug(f"\n{self.cm}")
        self.epoch = epoch

        if datatype == "test":
            self.test_acc = self.acc
            self.test_auc = self.auc
            self.test_f1 = self.f1
            self.test_sen = self.sen
            self.test_spe = self.spe
            self.test_pre = self.pre
            return

        if datatype == "val" and self.auc > self.best_val_result:
            self.best_epoch = epoch
            self.best_val_result = self.auc
        return


class ResultMLS:
    def __init__(self, num_cls, auc_calc=True, class_prediction='HRP', debug=False) -> None:
        self.epoch = 1
        self.best_epoch = 1
        self.best_val_result = 0.0
        self.test_auc = 0.0
        self.test_mls_auc = []
        self.num_cls = num_cls
        self.auc_calc = auc_calc
        self.debug = debug
        self.class_prediction = class_prediction
        return

    def eval(self, label, pred):
        self.pred_raw.append(pred)
        self.true_raw.append(label)
        return

    def init(self):
        self.st = time.time()
        self.pred_raw = []
        self.true_raw = []
        self.pred = []
        self.true = []
        self.probe = []
        return

    @torch.no_grad()
    def stastic(self):
        num_cls = self.num_cls

        if self.debug:
            print("==================debugging===================")

        # Concatenate predictions and true values
        pred = torch.cat(self.pred_raw, dim=0) # ie. the logits
        true = torch.cat(self.true_raw, dim=0)

        # Process predictions based on class type
        if self.class_prediction == 'Normal':
            probe, pred, true = self._process_normal(pred, true)
        elif self.class_prediction == 'Ca5':
            probe, pred = self._process_ca(pred)
        elif self.class_prediction == 'Ca3':
            probe, pred = self._process_ca3(pred)
        elif self.class_prediction == 'HRP':  # TODO: NEED CHECK!!!
            probe, pred, true = self._process_hrp(pred, true)
        elif self.class_prediction == 'HRP2':
            probe, pred, true = self._process_hrp2(pred, true)
        else:
            raise ValueError(f"Invalid class_prediction: {self.class_prediction}")

        self.pred = pred.cpu().detach().numpy() # ie. the predicted class
        self.probe = probe.cpu().detach().numpy() # ie. the probability
        self.true = true.cpu().detach().numpy()

        if self.debug:
            print(f"pred: {self.pred} \n true: {self.true}")

        # Calculate metrics
        self._calculate_metrics(self.probe)

        # Calculate confusion matrix
        self._calculate_confusion_matrix()

        self.time = np.round(time.time() - self.st, 1)

        self.pars = [self.acc, self.pre, self.rec, self.f1, self.auc, self.acc_cls, self.pre_cls, self.rec_cls, self.f1_cls, self.mls_auc]
        return

    def _process_normal(self, pred, true):
        probe = torch.sigmoid(pred)
        pred = (probe > 0.5).float()
        pred = pred.cpu().detach().numpy()
        probe = probe.cpu().detach().numpy()
        true = true.cpu().detach().numpy()
        return probe, pred, true

    def _process_ca(self, pred):
        probe = torch.sigmoid(pred)
        pred = torch.zeros_like(probe)

        # For the first four entries, pick the top 1
        _, top_1_indices = torch.topk(probe[:, :4], 1)
        row_indices = torch.arange(pred.size(0)).unsqueeze(-1)
        pred[row_indices, top_1_indices] = 1

        # Threshold the last entry
        pred[:,-1] = probe[:,-1] > 0.5
        pred = pred.float()
        return probe, pred
    
    def _process_ca3(self, pred):
        pred_ca = pred[:, :2]  # First two entries
        pred_stent = pred[:, 2]
        probe_ca = F.softmax(pred_ca, dim=1)
        probe_stent = torch.sigmoid(pred_stent)
        probe = torch.zeros_like(pred)
        probe[:, :2] = probe_ca
        probe[:, 2] = probe_stent

        pred = torch.zeros_like(probe)

        # For the first two entries, pick the top 1
        _, top_1_indices = torch.topk(probe[:, :2], 1)
        row_indices = torch.arange(pred.size(0)).unsqueeze(-1)
        pred[row_indices, top_1_indices] = 1

        # Threshold the last entry
        pred[:,-1] = probe[:,-1] > 0.5
        pred = pred.float()
        return probe, pred

    def _process_hrp(self, pred, true):
        probe = torch.sigmoid(pred)
        pred = (probe > 0.5).float()
        return probe, pred, true
    
    def _process_hrp2(self, pred, true):
        probe = torch.sigmoid(pred)
        # Binary classification
        if self.num_cls == 1:
            pred = (probe > 0.5).float()
        else:
            pred = torch.zeros_like(probe)
            pred[torch.arange(pred.size(0)), torch.argmax(probe, dim=1)] = 1
            pred = pred.to(torch.int64)
        
        return probe, pred, true

    def _calculate_metrics(self, probe):
        self.acc = accuracy_score(self.true, self.pred)
        self.acc_cls = [accuracy_score(self.true[:, i], self.pred[:, i]) for i in range(self.true.shape[1])]
        self.pre = precision_score(self.true, self.pred, average="macro", zero_division=1)
        self.pre_cls = precision_score(self.true, self.pred, average=None, zero_division=1)
        self.rec = recall_score(self.true, self.pred, average="macro", zero_division=1)
        self.rec_cls = recall_score(self.true, self.pred, average=None, zero_division=1)
        self.f1 = f1_score(self.true, self.pred, average="macro")
        self.f1_cls = f1_score(self.true, self.pred, average=None)

        # Calculate AUC and handle exceptions
        self.auc = self._safe_roc_auc_score(self.true, probe, "macro")
        self.mls_auc = self._safe_roc_auc_score(self.true, probe, None)

    def _safe_roc_auc_score(self, true, probe, average):
        try:
            return roc_auc_score(true, probe, average=average)
        except ValueError:
            print(f"ValueError: AUC calculation failed for average='{average}'. Setting AUC to 0.")
            if average is None:
                return [0 for _ in range(len(self.acc_cls))]
            return 0

    def _calculate_confusion_matrix(self):
        if self.class_prediction == 'Ca5':
            true_number = np.argmax(self.true[:, :4], axis=1)
            pred_number = np.argmax(self.pred[:, :4], axis=1)
            self.cm = confusion_matrix(true_number, pred_number)
        elif self.class_prediction == 'Ca3':
            true_number = np.argmax(self.true[:, :2], axis=1)
            pred_number = np.argmax(self.pred[:, :2], axis=1)
            self.cm = confusion_matrix(true_number, pred_number)
        elif self.class_prediction == 'HRP2' and self.num_cls == 1:
            self.cm = confusion_matrix(self.true, self.pred)
        else:
            if self.debug:
                print(f"Getting multilabel confusion matrix")
            self.cm = multilabel_confusion_matrix(self.true, self.pred)

    @staticmethod
    def extract_indices_ca(array):
        # Check if the array is a tensor and on the GPU
        if isinstance(array, torch.Tensor) and array.is_cuda:
            # Extract the indices where the first 4 columns contain the maximum value
            indices = torch.argmax(array[:, :4], dim=1)
        else:
            # Ensure the array is a numpy array
            array = np.array(array)
            
            # Extract the indices where the first 4 columns contain 1
            indices = np.argmax(array[:, :4], axis=1)

            indices = indices.tolist()
        
        return indices

    def print(self, epoch: int, datatype="val", criteria='auc', metrics=None, update=False, debug=False):
        self.stastic()     
        
        # Define titles and format strings
        titles = ["dataset", "ACC", "PRE", "REC", "F1", "AUC"]
        items = [datatype.upper()] + self.pars[:6]
        header_format = "\n|{:^8}" + "|{:^5}" * (len(titles) - 1) + "|"
        row_format = "\n|{:^8}" + "|{:^.3f}" * (len(titles) - 1) + "|"

        # Log the main metrics
        logging.info((header_format + row_format).format(*titles, *items))

        # If the class prediction is 'Ca', log the detailed class metrics
        if self.class_prediction == 'Ca5':
            if debug:
                # logging prediction   
                logging.info(f'\nPrediction: {self.extract_indices_ca(self.pred)}')
                logging.info(f'\nTrue: {self.extract_indices_ca(self.true)}')
                # Convert the array to a string with all elements shown
                prob_full_array_str = np.array2string(self.probe, separator=', ', threshold=np.inf, edgeitems=np.inf)
                logging.info(f'\nProbability: {prob_full_array_str}')
                true_full_array_str = np.array2string(self.true, separator=', ', threshold=np.inf, edgeitems=np.inf)
                logging.info(f'\nTrue: {true_full_array_str}')
            
            titles = ["class", "ACC", "PRE", "REC", "F1", "AUC"]
            table_str = header_format.format(*titles)
            
            for idx in range(4):
                items = [
                    str(idx), 
                    self.acc_cls[idx], 
                    self.pre_cls[idx], 
                    self.rec_cls[idx], 
                    self.f1_cls[idx], 
                    self.mls_auc[idx]
                ]
                row_str = row_format.format(*items)
                table_str += row_str
            
            logging.info(table_str)
        
            # Format and log the confusion matrix
            cm_titles = [" "] + [str(i) for i in range(len(self.cm))]
            cm_header_format = "\n|{:^8}" + "|{:^5}" * len(self.cm) + "|"
            cm_row_format = "\n|{:^8}" + "|{:^5}" * len(self.cm) + "|"

            cm_table_str = cm_header_format.format(*cm_titles)
            for i, row in enumerate(self.cm):
                cm_items = [str(i)] + list(row)
                cm_row_str = cm_row_format.format(*cm_items)
                cm_table_str += cm_row_str

            logging.info(f"\nConfusion Matrix:{cm_table_str}")
        elif self.class_prediction == 'Ca3':
            titles = ["class", "ACC", "PRE", "REC", "F1", "AUC"]
            table_str = header_format.format(*titles)
            
            for idx in range(2):
                items = [
                    str(idx), 
                    self.acc_cls[idx], 
                    self.pre_cls[idx], 
                    self.rec_cls[idx], 
                    self.f1_cls[idx], 
                    self.mls_auc[idx]
                ]
                row_str = row_format.format(*items)
                table_str += row_str
            
            logging.info(table_str)
        
            # Format and log the confusion matrix
            cm_titles = [" "] + [str(i) for i in range(len(self.cm))]
            cm_header_format = "\n|{:^8}" + "|{:^5}" * len(self.cm) + "|"
            cm_row_format = "\n|{:^8}" + "|{:^5}" * len(self.cm) + "|"

            cm_table_str = cm_header_format.format(*cm_titles)
            for i, row in enumerate(self.cm):
                cm_items = [str(i)] + list(row)
                cm_row_str = cm_row_format.format(*cm_items)
                cm_table_str += cm_row_str

            logging.info(f"\nConfusion Matrix:{cm_table_str}")
        elif self.class_prediction == 'HRP': 
            if debug:
                # logging prediction   
                logging.info(f'\nPrediction: {self.pred}')
                logging.info(f'\nTrue: {self.true}')
                # Convert the array to a string with all elements shown
                prob_full_array_str = np.array2string(self.probe, separator=', ', threshold=np.inf, edgeitems=np.inf)
                logging.info(f'\nProbability: {prob_full_array_str}')
                true_full_array_str = np.array2string(self.true, separator=', ', threshold=np.inf, edgeitems=np.inf)
                logging.info(f'\nTrue: {true_full_array_str}')
            
            titles = ["class", "ACC", "PRE", "REC", "F1", "AUC"]
            table_str = header_format.format(*titles)
            
            for idx in range(5):
                items = [
                    str(idx), 
                    self.acc_cls[idx], 
                    self.pre_cls[idx], 
                    self.rec_cls[idx], 
                    self.f1_cls[idx], 
                    self.mls_auc[idx]
                ]
                row_str = row_format.format(*items)
                table_str += row_str
            
            logging.info(table_str)
        
            # Format and log the confusion matrix
            cm_titles = [" "] + ["0", "1"]

            # Header format
            cm_header_format = "\n|{:^8}" + "|{:^5}" * 2 + "|"

            # Row format
            cm_row_format = "\n|{:^8}" + "|{:^5}" * 2 + "|"

            # Initialize the confusion matrix table string with headers
            cm_table_str = ""

            # Iterate through each label's confusion matrix
            for label_idx, label_cm in enumerate(self.cm):
                label_title = f"Label {label_idx}"
                cm_table_str += f"\nConfusion Matrix for {label_title}"
                
                # Add header row for this label
                cm_table_str += cm_header_format.format(*cm_titles)
                
                # Add rows for this label's confusion matrix
                for i, row in enumerate(label_cm):
                    cm_items = [str(i)] + [str(item) for item in row]
                    cm_row_str = cm_row_format.format(*cm_items)
                    cm_table_str += cm_row_str

            logging.info(f"\nConfusion Matrix:{cm_table_str}") 
        elif self.class_prediction == 'HRP2': 
            if debug:
                # logging prediction   
                logging.info(f'\nPrediction: {self.pred}')
                logging.info(f'\nTrue: {self.true}')
                # Convert the array to a string with all elements shown
                prob_full_array_str = np.array2string(self.probe, separator=', ', threshold=np.inf, edgeitems=np.inf)
                logging.info(f'\nProbability: {prob_full_array_str}')
                true_full_array_str = np.array2string(self.true, separator=', ', threshold=np.inf, edgeitems=np.inf)
                logging.info(f'\nTrue: {true_full_array_str}')
            
            # Binary classification
            if self.num_cls == 1:
                cm_titles = [" "] + [str(i) for i in range(len(self.cm))]
                cm_header_format = "\n|{:^8}" + "|{:^5}" * len(self.cm) + "|"
                cm_row_format = "\n|{:^8}" + "|{:^5}" * len(self.cm) + "|"

                cm_table_str = cm_header_format.format(*cm_titles)
                for i, row in enumerate(self.cm):
                    cm_items = [str(i)] + list(row)
                    cm_row_str = cm_row_format.format(*cm_items)
                    cm_table_str += cm_row_str
            else:
                titles = ["class", "ACC", "PRE", "REC", "F1", "AUC"]
                table_str = header_format.format(*titles)
                
                for idx in range(2):
                    items = [
                        str(idx), 
                        self.acc_cls[idx], 
                        self.pre_cls[idx], 
                        self.rec_cls[idx], 
                        self.f1_cls[idx], 
                        self.mls_auc[idx]
                    ]
                    row_str = row_format.format(*items)
                    table_str += row_str
                
                logging.info(table_str)
            
                # Format and log the confusion matrix
                cm_titles = [" "] + ["0", "1"]

                # Header format
                cm_header_format = "\n|{:^8}" + "|{:^5}" * 2 + "|"

                # Row format
                cm_row_format = "\n|{:^8}" + "|{:^5}" * 2 + "|"

                # Initialize the confusion matrix table string with headers
                cm_table_str = ""

                # Iterate through each label's confusion matrix
                for label_idx, label_cm in enumerate(self.cm):
                    label_title = f"Label {label_idx}"
                    cm_table_str += f"\nConfusion Matrix for {label_title}"
                    
                    # Add header row for this label
                    cm_table_str += cm_header_format.format(*cm_titles)
                    
                    # Add rows for this label's confusion matrix
                    for i, row in enumerate(label_cm):
                        cm_items = [str(i)] + [str(item) for item in row]
                        cm_row_str = cm_row_format.format(*cm_items)
                        cm_table_str += cm_row_str

            logging.info(f"\nConfusion Matrix:{cm_table_str}") 
        
        else:
            raise NotImplementedError(f"Class_prediction: {self.class_prediction} result printing not implemented")                                                                                                                                                                                                                                                                                                                                                                                                                                             


        # TODO: else if class_prediction == 'Normal' or 'HRP', log the detailed class metrics

        # Update epoch
        if update:
            self.update(epoch, datatype, criteria, metrics)

    def update(self, epoch, datatype="val", criteria='auc', metrics=None):
        # Update epoch
        self.epoch = epoch

        # best performance criteria
        if criteria == 'auc':
            if datatype == "test":
                self.test_result = self.auc
                self.test_mls_auc = self.mls_auc
                return

            if datatype == 'val':
                logging.info(f"\nValidation AUC:{self.auc}") 
                if self.auc > self.best_val_result:
                    self.best_epoch = epoch
                    self.best_val_result = self.auc 

        elif criteria == 'loss':
            if datatype == "test":
                logging.info(f"\nTest Loss:{metrics}") 
                self.test_result = metrics
                return
            
            elif datatype == 'val':
                logging.info(f"\nValidation Loss:{metrics}") 
                if metrics < self.best_val_result:
                    self.best_epoch = epoch
                    self.best_val_result = metrics

