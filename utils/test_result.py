import unittest
import torch
import numpy as np
from result import ResultMLS

# Load `label.npy` and `pred.npy`
label_np = np.load('label.npy')
pred_np = np.load('pred.npy')

label_np = np.random.randint(low=0, high=2, size=label_np.shape)
pred_np = np.random.rand(*pred_np.shape)

# Convert to torch tensors
label = torch.tensor(label_np)
pred = torch.tensor(pred_np)
print(type(label), type(pred))

# Initialize ResultMLS
result_mls = ResultMLS(num_cls=3, auc_calc=True, class_prediction='Normal', debug=True)
result_mls.init()

# Evaluate predictions
result_mls.eval(label, pred)

# Process high-risk predictions
probe, pred, true = result_mls._process_hrp2(pred, label)

print(type(probe), type(pred), type(true))
result_mls.pred = pred
result_mls.true = true

# Calculate metrics
result_mls._calculate_metrics(probe)

# Calculate confusion matrix
result_mls._calculate_confusion_matrix()

# Extract metrics
result_mls.pars = [
    result_mls.acc, result_mls.pre, result_mls.rec, 
    result_mls.f1, result_mls.auc, result_mls.acc_cls, 
    result_mls.pre_cls, result_mls.rec_cls, 
    result_mls.f1_cls, result_mls.mls_auc
]

# Define titles and format strings for dataset metrics
datatype = "example"  # Replace with the actual datatype name
titles = ["dataset", "ACC", "PRE", "REC", "F1", "AUC"]
items = [datatype.upper()] + result_mls.pars[:6]
header_format = "\n|{:^8}" + "|{:^5}" * (len(titles) - 1) + "|"
row_format = "\n|{:^8}" + "|{:^.3f}" * (len(titles) - 1) + "|"

# Print dataset metrics
table_str = header_format.format(*titles)
table_str += row_format.format(*items)
print(table_str)

# Define titles for class metrics
titles = ["class", "ACC", "PRE", "REC", "F1", "AUC"]
header_format = "\n|{:^8}" + "|{:^5}" * (len(titles) - 1) + "|"
row_format = "\n|{:^8}" + "|{:^.3f}" * (len(titles) - 1) + "|"

# Print class metrics
class_table_str = header_format.format(*titles)
for idx in range(len(result_mls.acc_cls)):
    items = [
        str(idx), 
        result_mls.acc_cls[idx], 
        result_mls.pre_cls[idx], 
        result_mls.rec_cls[idx], 
        result_mls.f1_cls[idx], 
        result_mls.mls_auc[idx]
    ]
    class_table_str += row_format.format(*items)

print(class_table_str)

# Format and print the confusion matrix
cm_titles = [" "] + ["0", "1"]
cm_header_format = "\n|{:^8}" + "|{:^5}" * 2 + "|"
cm_row_format = "\n|{:^8}" + "|{:^5}" * 2 + "|"

# Iterate through each label's confusion matrix
cm_table_str = ""
for label_idx, label_cm in enumerate(result_mls.cm):
    label_title = f"Label {label_idx}"
    cm_table_str += f"\nConfusion Matrix for {label_title}"
    cm_table_str += cm_header_format.format(*cm_titles)
    
    for i, row in enumerate(label_cm):
        cm_items = [str(i)] + [str(item) for item in row]
        cm_table_str += cm_row_format.format(*cm_items)

print(cm_table_str)
