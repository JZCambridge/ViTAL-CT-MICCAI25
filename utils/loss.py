import torch
import torch.nn as nn
import torch.nn.functional as F

class CRPSLoss(nn.Module):
    """
    Continuous Ranked Probability Score (CRPS) Loss
    CRPS measures the difference between the cumulative distribution functions (CDFs) of the observed and forecast distributions.
    This implementation assumes Gaussian distributions for simplicity.

    Args:
        mu (Tensor): Predicted mean of the distribution.
        sigma (Tensor): Predicted standard deviation of the distribution.
        y (Tensor): True values.

    Returns:
        Tensor: CRPS loss value.
    """
    def __init__(self):
        super(CRPSLoss, self).__init__()

    def forward(self, predicted_logits, true_labels):
        predicted_probs = torch.softmax(predicted_logits, dim=1)

        batch_size, num_classes = predicted_probs.shape

        predicted_cdf = torch.cumsum(predicted_probs, dim=1)

        true_cdf = torch.zeros_like(predicted_cdf)

        for i in range(batch_size):
            assert true_labels[i] < num_classes, "Check label, true label out of range!"
            true_cdf[i, true_labels[i]:] = 1
        print(true_cdf)

        crps_loss = torch.mean((predicted_cdf - true_cdf) ** 2)
        return crps_loss
    
class Ca3Loss(nn.Module):
    def __init__(self):
        super(Ca3Loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, label):
        # Separate the predictions
        pred_ce = logits[:, :2]  # First two entries for cross-entropy loss
        pred_bce = logits[:, -1]  # Last entry for binary cross-entropy loss

        # Compute the losses
        loss_ce = self.ce_loss(pred_ce, label[:, :2])
        loss_bce = self.bce_loss(pred_bce, label[:,-1])  # Ensure target_bce is float

        # Combine the losses (e.g., sum or average)
        total_loss = loss_ce + loss_bce 
        return total_loss

# Class-Weighted Focal Loss
class CoronaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.65, gamma=1.5, num_classes=2):
        super().__init__()
        self.alpha = torch.tensor([1-alpha, alpha])  # [0.25, 0.75]
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        # print(f'inputs: {inputs.shape}, targets: {targets.shape}')
        
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # print(f'bce_loss: {bce_loss}')
        # Focal weighting
        pt = torch.exp(-bce_loss)
        # print(f'pt: {pt}')
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        # print(f'focal_loss: {focal_loss}')
        
        # Class weighting
        # class_weights = self.alpha[targets.long()].to(inputs.device) # normal
        class_weights = self.alpha.to(inputs.device)[targets.long().to(inputs.device)] # for tan_gong
        return (class_weights * focal_loss).mean() * 2

if __name__ == "__main__":
    # Define the input tensors
    predicted_logits = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 1.0]])
    true_labels = torch.tensor([2, 1])

    # Instantiate the CRPS loss
    crps_loss = CRPSLoss()

    # Compute the CRPS loss
    loss = crps_loss(predicted_logits, true_labels)
    print(loss)