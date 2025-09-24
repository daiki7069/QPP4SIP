import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    where:
    - p_t is the probability of the true class
    - alpha_t is the weighting factor for class t
    - gamma is the focusing parameter
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass of Focal Loss
        
        Args:
            inputs: Model predictions (logits) [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        
        Returns:
            Focal loss value
        """
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute p_t (probability of true class)
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # alpha should be a list/tensor of size num_classes
                if self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)
                alpha_t = self.alpha[targets]
            ce_loss = alpha_t * ce_loss
        
        # Compute focal loss: (1 - pt)^gamma * ce_loss
        # This down-weights easy examples and focuses on hard examples
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def create_focal_loss_for_sip(class_imbalance_ratio=6.5, gamma=2.0):
    """
    Create Focal Loss optimized for SIP task with current class imbalance
    
    Args:
        class_imbalance_ratio: Ratio of non-initiative to initiative (default: 6.5)
        gamma: Focusing parameter (default: 2.0)
    
    Returns:
        FocalLoss instance
    """
    # Calculate alpha values for class weighting
    # For class imbalance ratio N:1, we want to give more weight to minority class
    # alpha_0 = 1 / (1 + class_imbalance_ratio) for majority class (SIP=0)
    # alpha_1 = class_imbalance_ratio / (1 + class_imbalance_ratio) for minority class (SIP=1)
    alpha_0 = 1.0 / (1.0 + class_imbalance_ratio)  # ~0.13 for ratio 6.5:1
    alpha_1 = class_imbalance_ratio / (1.0 + class_imbalance_ratio)  # ~0.87 for ratio 6.5:1
    
    # Create alpha tensor: [alpha_for_class_0, alpha_for_class_1]
    alpha_tensor = torch.tensor([alpha_0, alpha_1], dtype=torch.float32)
    
    return FocalLoss(
        alpha=alpha_tensor,
        gamma=gamma,
        reduction='mean'
    )