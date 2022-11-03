import torch
import torch.nn as nn
from typing import Optional

class Trainer():

    def __init__(self,
                 model: nn.Module,
                 train_dataloader: torch.utils.data.DataLoader,
                 valid_dataloader: torch.utils.data.DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                 loss_fn: nn.Module,
                 epochs: int = 25
    ) -> None:
        
        # Parse class arguments.
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.epochs = epochs

        # Set up hardware agnotic code.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set random seeds for reproducibility purposes.
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def run(self):
        pass

    def train_step(self):
        pass

    def validation_step(self):
        pass