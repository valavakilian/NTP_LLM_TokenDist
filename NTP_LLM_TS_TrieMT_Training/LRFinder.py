import torch
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F


class LRFinder:
    def __init__(
        self, 
        model, 
        optimizer, 
        criterion, 
        device,
        min_lr=1e-7,
        max_lr=10,
        num_iterations=100,
        step_mode="exp"
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_iterations = num_iterations
        self.step_mode = step_mode
        
        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.memory = {}


    def reset(self):
        """Restores the model and optimizer to their initial states."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.memory[name]
                
        self.optimizer.load_state_dict(self.memory["optimizer"])
    
    def range_test(self, train_loader, accumulation_steps=1, beta=0.98):
        """Performs the learning rate range test."""
        # Save the model and optimizer states
        self.memory = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.memory[name] = param.data.clone()
                
        self.memory["optimizer"] = self.optimizer.state_dict()
        
        # Calculate the learning rate progression
        if self.step_mode == "exp":
            self.lr_schedule = np.exp(np.linspace(np.log(self.min_lr),
                                                 np.log(self.max_lr),
                                                 self.num_iterations))
        else:
            self.lr_schedule = np.linspace(self.min_lr,
                                         self.max_lr,
                                         self.num_iterations)
        
        # Initialize variables
        avg_loss = 0.0
        best_loss = 0.0
        batch_num = 0
        is_infinite = False
        
        # Progress bar
        pbar = tqdm(total=self.num_iterations, desc="Finding optimal learning rate")
        
        train_iter = iter(train_loader)
        
        try:
            for iteration in range(self.num_iterations):
                # Get a batch
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                
                # Extract input and target
                x_batch = batch[:, :-1].to(self.device)
                y_batch = batch[:, 1:].to(self.device)
                
                # Train step
                loss = self._train_batch(x_batch, y_batch)
                
                # Update running loss
                avg_loss = beta * avg_loss + (1 - beta) * loss
                smoothed_loss = avg_loss / (1 - beta ** (iteration + 1))
                
                # Track best loss
                if iteration == 0:
                    best_loss = smoothed_loss
                else:
                    if smoothed_loss < best_loss:
                        best_loss = smoothed_loss
                
                # Stop if loss is exploding
                if smoothed_loss > 4 * best_loss or torch.isnan(torch.tensor(smoothed_loss)):
                    print("Loss is exploding, stopping early...")
                    is_infinite = True
                    break
                
                # Record values
                self.history["lr"].append(self.lr_schedule[iteration])
                self.history["loss"].append(smoothed_loss)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"loss": f"{smoothed_loss:.4f}", 
                                "lr": f"{self.lr_schedule[iteration]:.2e}"})
                
                # Update learning rate
                self._set_learning_rate(self.lr_schedule[iteration])
                
        except KeyboardInterrupt:
            print("Interrupted by user")
            
        pbar.close()
        
        # Reset the model and optimizer to their initial states
        self.reset()
        
        return self.history
    
    def _train_batch(self, x, y):
        """Trains the model for one batch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        output = self.model(x)
        loss = self.criterion(output.logits, y)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _set_learning_rate(self, lr):
        """Sets the learning rate for the optimizer."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
    
    def plot(self, skip_start=10, skip_end=5, log_lr=True):
        """Plots the learning rate range test results."""
        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")
            
        # Get the data to plot from the history dictionary
        lrs = self.history["lr"]
        losses = self.history["loss"]
        
        # Remove the skipped points
        if skip_start:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        if skip_end:
            lrs = lrs[:-skip_end]
            losses = losses[:-skip_end]
            
        # Create the figure and plot the data
        plt.figure(figsize=(10, 6))
        if log_lr:
            plt.semilogx(lrs, losses)
        else:
            plt.plot(lrs, losses)
            
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder Results")
        plt.grid(True, alpha=0.3)
        
        return plt

# Example usage
def find_optimal_lr(model, train_loader, device):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-7)  # Start with a very small learning rate
    criterion = torch.nn.CrossEntropyLoss()
    
    lr_finder = LRFinder(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        min_lr=1e-7,
        max_lr=10,
        num_iterations=100
    )
    
    history = lr_finder.range_test(train_loader)
    
    # Plot the results
    lr_finder.plot()
    plt.show()
    
    # Find the learning rate with the steepest negative gradient
    losses = np.array(history['loss'])
    lrs = np.array(history['lr'])
    
    # Calculate the gradient
    gradients = (losses[1:] - losses[:-1]) / (lrs[1:] - lrs[:-1])
    
    # Find the point with the steepest negative gradient
    steepest_point = np.argmin(gradients)
    optimal_lr = lrs[steepest_point]
    
    return optimal_lr