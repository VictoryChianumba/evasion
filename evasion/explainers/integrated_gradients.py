import torch
import captum.attr as CA
from typing import Optional


class IGExplainer:
    """
    Integrated Gradients explainer wrapper.

    Args:
        model:    PyTorch model to explain
        n_steps:  number of interpolation steps
        max_n:    max samples to attribute at once (memory control)
    """

    def __init__(self, model: torch.nn.Module, n_steps: int = 16, max_n: int = 128):
        self.model = model
        self.n_steps = n_steps
        self.max_n = max_n
        self.ig = CA.IntegratedGradients(model)

    def attribute(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute IG attributions for a batch of inputs.

        Args:
            X: input tensor (N, C, T)
            y: target labels (N,)

        Returns:
            attribution tensor (N, C, T)
        """
        if self.max_n and X.size(0) > self.max_n:
            X = X[:self.max_n]
            y = y[:self.max_n]

        X = X.detach().clone().requires_grad_(True)
        return self.ig.attribute(
            X,
            target=y,
            n_steps=self.n_steps,
            baselines=torch.zeros_like(X),
            internal_batch_size=int(X.size(0)),
        )
