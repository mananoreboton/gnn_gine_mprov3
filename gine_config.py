"""
GINE model configuration: separate configuration from model logic.
Build MProGNN from a typed config.
"""

from dataclasses import dataclass

from model import MProGNN


@dataclass(frozen=True)
class GineConfig:
    """Configuration for the GINE-based MPro GNN classification)."""

    in_channels: int = 4  # x, y, z, atomic number
    hidden_channels: int = 64
    num_layers: int = 3
    dropout: float = 0.2
    out_classes: int = 3
    pool: str = "mean"
    edge_dim: int = 1

    def build(self) -> MProGNN:
        """Construct MProGNN from this config."""
        return MProGNN(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            dropout=self.dropout,
            out_classes=self.out_classes,
            pool=self.pool,
            edge_dim=self.edge_dim,
        )
