"""The frozen vision encoder: SigLIP2 so400m/16 at 384 px, patch features merged 2 x 2."""
import torch

GRID = 24          # 384 / 16 patches per side
MERGE = 2
WIDTH = 1152


class FrozenSiglip(torch.nn.Module):
    def __init__(self, path, device, dtype=torch.bfloat16):
        super().__init__()
        from transformers import SiglipVisionModel
        self.model = SiglipVisionModel.from_pretrained(path, dtype=dtype).to(device).eval().requires_grad_(False)
        self.dtype = dtype

    @torch.no_grad()
    def forward(self, images):
        """uint8 (N, 3, 384, 384) -> (N * 144, 4608), images in order, tokens row-major over the merged grid."""
        if images.shape[0] == 0:
            return torch.zeros((0, MERGE * MERGE * WIDTH), device=images.device, dtype=self.dtype)
        pixels = images.to(self.dtype).div(127.5).sub(1.)  # SigLIP normalization: mean 0.5, std 0.5
        hidden = self.model(pixel_values=pixels).last_hidden_state
        n = hidden.shape[0]
        if hidden.shape[1:] != (GRID * GRID, WIDTH):
            raise ValueError(f"unexpected encoder output {tuple(hidden.shape)}")
        g = GRID // MERGE
        merged = hidden.view(n, g, MERGE, g, MERGE, WIDTH).permute(0, 1, 3, 2, 4, 5)
        return merged.reshape(n * g * g, MERGE * MERGE * WIDTH)
