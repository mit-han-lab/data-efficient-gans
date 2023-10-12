"""Generate GIF using pretrained network pickle."""

import os

import click
import dnnlib
import numpy as np
from PIL import Image
import torch

import legacy

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seed', help='Random seed', default=0, type=int)
@click.option('--num-rows', help='Number of rows', default=1, type=int)
@click.option('--num-cols', help='Number of columns', default=8, type=int)
@click.option('--resolution', help='Resolution of the output images', default=128, type=int)
@click.option('--num-phases', help='Number of phases', default=5, type=int)
@click.option('--transition-frames', help='Number of transition frames per phase', default=20, type=int)
@click.option('--static-frames', help='Number of static frames per phase', default=5, type=int)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--output', type=str, required=True)
def generate_gif(
    network_pkl: str,
    seed: int,
    num_rows: int,
    num_cols: int,
    resolution: int,
    num_phases: int,
    transition_frames: int,
    static_frames: int,
    truncation_psi: float,
    noise_mode: str,
    output: str
):
    """Generate gif using pretrained network pickle.

    Examples:

    \b
    python generate_gif.py --output=obama.gif --seed=0 --num-rows=1 --num-cols=8 \\
        --network=https://data-efficient-gans.mit.edu/models/DiffAugment-stylegan2-100-shot-obama.pkl
    """
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    outdir = os.path.dirname(output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    np.random.seed(seed)

    output_seq = []
    batch_size = num_rows * num_cols
    latent_size = G.z_dim
    latents = [np.random.randn(batch_size, latent_size) for _ in range(num_phases)]

    def to_image_grid(outputs):
        outputs = np.reshape(outputs, [num_rows, num_cols, *outputs.shape[1:]])
        outputs = np.concatenate(outputs, axis=1)
        outputs = np.concatenate(outputs, axis=1)
        return Image.fromarray(outputs).resize((resolution * num_cols, resolution * num_rows), Image.ANTIALIAS)
    
    def generate(dlatents):
        images = G.synthesis(dlatents, noise_mode=noise_mode)
        images = (images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        return to_image_grid(images)
    
    for i in range(num_phases):
        dlatents0 = G.mapping(torch.from_numpy(latents[i - 1]).to(device), None)
        dlatents1 = G.mapping(torch.from_numpy(latents[i]).to(device), None)
        for j in range(transition_frames):
            dlatents = (dlatents0 * (transition_frames - j) + dlatents1 * j) / transition_frames
            output_seq.append(generate(dlatents))
        output_seq.extend([generate(dlatents1)] * static_frames)
    
    if not output.endswith('.gif'):
        output += '.gif'
    output_seq[0].save(output, save_all=True, append_images=output_seq[1:], optimize=False, duration=50, loop=0)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_gif() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
