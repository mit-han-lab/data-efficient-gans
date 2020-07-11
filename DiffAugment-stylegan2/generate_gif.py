import argparse
import os
import numpy as np
from PIL import Image

import dnnlib
from dnnlib import tflib

from training import misc

# ----------------------------------------------------------------------------


def run(resume, output, num_rows, num_cols, num_phases, transition_frames, static_frames, seed):
    tflib.init_tf({'rnd.np_random_seed': seed})
    _, _, Gs = misc.load_pkl(resume)
    output_seq = []
    batch_size = num_rows * num_cols
    latent_size = Gs.input_shape[1]
    latents = [np.random.randn(batch_size, latent_size) for _ in range(num_phases)]
    
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False

    def to_image_grid(outputs):
        outputs = np.reshape(outputs, [num_rows, num_cols, *outputs.shape[1:]])
        outputs = np.concatenate(outputs, axis=1)
        outputs = np.concatenate(outputs, axis=1)
        return Image.fromarray(outputs)
    
    for i in range(num_phases):
        dlatents0 = Gs.components.mapping.run(latents[i - 1], None)
        dlatents1 = Gs.components.mapping.run(latents[i], None)
        for j in range(transition_frames):
            dlatents = (dlatents0 * (transition_frames - j) + dlatents1 * j) / transition_frames
            output_seq.append(to_image_grid(Gs.components.synthesis.run(dlatents, **Gs_kwargs)))
        output_seq.extend([to_image_grid(Gs.components.synthesis.run(dlatents1, **Gs_kwargs))] * static_frames)
    if not output.endswith('.gif'):
        output += '.gif'
    output_seq[0].save(output, save_all=True, append_images=output_seq[1:], optimize=False, duration=50, loop=0)


def main():
    parser = argparse.ArgumentParser(
        description='Generate GIF.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-r', '--resume', help='Resume checkpoint path', required=True)
    parser.add_argument('-o', '--output', help='Output file name', required=True)
    parser.add_argument('--num-rows', help='Number of rows', default=2, type=int)
    parser.add_argument('--num-cols', help='Number of columns', default=3, type=int)
    parser.add_argument('--num-phases', help='Number of phases', default=5, type=int)
    parser.add_argument('--transition-frames', help='Number of transition frames per phase', default=20, type=int)
    parser.add_argument('--static-frames', help='Number of static frames per phase', default=5, type=int)
    parser.add_argument('--seed', help='Random seed', default=1000, type=int)

    args = parser.parse_args()

    run(**vars(args))


if __name__ == "__main__":
    main()
