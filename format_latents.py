import numpy as np
import os
import argparse
import re


def format_latents(args):
    latents_dir = args.latents_dir
    latents_save_dir = args.latents_save_dir
    latent_save_name = args.latents_save_name 

    with open(os.path.join(latents_dir, 'latents.npy'), 'rb') as f:
        restyle_latents = np.load(f, allow_pickle=True).item()

    num_channels, hidden_dim = restyle_latents[list(restyle_latents.keys())[0]][-1].shape
    num_img = len(restyle_latents)
    latents_all = np.zeros((num_img, num_channels, hidden_dim))

    for k, v in restyle_latents.items():
        idx = int(re.findall(r'\d+', k)[0])
        latents_all[idx, :, :] = v[-1]

    np.save(os.path.join(latents_save_dir, latent_save_name), latents_all)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latents_dir', type=str)
    parser.add_argument('--latents_save_dir', type=str, default='')
    parser.add_argument('--latents_save_name', type=str, default='', help='Name of saved file. Should end with .npy') 

    args = parser.parse_args()

    if not os.path.exists(args.latents_save_dir):
        os.makedirs(args.latents_save_dir)
    
    format_latents(args)