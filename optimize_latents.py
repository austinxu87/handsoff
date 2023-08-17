import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

torch.manual_seed(0)

import numpy as np
import pickle

from tqdm import tqdm
import json
import os
import imageio
import argparse
import glob

device_ids = [0]

from handsoff.utils.data_utils import *
from handsoff.utils.model_utils import *
import lpips

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def embed_one_example_latent_init(
    args, 
    latent_in, 
    path, 
    g_all, 
    upsamplers,
    inter, 
    percept, 
    steps, 
    sv_dir,
    skip_exist=False
):
    if os.path.exists(sv_dir):
        if skip_exist:
            return 0,0,[], []
        else:
            pass
    else:
        os.system('mkdir -p %s' % (sv_dir))

    print(f'Save folder {sv_dir}')
    image_path = path

    label_im_tensor, im_id = load_one_image_for_embedding(image_path, args['dim'][:2])

    label_im_tensor = label_im_tensor.to(device)
    label_im_tensor = label_im_tensor * 2.0 - 1.0
    label_im_tensor = label_im_tensor.unsqueeze(0)

    im_out_wo_encoder, _ = latent_to_image(
        g_all, 
        upsamplers, 
        latent_in,
        process_out = True, 
        use_style_latents = True,
        return_only_im = True
    )

    out = run_embedding_optimization(
        args, 
        g_all,
        upsamplers, 
        inter, 
        percept,
        label_im_tensor, 
        latent_in, 
        steps=steps,
        regular_by_org_latent=True,
        use_noise=False,
        early_stop=False
    )
    
    optimized_latent, optimized_noise, loss_cache = out
    optimized_noise = None

    print(f"Loss: {loss_cache[0]} -> {loss_cache[-1]}")
    optimized_latent_np = optimized_latent.detach().cpu().numpy()[0]
    loss_cache_np = []

    # Generate visualization
    img_out, _, _ = latent_to_image(
        g_all, 
        upsamplers, 
        optimized_latent,
        process_out=True, 
        use_style_latents=True,
        return_only_im=True, 
        noise=optimized_noise
    )

    raw_im_show = (np.transpose(label_im_tensor.cpu().numpy(), (0, 2, 3, 1))) * 255.

    vis_list = [im_out_wo_encoder[0], img_out[0]]
    curr_vis = np.concatenate(vis_list, 0)

    imageio.imsave(
        os.path.join(sv_dir, "reconstruction.jpg"),
        curr_vis
    )

    imageio.imsave(
        os.path.join(sv_dir, "real_im.jpg"),
        raw_im_show[0]
    )

    return loss_cache[0], loss_cache[-1], optimized_latent_np, loss_cache_np



def optimize(args, args_opt):

    steps = args_opt.steps
    latent_in_path = args_opt.latents_path 
    latent_save_name = args_opt.latents_save_name
    latent_save_folder = args_opt.latents_save_dir

    g_all, _, upsamplers, _, avg_latent = prepare_model(args)
    inter = Interpolate(args['dim'][1], 'bilinear')

    percept = lpips.PerceptualLoss(
        model='net-lin', net='vgg', use_gpu=device.startswith('cuda'), normalize=args['normalize']
    ).to(device)

    assert latent_save_folder != ""

    all_images = []
    all_id = []

    #Iterate over all images to refine, get image names
    curr_images_all = glob.glob(args['images_dir'] + '/*')

    curr_images_all = [
        data for data in curr_images_all 
        if ('jpg' in data or 'webp' in data or 'png' in data  or 'jpeg' in data or 'JPG' in data) 
        and not os.path.isdir(data)  
        and not 'npy' in data 
    ]

    for i, image in enumerate(curr_images_all):
        all_id.append(image.split("/")[-1].split(".")[0])
        all_images.append(image)

    all_images, all_id = zip(*sorted(zip(all_images, all_id)))
    all_id = all_id[:args['max_training']]
    all_images = all_images[:args['max_training']]
    
    print(f'Number of images to optimize: {len(all_images)}')

    latents_save_dir = os.path.join(latent_save_folder, f'opt_{len(all_images)}_latents')
    if not os.path.exists(latents_save_dir):
        os.makedirs(latents_save_dir)


    all_loss_before_opti, all_loss_after_opti = [], []
    
    latents_orig = np.load(latent_in_path)
    latents_cat = np.zeros(latents_orig.shape)
    for i, id in enumerate(tqdm(all_id)):
        print(f"Current ID: {id}")

        img_num = int(id.split('_')[-1])
        sv_folder = os.path.join(latents_save_dir, id, f'crop_latent_{steps}')

        latent_in = torch.from_numpy(latents_orig[img_num]).type(torch.FloatTensor).unsqueeze(0).to(device)     

        loss_before_opti, loss_after_opti , all_final_latent, all_final_noise = embed_one_example_latent_init(
            args, 
            latent_in, 
            all_images[i],
            g_all, 
            upsamplers, 
            inter, 
            percept, 
            steps, 
            sv_folder,
            skip_exist=False
        )

        all_loss_before_opti.append(loss_before_opti)
        all_loss_after_opti.append(loss_after_opti)

        id_num = id.split("_")[-1]
        latent_name = os.path.join(latents_save_dir, f'latents_image_{id_num}.npy')
        print(latent_name)
        np.save(latent_name, all_final_latent)

        if len(all_final_noise) != 0:
            latent_name = os.path.join(latents_save_dir, f'latents_image_{id_num}_npose.npy')
            with open(latent_name, 'wb') as handle:
                pickle.dump(all_final_noise, handle)

        latents_cat[img_num] = all_final_latent

    result = {
        "before:": np.mean(all_loss_before_opti), 
        "after": np.mean(all_loss_after_opti)
    }

    with open(latents_save_dir + '/result.json', 'w') as f:
        json.dump(result, f)

    np.save(os.path.join(latent_save_folder, latent_save_name), latents_cat)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str)
    parser.add_argument('--latents_save_dir', type=str, default='')
    parser.add_argument('--images_dir', type=str)
    parser.add_argument('--latents_path', type=str)
    parser.add_argument('--latents_save_name', type=str)
    parser.add_argument('--steps', type=int, default=500)

    args = parser.parse_args()
    opts = json.load(open(args.exp, 'r'))

    opts['im_size'] = opts['dim']
    opts['images_dir'] = args.images_dir

    if not os.path.exists(args.latents_save_dir):
        os.makedirs(args.latents_save_dir)
    
    optimize(opts, args)

    #update exp.json with save path
    saved_opt_latents_path = os.path.join(args.latents_save_dir, args.latents_save_name)
    del opts['im_size']
    del opts['images_dir']
    opts['optimized_latent_path']['train'] = saved_opt_latents_path

    with open(args.exp, 'w') as f:
        json.dump(opts, f, indent=4)
