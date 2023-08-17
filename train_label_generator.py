# Code adopted from: https://github.com/nv-tlabs/editGAN_release/blob/release_final/train_interpreter.py

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import cv2


from PIL import Image
import imageio
import scipy.misc
import json

from handsoff.utils.model_utils import *
from handsoff.utils.data_utils import *
from handsoff.models.classifer import *

import scipy.stats
import argparse
import time

device_ids = [0]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

class trainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

def prepare_data(args, palette):

    # get generator and upsamplers to form hypercolumn representations
    g_all, _, upsamplers, _, avg_latent = prepare_model(args)

    #Load latent codes of training images
    #   Expected format: latent_all is a [num_style_blocks (18 or 16) x latent_dim (512) x num_train] numpy array
    latent_all = np.load(args['optimized_latent_path']['train'])    
    latent_all = torch.from_numpy(latent_all).cuda()
    latent_all = latent_all[:args['max_training']]

    num_data = len(latent_all)

    # Load existing labels
    mask_list = []
    im_list = []

    d1, d2 = args['dim'][0], args['dim'][1]

    for i in range(len(latent_all)):
        print("loading ", i+1, "/", len(latent_all))


        if i >= args['max_training']:
            break

        name = 'image_mask%0d.npy' % i

        im_frame = np.load(os.path.join( args['annotation_mask_path'] , name))

        mask = np.array(im_frame)
        mask =  cv2.resize(np.squeeze(mask), dsize=(d2, d1), interpolation=cv2.INTER_NEAREST)

        mask_list.append(mask)

        #training images should be pngs or jpgs
        im_name = os.path.join( args['annotation_mask_path'], 'image_%d.png' % i)
        if not os.path.isfile(im_name):
            im_name = os.path.join( args['annotation_mask_path'], 'image_%d.jpg' % i)

        img = Image.open(im_name)
        img = img.resize((d2, d1))

        im_list.append(np.array(img))

    print("Loaded all training data")
    if args['task'] == 'segmentation':
        for i in range(len(mask_list)): 
            for target in range(1, 50):
                if (mask_list[i] == target).sum() < 30:
                    mask_list[i][mask_list[i] == target] = 0

    all_mask = np.stack(mask_list)
    num_feat = args['dim'][-1]

    all_mask_train = np.zeros((d1 * d2 * len(latent_all),), dtype=np.float16) 
    all_feature_maps_train = np.zeros((d1 * d2 * len(latent_all), num_feat), dtype=np.float16)

    vis = []
    for i in range(len(latent_all) ):
        print(f"Creating hypercolumn representation for image {i+1} / {len(latent_all)}")

        gc.collect()

        latent_input = latent_all[i].float()

        #Form hypercolumn representations + get image from latent code for sanity check
        img, feature_maps, _ = latent_to_image(
            g_all, upsamplers, 
            latent_input.unsqueeze(0), 
            dim=args['dim'][1],
            return_upsampled_layers=True, 
            use_style_latents=args['annotation_data_from_w']
        )

        if d1 != d2 and 'car' in args['domain']:
            img = img[:, 64:448]
            feature_maps = feature_maps[:, :, 64:448]

        mask = all_mask[i:i + 1]
        feature_maps = feature_maps.permute(0, 2, 3, 1)

        feature_maps = feature_maps.reshape(-1, num_feat)
        new_mask =  np.squeeze(mask)

        mask = mask.reshape(-1)

        all_feature_maps_train[d1 * d2 * i: d1 * d2 * (i+1)] = feature_maps.cpu().detach().numpy().astype(np.float16)        
        all_mask_train[d1 * d2 * i : d1 * d2 * (i+1)] = mask.astype(np.float16)


        img_show =  cv2.resize(np.squeeze(img[0]), dsize=(d2, d1), interpolation=cv2.INTER_NEAREST)
        mask_color = colorize_mask(new_mask, palette)

        img_mask =  np.uint8(0.7 * img_show + 0.3 * mask_color)

        curr_vis = np.concatenate( [im_list[i], img_show, mask_color, img_mask], 0 )

        vis.append( curr_vis )

    print(f'Saving visualization')
    vis = np.concatenate(vis, 1)
    imageio.imsave(
        os.path.join(args['exp_dir'], "train_data.jpg"),
        vis
    )

    return all_feature_maps_train, all_mask_train, num_data



def main(args):
    if args['domain'] == 'car':
        from handsoff.utils.data_utils import car_20_palette as palette
    elif args['domain'] == 'face':
        from handsoff.utils.data_utils import face_palette as palette
    else:
        assert False

    # Get hypercolumn representions and corresponding masks
    all_feature_maps_train_all, all_mask_train_all, num_data = prepare_data(args, palette)

    train_data = trainData(torch.FloatTensor(all_feature_maps_train_all),
                           torch.FloatTensor(all_mask_train_all))

    max_label = args['number_class']
    batch_size = args['batch_size']
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    print(f"Number of classes: {max_label} --------------------------")
    print(f'Number of training examples: {num_data} -----------------')
    print(f"Dataloader length: {len(train_loader)} ------------------")

    for MODEL_NUMBER in range(args['ensemble_size']): 
        gc.collect()
        if args['task'] == 'segmentation':
            classifier = label_generator_mlp(numpy_class=args['number_class'], dim = args['dim'][-1], layer_sizes = args['classifier']) 

        classifier.init_weights()

        classifier = nn.DataParallel(classifier, device_ids=device_ids).cuda()
        criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)
        classifier.train()

        iteration = 0
        break_count = 0
        best_loss = 10000000
        stop_sign = 0
        
        for epoch in range(100):
            for X_batch, y_batch in train_loader:

                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                if args['task'] == 'segmentation':
                    y_batch = y_batch.type(torch.long)

                optimizer.zero_grad()
                y_pred = classifier(X_batch)

                loss = criterion(y_pred, y_batch)

                loss.backward()
                optimizer.step()

                iteration += 1
                if iteration % 1000 == 0:
                    print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item())
                    gc.collect()


                if iteration % 10000 == 0:
                    model_iter_path = os.path.join(args['exp_dir'], 'iter_pth')
                    if not os.path.exists(os.path.join(model_iter_path)):
                        os.makedirs(os.path.join(model_iter_path))

                    model_path = os.path.join(
                        model_iter_path,
                        f'model_iter{iteration}_number_{MODEL_NUMBER}.pth'
                    )

                    print('Save checkpoint, Epoch : ', str(epoch), ' Path: ', model_path)
                    torch.save({'model_state_dict': classifier.state_dict()}, model_path)

                if epoch > 3:
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        break_count = 0
                    else:
                        break_count += 1

                    if break_count > 50:
                        stop_sign = 1
                        print(f"Break criteria met, total iterations: {iteration} at epoch {epoch} ---------------")
                        break

            if stop_sign == 1:
                break

        gc.collect()
        model_path = os.path.join(
            args['exp_dir'],
            f'model_{MODEL_NUMBER}.pth'
        )
        
        MODEL_NUMBER += 1
        
        print(f'Saving model to {model_path}')
        torch.save({'model_state_dict': classifier.state_dict()}, model_path)
        
        gc.collect()
        torch.cuda.empty_cache()    # clear cache memory on GPU


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str)

    args = parser.parse_args()

    opts = json.load(open(args.exp, 'r'))
    
    path = opts['exp_dir']
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))

    os.system('checkpoint: %s %s' % (args.exp, opts['exp_dir']))
    main(opts)
