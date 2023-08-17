# Adapted from https://github.com/nv-tlabs/datasetGAN_release/blob/master/datasetGAN/train_interpreter.py

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import sys
sys.path.append('..')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.distributions import Categorical

torch.manual_seed(0)

import scipy.misc
import json
import numpy as np

device_ids = [0]
from PIL import Image

from handsoff.utils.model_utils import *
from handsoff.utils.data_utils import *
from handsoff.models.classifer import *
from handsoff.utils.data_utils import face_palette as palette

import pickle
import scipy.stats
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_data(args, checkpoint_path, num_sample, start_step=0, vis=True):
    if args['domain'] == 'car':
        from handsoff.utils.data_utils import car_20_palette as palette
    elif args['domain'] == 'face':
        from handsoff.utils.data_utils import face_palette as palette
    else:
        assert False

    if not vis:
        result_path = os.path.join(checkpoint_path, 'samples')
    else:
        result_path = os.path.join(checkpoint_path, 'vis_%d' % num_sample)
    if os.path.exists(result_path):
        pass
    else:
        os.system('mkdir -p %s' % (result_path))
        print('Experiment folder created at: %s' % (result_path))


    g_all, _, upsamplers, _, avg_latent = prepare_model(args)

    classifier_list = []
    for MODEL_NUMBER in range(args['ensemble_size']):
        print('MODEL_NUMBER', MODEL_NUMBER)
        classifier = label_generator_mlp(
            numpy_class=args['number_class'], 
            dim = args['dim'][-1], 
            layer_sizes = args['classifier']
        )
        classifier =  nn.DataParallel(classifier, device_ids=device_ids).cuda()
        checkpoint = torch.load(os.path.join(checkpoint_path, 'model_' + str(MODEL_NUMBER) + '.pth'))
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier.eval()
        classifier_list.append(classifier)

    softmax_f = nn.Softmax(dim=1)
    with torch.no_grad():
        latent_cache = []
        image_cache = []
        seg_cache = []
        entropy_calculate = []
        results = []
        np.random.seed(start_step)
        count_step = start_step

        print( "num_sample: ", num_sample)

        for i in range(num_sample):
            if i % 100 == 0:
                print("Generate", i, "Out of:", num_sample)

            curr_result = {}

            latent = np.random.randn(1, 512)

            curr_result['latent'] = latent


            latent = torch.from_numpy(latent).type(torch.FloatTensor).to(device)
            latent_cache.append(latent)


            img, affine_layers, num_feat = latent_to_image(
                g_all, 
                upsamplers, 
                latent, 
                dim=args['dim'][1],
                return_upsampled_layers=True
            )

            args['dim'][-1] = num_feat

            if args['dim'][0] != args['dim'][1] and args['domain'] == 'car':
                img = img[:, 64:448][0]
            else:
                img = img[0]

            image_cache.append(img)
            if args['dim'][0] != args['dim'][1]:
                affine_layers = affine_layers[:, :, 64:448]
            affine_layers = affine_layers[0]

            affine_layers = affine_layers.reshape(args['dim'][-1], -1).transpose(1, 0)

            all_seg = []
            all_entropy = []
            mean_seg = None

            seg_mode_ensemble = []
            for MODEL_NUMBER in range(args['ensemble_size']):
                classifier = classifier_list[MODEL_NUMBER]
                img_seg = classifier(affine_layers)
                img_seg = img_seg.squeeze()

                entropy = Categorical(logits=img_seg).entropy()
                all_entropy.append(entropy)

                all_seg.append(img_seg)
                if mean_seg is None:
                    mean_seg = softmax_f(img_seg)
                else:
                    mean_seg += softmax_f(img_seg)

                img_seg_final = oht_to_scalar(img_seg)
                img_seg_final = img_seg_final.reshape(args['dim'][0], args['dim'][1], 1)
                img_seg_final = img_seg_final.cpu().detach().numpy()

                seg_mode_ensemble.append(img_seg_final)

            mean_seg = mean_seg / len(all_seg)
            full_entropy = Categorical(mean_seg).entropy()
            js = full_entropy - torch.mean(torch.stack(all_entropy), 0)
            top_k = js.sort()[0][- int(js.shape[0] / 10):].mean()
            entropy_calculate.append(top_k)

            img_seg_final = np.concatenate(seg_mode_ensemble, axis=-1)
            img_seg_final = scipy.stats.mode(img_seg_final, 2)[0].reshape(args['dim'][0], args['dim'][1])
            del (affine_layers)

            if vis:
                color_mask = 0.7 * colorize_mask(img_seg_final, palette) + 0.3 * img
                color_mask = Image.fromarray(color_mask.astype('uint8'))
                color_mask.save(os.path.join(result_path, "vis_" + str(i) + '.png'))
                img = Image.fromarray(img)
                img.save(os.path.join(result_path, "vis_" + str(i) + '_image.png'))
            else:
                seg_cache.append(img_seg_final)
                curr_result['uncertrainty_score'] = top_k.item()
                image_label_name = os.path.join(result_path, 'label_' + str(count_step) + '.png')
                image_name = os.path.join(result_path,  str(count_step) + '.png')

                js_name = os.path.join(result_path, str(count_step) + '.npy')
                img = Image.fromarray(img)
                img_seg = Image.fromarray(img_seg_final.astype('uint8'))
                js = js.cpu().numpy().reshape(args['dim'][0], args['dim'][1])
                img.save(image_name)
                img_seg.save(image_label_name)
                np.save(js_name, js)
                curr_result['image_name'] = image_name
                curr_result['image_label_name'] = image_label_name
                curr_result['js_name'] = js_name
                count_step += 1

                results.append(curr_result)
                if i % 1000 == 0 and i != 0:
                    with open(os.path.join(result_path, str(i) + "_" + str(start_step) + '.pickle'), 'wb') as f:
                        pickle.dump(results, f)

        with open(os.path.join(result_path, str(num_sample) + "_" + str(start_step) + '.pickle'), 'wb') as f:
            pickle.dump(results, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str)
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--resume', type=str,  default="")
    parser.add_argument('--num_sample', type=int,  default=1000)
    parser.add_argument('--save_vis', type=str, default=False)

    args = parser.parse_args()
    opts = json.load(open(args.exp, 'r'))

    path = opts['exp_dir']
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))

    os.system('cp %s %s' % (args.exp, opts['exp_dir']))

    save_vis = (args.save_vis.lower() == 'true')
    print(save_vis)
    generate_data(opts, args.resume, args.num_sample, vis=save_vis, start_step = args.start_step)
    
    
