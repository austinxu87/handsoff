# Adapted from https://github.com/nv-tlabs/datasetGAN_release/blob/master/datasetGAN/train_deeplab.py

import sys
sys.path.append('../')
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

import argparse
import gc
import os
import glob
from PIL import Image
import json
import pickle
from handsoff.utils.data_utils import *

import matplotlib.pyplot as plt

class ImageLabelDataset(Dataset):
    def __init__(
            self,
            img_path_list,
            label_path_list, trans,
            img_size=(128, 128),
    ):
        self.label_trans = trans
        self.img_path_list = img_path_list
        self.label_path_list = label_path_list
        self.img_size = img_size

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        im_path = self.img_path_list[index]
        lbl_path = self.label_path_list[index]
        im = Image.open(im_path)
        try:
            lbl = np.load(lbl_path)
        except:
            lbl = np.array(Image.open(lbl_path))
        if len(lbl.shape) == 3:
            lbl = lbl[:, :, 0]

        lbl = self.label_trans(lbl)
        lbl = Image.fromarray(lbl.astype('uint8'))
        im, lbl = self.transform(im, lbl)

        return im, lbl, im_path

    def transform(self, img, lbl):
        img = img.resize((self.img_size[0], self.img_size[1]))
        lbl = lbl.resize((self.img_size[0], self.img_size[1]), resample=Image.NEAREST)
        lbl = torch.from_numpy(np.array(lbl)).long()
        img = transforms.ToTensor()(img)
        return img, lbl

def main(data_path, args, resume, max_data=0):
    uncertainty_portion = args['uncertainty_portion']
    exp_path = args['exp_dir']
    max_data = args['num_synth_train']

    base_path = os.path.join(exp_path, 
                    "deeplab_class_%d_checkpoint_%d_filter_out_%f_merge_classes_%s_num_train_%0d" % 
                    (
                        num_test_class,
                        int(max_data),
                        uncertainty_portion,
                        str(args['merge_classes']),
                        max_data
                    )
    )

    if not os.path.exists(base_path):
        os.mkdir(base_path)

    print("Model dir,", base_path)
    num_test_class = args['testing_data_number_class']

    dump_data = []
    all_pickle = glob.glob(data_path + '/*.pickle')

    used_image = []
    for p in all_pickle:

        with open(p, 'rb') as f:
            curr_dict = pickle.load(f)

        for dd in curr_dict:
            if not dd['image_name'] in used_image:
                used_image.append(dd['image_name'] )
                dump_data.append(dd)
    if max_data > 0:
        dump_data = dump_data[:max_data]

    stylegan_images = [data['image_name'] for data in dump_data]
    stylegan_labels = [data['image_label_name'] for data in dump_data]

    stylegan_images.sort()
    stylegan_labels.sort()
    if uncertainty_portion > 0:
        sort_by_uncertainty = sorted(dump_data, key=lambda k: k['uncertrainty_score'])
        filter_out_num = int(len(sort_by_uncertainty) * uncertainty_portion)
        sort_by_uncertainty = sort_by_uncertainty[30:-filter_out_num+ 30]
        out_idx = range(len(sort_by_uncertainty))
        stylegan_images = [sort_by_uncertainty[idx]['image_name'] for idx in out_idx]
        stylegan_labels = [sort_by_uncertainty[idx]['image_label_name'] for idx in out_idx]

    if args['number_class'] ==  num_test_class:
        trans_method = trans_mask
    else:
        # Need to merge labels based on testing set.
        if 'car' in args['category']:
            if num_test_class == 12:
                trans_method = trans_mask_stylegan_20classTo12
        elif args['category'] == 'face' and args['number_class'] == 19:
            if num_test_class == 8:
                trans_method = trans_mask_stylegan_19classTo8    

    assert  len(stylegan_images) == len(stylegan_labels)
    print( "Train data length,", str(len(stylegan_labels)))

    train_data = ImageLabelDataset(
        img_path_list = stylegan_images,
        label_path_list = stylegan_labels, 
        trans = trans_method,
        img_size=(args['deeplab_res'], args['deeplab_res'])
    )

    train_data = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=16)
    classifier = torchvision.models.segmentation.deeplabv3_resnet101(
        pretrained=False, 
        progress=False,
        num_test_classes=num_test_class, 
        aux_loss=None
    )

    if resume != "":
        checkpoint = torch.load(resume)
        classifier.load_state_dict(checkpoint['model_state_dict'])

    classifier.cuda()
    classifier.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    resnet_transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    for epoch in range(20):
        for i, da, in enumerate(train_data):
            if da[0].shape[0] != 8:
                continue
            if i % 10 == 0:
                gc.collect()

            classifier.train()

            optimizer.zero_grad()
            img, mask = da[0], da[1]

            img = img.cuda()
            mask = mask.cuda()

            input_img_tensor = []
            for b in range(img.size(0)):
                if img.size(1) == 4:
                    input_img_tensor.append(resnet_transform(img[b][:-1,:,:]))
                else:
                    input_img_tensor.append(resnet_transform(img[b]))

            input_img_tensor = torch.stack(input_img_tensor)

            y_pred = classifier(input_img_tensor)['out']

            loss = criterion(y_pred, mask)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(epoch, 'epoch', 'iteration', i, 'loss', loss.item())

        model_path = os.path.join(base_path, 'deeplab_epoch_' + str(epoch) + '.pth')

        print('Save to:', model_path)
        torch.save({'model_state_dict': classifier.state_dict()},
                   model_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--resume', type=str,  default="")
    parser.add_argument('--max_data', type=int,  default=0)
    args = parser.parse_args()

    opts = json.load(open(args.exp, 'r'))
    path =opts['exp_dir']
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))
    
    opts['merge_classes'] = (opts['testing_data_number_class'] != opts['number_class'])

    main(args.data_path, opts, args.resume, args.max_data)
