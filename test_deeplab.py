# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/master/datasetGAN/test_deeplab_cross_validation.py

import sys
sys.path.append('../')
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import argparse
import gc
import os

import torch
import torchvision
import torch.nn as nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torch.utils.data import Dataset, DataLoader
from train_deeplab import ImageLabelDataset

from handsoff.utils.data_utils import *

import glob
import json
import imageio
import random
import re


def cross_validate(cp_path, args):
    num_class = args['testing_data_number_class']
    
    if 'car' in args['domain']:
        from handsoff.utils.data_utils import car_20_palette as palette
        if num_class == args['number_class'] and num_class == 10:
            from handsoff.utils.data_utils import trans_mask
            from handsoff.utils.data_utils import car_10_class as class_name
            ignore_index = -1
        else:
            assert False
    elif args['domain'] == 'face':
        if args['merge_classes'] and num_class == 8:
            from handsoff.utils.data_utils import face_8_palette as palette
            from handsoff.utils.data_utils import face_8_class as class_name
            trans_mask = trans_mask_stylegan_19classTo8
        elif num_class == 19:
            from handsoff.utils.data_utils import face_palette as palette
            from handsoff.utils.data_utils import face_class as class_name
            ignore_index = -1
        else:
            assert False

    base_path = os.path.join(cp_path, "cross_validation")
    if not os.path.exists(base_path):
        os.mkdir(base_path)


    cps_all = glob.glob(cp_path + "/*")

    cp_list = [data for data in cps_all if '.pth' in data and 'BEST' not in data]
    cp_list.sort()


    ids = range(num_class)

    data_all = glob.glob(args['testing_path'] + "/*")
    images = [path for path in data_all if 'npy' not in path]
    labels = [path for path in data_all if 'npy' in path]
    images.sort()
    labels.sort()

    vis_data = ImageLabelDataset(img_path_list=images[:50],
                                  label_path_list=labels[:50], trans=trans_mask,
                                  img_size=(args['deeplab_res'], args['deeplab_res']))
    vis_data = DataLoader(vis_data, batch_size=1, shuffle=False, num_workers=0)
    vis = []
   
    for j, da, in enumerate(vis_data):
        img, mask = da[0], da[1]
        #print(np.unique(mask))

        img = img.numpy()
        img = img * 255.

        img = np.transpose(img, (0, 2, 3, 1)).astype(np.uint8)

        mask = mask.numpy()

        mask_color = colorize_mask(mask[0], palette)
        img_mask =  np.uint8(0.7 * mask_color + 0.3 * img[0])

        curr_vis = np.concatenate( [img[0], mask_color, img_mask], 0 )
        
        vis.append(curr_vis)


    vis = np.concatenate(vis, 1)
    imageio.imsave(   os.path.join(base_path, "testing.jpg"),
                      vis)

    fold_num =int( len(images) / 5)
    resnet_transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])


    classifier = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=False,
                                                                     num_classes=num_class, aux_loss=None)

    cross_mIOU = []

    for i in range(5):
        val_image = images[fold_num * i: fold_num *i + fold_num]
        val_label = labels[fold_num * i: fold_num *i + fold_num]
        test_image = [img for img in images if img not in val_image]
        test_label =[label for label in labels if label not in val_label]

        print("Number of validation datapoints: ", str(len(val_image)))
        print("Number of test datapoints: ", str(len(test_image)))

        val_data = ImageLabelDataset(
            img_path_list=val_image,
            label_path_list=val_label, trans=trans_mask,
            img_size=(args['deeplab_res'], args['deeplab_res'])
        )
        val_data = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

        test_data = ImageLabelDataset(
            img_path_list=test_image,
            label_path_list=test_label, trans=trans_mask,
            img_size=(args['deeplab_res'], args['deeplab_res'])
        )
        test_data = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)


        best_miou = 0
        best_val_miou = 0
        for resume in cp_list:

            checkpoint = torch.load(resume)
            classifier.load_state_dict(checkpoint['model_state_dict'])


            classifier.cuda()
            classifier.eval()

            unions = {}
            intersections = {}
            for target_num in ids:
                unions[target_num] = 0
                intersections[target_num] = 0

            with torch.no_grad():
                for _, da, in enumerate(val_data):

                    img, mask = da[0], da[1]

                    if img.size(1) == 4:
                        img = img[:, :-1, :, :]

                    img = img.cuda()
                    mask = mask.cuda()
                    input_img_tensor = []
                    for b in range(img.size(0)):
                        input_img_tensor.append(resnet_transform(img[b]))
                    input_img_tensor = torch.stack(input_img_tensor)

                    y_pred = classifier(input_img_tensor)['out']
                    y_pred = torch.log_softmax(y_pred, dim=1)
                    _, y_pred = torch.max(y_pred, dim=1)
                    y_pred = y_pred.cpu().detach().numpy()
                    mask = mask.cpu().detach().numpy()
                    bs = y_pred.shape[0]

                    curr_iou = []
                    if ignore_index > 0:
                        y_pred = y_pred * (mask != ignore_index)
                    for target_num in ids:
                        y_pred_tmp = (y_pred == target_num).astype(int)
                        mask_tmp = (mask == target_num).astype(int)

                        intersection = (y_pred_tmp & mask_tmp).sum()
                        union = (y_pred_tmp | mask_tmp).sum()

                        unions[target_num] += union
                        intersections[target_num] += intersection

                        if not union == 0:
                            curr_iou.append(intersection / union)
                mean_ious = []

                for target_num in ids:
                    mean_ious.append(intersections[target_num] / (1e-8 + unions[target_num]))
                mean_iou_val = np.array(mean_ious).mean()

                if mean_iou_val > best_val_miou:
                    best_val_miou = mean_iou_val
                    unions = {}
                    intersections = {}
                    for target_num in ids:
                        unions[target_num] = 0
                        intersections[target_num] = 0

                    with torch.no_grad():
                        testing_vis = []
                        for _, da, in enumerate(test_data):

                            img, mask = da[0], da[1]

                            if img.size(1) == 4:
                                img = img[:, :-1, :, :]

                            img = img.cuda()
                            mask = mask.cuda()
                            input_img_tensor = []
                            for b in range(img.size(0)):
                                input_img_tensor.append(resnet_transform(img[b]))
                            input_img_tensor = torch.stack(input_img_tensor)

                            y_pred = classifier(input_img_tensor)['out']
                            y_pred = torch.log_softmax(y_pred, dim=1)
                            _, y_pred = torch.max(y_pred, dim=1)
                            y_pred = y_pred.cpu().detach().numpy()
                            mask = mask.cpu().detach().numpy()

                            curr_iou = []
                            if ignore_index > 0:
                                y_pred = y_pred * (mask != ignore_index)
                            for target_num in ids:
                                y_pred_tmp = (y_pred == target_num).astype(int)
                                mask_tmp = (mask == target_num).astype(int)

                                intersection = (y_pred_tmp & mask_tmp).sum()
                                union = (y_pred_tmp | mask_tmp).sum()

                                unions[target_num] += union
                                intersections[target_num] += intersection

                                if not union == 0:
                                    curr_iou.append(intersection / union)


                            img = img.cpu().numpy()
                            img =  img * 255.
                            img = np.transpose(img, (0, 2, 3, 1)).astype(np.uint8)

                            curr_vis = np.concatenate([img[0], colorize_mask(y_pred[0], palette)], 0)
                            if len(testing_vis) < 50:
                                testing_vis.append(curr_vis)

                        testing_vis = np.concatenate(testing_vis, 1)
                        imageio.imsave(os.path.join(base_path, "testing_round_%d.jpg" % i), testing_vis)

                        test_mean_ious = []

                        for target_num in ids:
                            test_mean_ious.append(intersections[target_num] / (1e-8 + unions[target_num]))
                        best_test_miou = np.array(test_mean_ious).mean()


                        print("Best IOU ,", str(best_test_miou), "CP: ", resume)

        cross_mIOU.append(best_test_miou)

    print(cross_mIOU)
    print(" cross validation mean:" , np.mean(cross_mIOU) )
    print(" cross validation std:", np.std(cross_mIOU))
    result = {"Cross validation mean": np.mean(cross_mIOU), "Cross validation std": np.std(cross_mIOU), "Cross validation":cross_mIOU }
    with open(os.path.join(cp_path, 'cross_cv.json'), 'w') as f:
        json.dump(result, f)


def test(cp_path, args, validation_number=50):
    num_class = args['testing_data_number_class']
    base_path = os.path.join(cp_path, "validation")
    if args["merge_classes"]:
        base_path = os.path.join(cp_path, "validation_merged")

    cross_name = 'cross_full.json'

    if not os.path.exists(base_path):
        os.mkdir(base_path)


    if 'car' in args['domain']:
        from handsoff.utils.data_utils import car_20_palette as palette
        if num_class == args['number_class'] and num_class == 10:
            from handsoff.utils.data_utils import trans_mask
            from handsoff.utils.data_utils import car_10_class as class_name
        else:
            assert False
    elif args['domain'] == 'face':
        if args['merge_classes'] and num_class == 8:
            from handsoff.utils.data_utils import face_8_palette as palette
            from handsoff.utils.data_utils import face_8_class as class_name
            trans_mask = trans_mask_stylegan_19classTo8
        elif num_class == 19:
            from handsoff.utils.data_utils import face_palette as palette
            from handsoff.utils.data_utils import face_class as class_name
        else:
            assert False

    cps_all = glob.glob(cp_path + "/*")
    cp_list = [data for data in cps_all if '.pth' in data and 'BEST' not in data]

    ids = range(num_class)

    data_all = glob.glob(args['testing_path'] + "/*")
    images = [path for path in data_all if 'npy' not in path]
    labels = [path for path in data_all if 'npy' in path]

    images.sort()
    labels.sort()

    vis_data = ImageLabelDataset(
        img_path_list=images[:50],
        label_path_list=labels[:50], 
        trans=trans_mask,
        img_size=(args['deeplab_res'], args['deeplab_res'])
    )

    vis_data = DataLoader(vis_data, batch_size=1, shuffle=False, num_workers=0)
    vis = []
    for j, da, in enumerate(vis_data):
        img, mask = da[0], da[1]
        img = img.numpy()
        img = img * 255.
        img = np.transpose(img, (0, 2, 3, 1)).astype(np.uint8)

        mask = mask.numpy()
        curr_vis = np.concatenate( [img[0], colorize_mask(mask[0], palette)], 0 )
        vis.append(curr_vis)

    vis = np.concatenate(vis, 1)
    imageio.imsave(os.path.join(base_path, "testing_gt.jpg"), vis)


    resnet_transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    classifier = torchvision.models.segmentation.deeplabv3_resnet101(
        pretrained=False, 
        progress=False,
        num_classes=num_class, 
        aux_loss=None
    )

    cross_mIOU = []
    if validation_number == 0:
        print("Report performance on the best checkpoint")
        val_image = images
        val_label = labels
        test_image = images
        test_label = labels
    else:
        val_image = images[:validation_number]
        val_label = labels[:validation_number]
        test_image = [img for img in images if img not in val_image]
        test_label =[label for label in labels if label not in val_label]

    print("Val Data length,", str(len(val_image)))
    print("Testing Data length,", str(len(test_image)))

    val_data = ImageLabelDataset(
        img_path_list=val_image,
        label_path_list=val_label, 
        trans=trans_mask,
        img_size=(args['deeplab_res'], args['deeplab_res'])
    )
    val_data = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

    test_data = ImageLabelDataset(
        img_path_list=test_image,
        label_path_list=test_label, 
        trans=trans_mask,
        img_size=(args['deeplab_res'], args['deeplab_res'])
    )
    test_data = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    best_val_miou = 0
    cp_list.sort()

    ct = 0
    for resume in cp_list:
        print(f'Testing model {ct + 1} / {len(cp_list)}: {resume}')

        checkpoint = torch.load(resume)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier.cuda()
        classifier.eval()

        unions = {}
        intersections = {}
        for target_num in ids:
            unions[target_num] = 0
            intersections[target_num] = 0

        with torch.no_grad():
            for _, da, in enumerate(val_data):

                img, mask = da[0], da[1]

                if img.size(1) == 4:
                    img = img[:, :-1, :, :]

                img = img.cuda()
                mask = mask.cuda()
                input_img_tensor = []
                for b in range(img.size(0)):
                    input_img_tensor.append(resnet_transform(img[b]))
                input_img_tensor = torch.stack(input_img_tensor)

                y_pred = classifier(input_img_tensor)['out']
                y_pred = torch.log_softmax(y_pred, dim=1)
                _, y_pred = torch.max(y_pred, dim=1)
                y_pred = y_pred.cpu().detach().numpy()
                mask = mask.cpu().detach().numpy()
                bs = y_pred.shape[0]

                curr_iou = []

                for target_num in ids:
                    y_pred_tmp = (y_pred == target_num).astype(int)
                    mask_tmp = (mask == target_num).astype(int)

                    intersection = (y_pred_tmp & mask_tmp).sum()
                    union = (y_pred_tmp | mask_tmp).sum()

                    unions[target_num] += union
                    intersections[target_num] += intersection

                    if not union == 0:
                        curr_iou.append(intersection / union)
            mean_ious = []

            for target_num in ids:
                mean_ious.append(intersections[target_num] / (1e-8 + unions[target_num]))
            mean_iou_val = np.array(mean_ious).mean()

            if mean_iou_val > best_val_miou:
                best_val_miou = mean_iou_val
                unions = {}
                intersections = {}
                for target_num in ids:
                    unions[target_num] = 0
                    intersections[target_num] = 0

                with torch.no_grad():
                    testing_vis = []
                    for _, da, in enumerate(test_data):
                        img, mask = da[0], da[1]

                        if img.size(1) == 4:
                            img = img[:, :-1, :, :]

                        img = img.cuda()
                        mask = mask.cuda()
                        input_img_tensor = []
                        for b in range(img.size(0)):
                            input_img_tensor.append(resnet_transform(img[b]))
                        input_img_tensor = torch.stack(input_img_tensor)

                        y_pred = classifier(input_img_tensor)['out']
                        y_pred = torch.log_softmax(y_pred, dim=1)
                        _, y_pred = torch.max(y_pred, dim=1)
                        y_pred = y_pred.cpu().detach().numpy()
                        mask = mask.cpu().detach().numpy()

                        curr_iou = []

                        for target_num in ids:
                            y_pred_tmp = (y_pred == target_num).astype(int)
                            mask_tmp = (mask == target_num).astype(int)

                            intersection = (y_pred_tmp & mask_tmp).sum()
                            union = (y_pred_tmp | mask_tmp).sum()

                            unions[target_num] += union
                            intersections[target_num] += intersection

                            if not union == 0:
                                curr_iou.append(intersection / union)


                        img = img.cpu().numpy()
                        img =  img * 255.
                        img = np.transpose(img, (0, 2, 3, 1)).astype(np.uint8)

                        mask_color = colorize_mask(y_pred[0], palette)
                        img_mask =  np.uint8(0.7 * mask_color + 0.3 * img[0])
                        

                        curr_vis = np.concatenate([img[0], mask_color, img_mask], 0)
                        if len(testing_vis) < 50:
                            testing_vis.append(curr_vis)

                    testing_vis = np.concatenate(testing_vis, 1)
                    imageio.imsave(os.path.join(base_path, "testing.jpg"),
                                      testing_vis)

                    test_mean_ious = []
                    test_class_ious = []

                    for j, target_num in enumerate(ids):
                        iou = intersections[target_num] / (1e-8 + unions[target_num])
                        print("IOU for ", class_name[j], iou)

                        test_mean_ious.append(iou)
                        test_class_ious.append((class_name[j], iou))

                    best_test_miou = np.array(test_mean_ious).mean()
                    best_class_iou = test_class_ious
                    print("Best IOU ,", str(best_test_miou), "CP: ", resume)

        ct += 1
        
    print('FINAL RESULTS: ---------------------------------')
    print(cross_mIOU)
    print("Validation mIOU:" ,best_val_miou)
    print("Testing mIOU:" , best_test_miou )
    print("Testing Class IOU:", best_class_iou)

    result = {
        "Validation": best_val_miou, 
        "Testing": best_test_miou, 
        'Class': best_class_iou
    }

    with open(os.path.join(cp_path, cross_name), 'w') as f:
        json.dump(result, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str)
    parser.add_argument('--resume', type=str,  default="")
    parser.add_argument('--cross_validate', type=bool, default=False)
    parser.add_argument('--validation_number', type=int, default=0)

    args = parser.parse_args()

    opts = json.load(open(args.exp, 'r'))
    opts['merge_classes'] = (opts['testing_data_number_class'] != opts['number_class'])
    print("Opt", opts)

    path =opts['exp_dir']
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))


    if not args.cross_validate:
        test(args.resume, opts, args.validation_number)
    else:
        cross_validate(args.resume, opts)

