# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import torch
torch.manual_seed(0)
device_ids = [0]
from PIL import Image
import re
import base64
import cv2

class trainData(Dataset):

    def __init__(self, X_data, load_raw=False, metas=[], raw_images=[], img_size=(512, 512), fill_blank=True):
        self.X_data = X_data
        self.img_size = img_size
        self.load_raw = load_raw
        self.metas = metas
        self.raw_images = raw_images
        self.fill_blank = fill_blank

    def __getitem__(self, index):

        width = self.img_size[1]
        height =  self.img_size[0]

        img, image_id = load_one_image_for_embedding(self.X_data[index], self.img_size, self.fill_blank)

        if self.load_raw:
            meta = self.metas[index]
            raw_image = self.raw_images[index]
            raw_image = Image.open(raw_image)
            raw_image = raw_image.resize((width, height), Image.ANTIALIAS)
            raw_image = transforms.ToTensor()(raw_image)
            return img, image_id, meta, raw_image

        else:
            return img, image_id

    def __len__(self):
        return len(self.X_data)


def load_one_image_for_embedding(im_path, img_size, fill_blank=True):
    width = img_size[1]
    height = img_size[0]
    image_id = im_path.split("/")[-1].split(".")[0]
    img = Image.open(im_path)
    img = np.asarray(img)
    img = img[:, :, :3]
    img = Image.fromarray(img, 'RGB')
    if width != height and fill_blank:
        img = img.resize((width, height), Image.ANTIALIAS)
        img = np.asarray(img)
        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2: (width + height) // 2, :, :] = img
        canvas = Image.fromarray(canvas, 'RGB')
    else:
        canvas = img.resize((img_size[1], img_size[0]))  # uint8 with RGB mode

    img = transforms.ToTensor()(canvas)
    return img, image_id


def crop2fullImg(crop_im, bbox, org_im, im_size=None,
                      interpolation=cv2.INTER_CUBIC):
    im_si = im_size
    bounds = (0, 0, im_si[1] - 1, im_si[0] - 1)
    # Valid bounding box locations as (x_min, y_min, x_max, y_max)
    bbox_valid = (max(bbox[0], bounds[0]),
                  max(bbox[1], bounds[1]),
                  min(bbox[2], bounds[2]),
                  min(bbox[3], bounds[3]))
    offsets = (-bbox_valid[0], -bbox_valid[1])
    inds = tuple(map(sum, zip(bbox_valid, offsets + offsets)))
    crop_mask = cv2.resize(crop_im, (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1), interpolation=interpolation)
    result_ = org_im
    result_[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1] = \
        crop_mask[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1]
    return result_

def crop_from_bbox(img, bbox, square=True, padding=30):

    # Borders of image
    bounds = (0, 0, img.shape[1] - 1, img.shape[0] - 1)
    if square:
        max_hw =  max(bbox[3] - bbox[1] ,bbox[2] - bbox[0] ) + padding
        center = [ (bbox[2] + bbox[0]) /2 , (bbox[1] + bbox[3]) /2 ]
        bbox = [int(center[0] -  max_hw / 2 ), int(center[1] -  max_hw / 2),  int(center[0]   +  max_hw / 2 ),int( center[1]  +  max_hw / 2)]
    # Valid bounding box locations as (x_min, y_min, x_max, y_max)
    bbox_valid = (max(bbox[0] , bounds[0]),
                  max(bbox[1] , bounds[1]),
                  min(bbox[2] , bounds[2]),
                  min(bbox[3]  , bounds[3]))
    crop = np.zeros((bbox_valid[3] - bbox_valid[1] + 1, bbox_valid[2] - bbox_valid[0] + 1), dtype=img.dtype)
    offsets = (-bbox_valid[0], -bbox_valid[1])
    # Simple per element addition in the tuple
    inds = tuple(map(sum, zip(bbox_valid, offsets + offsets)))
    img = np.squeeze(img)
    if img.ndim == 2:
        crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1] = \
            img[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1]
    else:
        crop = np.tile(crop[:, :, np.newaxis], [1, 1, 3])  # Add 3 RGB Channels
        crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1, :] = \
            img[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1, :]



    return crop,bbox_valid

def colorize_mask(mask, palette):
    # mask: numpy array of the mask

    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return np.array(new_mask.convert('RGB'))

def process_image(images):
    drange = [-1, 1]
    scale = 255 / (drange[1] - drange[0])
    images = images * scale + (0.5 - drange[0] * scale)

    images = images.astype(int)
    images[images > 255] = 255
    images[images < 0] = 0

    return images.astype(int)

def decode_base64(data, altchars=b'+/'):
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.

    """
    data = re.sub(rb'[^a-zA-Z0-9%s]+' % altchars, b'', data)  # normalize
    missing_padding = len(data) % 4
    if missing_padding:
        data += b'='* (4 - missing_padding)
    return base64.b64decode(data, altchars)


def oht_to_scalar(y_pred):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    return y_pred_tags



def get_label_stas(data_loader):
    count_dict = {}
    for i in range(data_loader.__len__()):
        x, y = data_loader.__getitem__(i)
        if int(y.item()) not in count_dict:
            count_dict[int(y.item())] = 1
        else:
            count_dict[int(y.item())] += 1

    return count_dict

def depth_metrics(y_pred, y_true):
    nmse = np.linalg.norm(y_pred[y_true > 0] - y_true[y_true > 0])**2 / np.linalg.norm(y_true[y_true > 0])**2

    #clamp both y_pred and y_true to be between 1e-3 and 80
    y_pred[y_pred < 1e-3] = 1e-3
    y_pred[y_pred > 80] = 80

    y_true[y_true < 1e-3] = 1e-3
    y_true[y_true > 80] = 80

    rmse = (y_pred - y_true)**2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(y_true) - np.log(y_pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(y_true - y_pred) / y_true)

    sq_rel = np.mean(((y_true - y_pred) ** 2) / y_true)

    return nmse, rmse, rmse_log, abs_rel, sq_rel




    



# missing: beard, cupid bow
face_class_34 = ['background', 'head', 'head***cheek', 'head***chin',
              'head***ear', 'head***ear***helix',  'head***ear***lobule',
              'head***eye***botton lid', 'head***eye***eyelashes', 'head***eye***iris',
              'head***eye***pupil', 'head***eye***sclera', 'head***eye***tear duct', 'head***eye***top lid',
              'head***eyebrow', 'head***forehead', 'head***frown', 'head***hair', 'head***hair***sideburns',
              'head***jaw', 'head***moustache', 'head***mouth***inferior lip', 'head***mouth***oral comisure',
              'head***mouth***superior lip', 'head***mouth***teeth', 'head***neck', 'head***nose',
              'head***nose***ala of nose', 'head***nose***bridge', 'head***nose***nose tip', 'head***nose***nostril',
              'head***philtrum', 'head***temple', 'head***wrinkles']

face_class = ['background', 'skin', 'nose', 'glasses', 'left_eye', 'right_eye', 'left_brow', 'right_brow',
                'left_ear', 'right_ear', 'mouth', 'upper_lip', 'lower_lip', 'hair', 'hat', 'earring', 
                'neck_l', 'neck', 'clothing']

face_8_class = ['background', 'skin', 'nose', 'eye', 'brow', 'ear', 'mouth', 'hair']
face_8_class_hat = ['background', 'skin', 'nose', 'eye', 'brow', 'ear', 'mouth', 'hair', 'hat']
face_8_class_glasses = ['background', 'skin', 'nose', 'eye', 'brow', 'ear', 'mouth', 'hair', 'glasses']
face_8_class_necklace = ['background', 'skin', 'nose', 'eye', 'brow', 'ear', 'mouth', 'hair', 'necklace']
face_10_class = ['background', 'skin', 'nose', 'eye', 'brow', 'ear', 'mouth', 'hair', 'hat', 'glasses']

car_12_class = ['background', 'car_body', 'head light', 'tail light', 'licence plate',
                'wind shield', 'wheel', 'door', 'handle' , 'wheelhub', 'window', 'mirror']
car_20_class = ['background', 'back_bumper', 'bumper', 'car_body', 'car_lights', 'door', 'fender','grilles','handles',
                'hoods', 'licensePlate', 'mirror','roof', 'running_boards', 'tailLight','tire', 'trunk_lids','wheelhub', 'window', 'windshield']



car_32_platette_hex = ['#ffffff','#eee566','#000000','#7c6322','#c17f0f','#6ab115',  '#f8d52a','#fc9b53','#dc934d',
                           '#635303','#74748a', '#3fb618', '#c8e225',  '#e1b8a1', '#e905db',  '#8eacf8', '#997092','#2670fe',
                            '#e51e8d','#73d083', '#345354', '#e53f6e', '#c2577d', '#e16012', '#498be2', '#ac8f10',
                            '#a9656f','#1f66d3', '#688365','#46a89c','#b7f2d1','#48b8e2']
car_20_palette =[ 255,  255,  255, # 0 background
  238,  229,  102,# 1 back_bumper
  0, 0, 0,# 2 bumper
  124,  99 , 34, # 3 car
  193 , 127,  15,# 4 car_lights
  248  ,213 , 42, # 5 door
  220  ,147 , 77, # 6 fender
  99 , 83  , 3, # 7 grilles
  116 , 116 , 138,  # 8 handles
  200  ,226 , 37, # 9 hoods
  225 , 184 , 161, # 10 licensePlate
  142 , 172  ,248, # 11 mirror
  153 , 112 , 146, # 12 roof
  38  ,112 , 254, # 13 running_boards
  229 , 30  ,141, # 14 tailLight
  52 , 83  ,84, # 15 tire
  194 , 87 , 125, # 16 trunk_lids
  225,  96  ,18,  # 17 wheelhub
  31 , 102 , 211, # 18 window
  104 , 131 , 101# 19 windshield
         ]



face_palette = [  1.0000,  1.0000 , 1.0000,
              0.4420,  0.5100 , 0.4234,
              0.8562,  0.9537 , 0.3188,
              0.2405,  0.4699 , 0.9918,
              0.8434,  0.9329  ,0.7544,
              0.3748,  0.7917 , 0.3256,
              0.0190,  0.4943 , 0.3782,
              0.7461 , 0.0137 , 0.5684,
              0.1644,  0.2402 , 0.7324,
              0.0200 , 0.4379 , 0.4100,
              0.5853 , 0.8880 , 0.6137,
              0.7991 , 0.9132 , 0.9720,
              0.6816 , 0.6237  ,0.8562,
              0.9981 , 0.4692 , 0.3849,
              0.5351 , 0.8242 , 0.2731,
              0.1747 , 0.3626 , 0.8345,
              0.5323 , 0.6668 , 0.4922,
              0.2122 , 0.3483 , 0.4707,
              0.6844,  0.1238 , 0.1452,
              0.3882 , 0.4664 , 0.1003,
              0.2296,  0.0401 , 0.3030,
              0.5751 , 0.5467 , 0.9835,
              0.1308 , 0.9628,  0.0777,
              0.2849  ,0.1846 , 0.2625,
              0.9764 , 0.9420 , 0.6628,
              0.3893 , 0.4456 , 0.6433,
              0.8705 , 0.3957 , 0.0963,
              0.6117 , 0.9702 , 0.0247,
              0.3668 , 0.6694 , 0.3117,
              0.6451 , 0.7302,  0.9542,
              0.6171 , 0.1097,  0.9053,
              0.3377 , 0.4950,  0.7284,
              0.1655,  0.9254,  0.6557,
              0.9450  ,0.6721,  0.6162]


face_palette_extra = [  1.0000,  1.0000 , 1.0000,
              0.4420,  0.5100 , 0.4234,
              0.8562,  0.9537 , 0.3188,
              0.2405,  0.4699 , 0.9918,
              0.8434,  0.9329  ,0.7544,
              0.3748,  0.7917 , 0.3256,
              0.0190,  0.4943 , 0.3782,
              0.7461 , 0.0137 , 0.5684,
              0.1644,  0.2402 , 0.7324,
              0.0200 , 0.4379 , 0.4100,
              0.5853 , 0.8880 , 0.6137,
              0.7991 , 0.9132 , 0.9720,
              0.6816 , 0.6237  ,0.8562,
              0.9981 , 0.4692 , 0.3849,
              0.5351 , 0.8242 , 0.2731,
              0.1747 , 0.3626 , 0.8345,
              0.5323 , 0.6668 , 0.4922,
              0.2122 , 0.3483 , 0.4707,
              0.6844,  0.1238 , 0.1452,
              0.3882 , 0.4664 , 0.1003,
              0.2296,  0.0401 , 0.3030,
              0.5751 , 0.5467 , 0.9835,
              0.1308 , 0.9628,  0.0777,
              0.2849  ,0.1846 , 0.2625,
              0.9764 , 0.9420 , 0.6628,
              0.3893 , 0.4456 , 0.6433,
              0.8705 , 0.3957 , 0.0963,
              0.6117 , 0.9702 , 0.0247,
              0.3668 , 0.6694 , 0.3117,
              0.6451 , 0.7302,  0.9542,
              0.6171 , 0.1097,  0.9053,
              0.3377 , 0.4950,  0.7284,
              0.1655,  0.9254,  0.6557,
              0.9450  ,0.6721,  0.6162,
                  0.3882, 0.4664, 0.1003,
                  0.2296, 0.0401, 0.3030,
                  0.5751, 0.5467, 0.9835,
                  0.1308, 0.9628, 0.0777,
                  0.2849, 0.1846, 0.2625,
                  0.9764, 0.9420, 0.6628,
                  0.3893, 0.4456, 0.6433,
                  0.8705, 0.3957, 0.0963,
                  0.6117, 0.9702, 0.0247,
                  0.3668, 0.6694, 0.3117,
                  0.6451, 0.7302, 0.9542,
                  0.6171, 0.1097, 0.9053,
                  0.3377, 0.4950, 0.7284,
                  0.1655, 0.9254, 0.6557,
                  0.9450, 0.6721, 0.6162,
                0.4420,  0.5100 , 0.4234,
              0.8562,  0.9537 , 0.3188,
              0.2405,  0.4699 , 0.9918,
              0.8434,  0.9329  ,0.7544,
              0.3748,  0.7917 , 0.3256,
              0.0190,  0.4943 , 0.3782,
              0.7461 , 0.0137 , 0.5684,
              0.1644,  0.2402 , 0.7324,
              0.0200 , 0.4379 , 0.4100,
              0.5853 , 0.8880 , 0.6137,
              0.7991 , 0.9132 , 0.9720,
              0.6816 , 0.6237  ,0.8562,
              0.9981 , 0.4692 , 0.3849,
              0.5351 , 0.8242 , 0.2731,
              0.1747 , 0.3626 , 0.8345,
              0.5323 , 0.6668 , 0.4922,
              0.2122 , 0.3483 , 0.4707,
              0.6844,  0.1238 , 0.1452,
              0.3882 , 0.4664 , 0.1003,
              0.2296,  0.0401 , 0.3030,
              0.5751 , 0.5467 , 0.9835,
              0.1308 , 0.9628,  0.0777,
              0.2849  ,0.1846 , 0.2625,
              0.9764 , 0.9420 , 0.6628,
              0.3893 , 0.4456 , 0.6433,
              0.8705 , 0.3957 , 0.0963,
              0.6117 , 0.9702 , 0.0247,
              0.3668 , 0.6694 , 0.3117,
              0.6451 , 0.7302,  0.9542,
              0.6171 , 0.1097,  0.9053,
              0.3377 , 0.4950,  0.7284,
              0.1655,  0.9254,  0.6557,
              0.9450  ,0.6721,  0.6162,
                  0.3882, 0.4664, 0.1003,
                  0.2296, 0.0401, 0.3030,
                  0.5751, 0.5467, 0.9835,
                  0.1308, 0.9628, 0.0777,
                  0.2849, 0.1846, 0.2625,
                  0.9764, 0.9420, 0.6628,
                  0.3893, 0.4456, 0.6433,
                  0.8705, 0.3957, 0.0963,
                  0.6117, 0.9702, 0.0247,
                  0.3668, 0.6694, 0.3117,
                  0.6451, 0.7302, 0.9542,
                  0.6171, 0.1097, 0.9053,
                  0.3377, 0.4950, 0.7284,
                  0.1655, 0.9254, 0.6557,
                  0.9450, 0.6721, 0.6162
                  ]

face_8_palette = face_palette[:24]

face_palette_extra = [int(item * 255) for item in face_palette_extra
                      ]
face_palette = [int(item * 255) for item in face_palette]

face_8_palette = [int(item * 255) for item in face_8_palette]





car_12_palette =[ 255,  255,  255, # 0 background
         124,  99 , 34, # 3 car
         193 , 127,  15,# 4 car_lights
         229 , 30  ,141, # 14 tailLight
        225 , 184 , 161, # 10 licensePlate
        104 , 131 , 101,# 19 windshield
        52 , 83  ,84, # 15 tire
        248  ,213 , 42, # 5 door
         116 , 116 , 138,  # 8 handles
           225,  96  ,18,  # 17 wheelhub
         31 , 102 , 211, # 18 window
         142 , 172  ,248, # 11 mirror
         ]



car_32_palette =[ 255,  255,  255,
  238,  229,  102,
  0, 0, 0,
  124,  99 , 34,
  193 , 127,  15,
  106,  177,  21,
  248  ,213 , 42,
  252 , 155,  83,
  220  ,147 , 77,
  99 , 83  , 3,
  116 , 116 , 138,
  63  ,182 , 24,
  200  ,226 , 37,
  225 , 184 , 161,
  233 ,  5  ,219,
  142 , 172  ,248,
  153 , 112 , 146,
  38  ,112 , 254,
  229 , 30  ,141,
  115  ,208 , 131,
  52 , 83  ,84,
  229 , 63 , 110,
  194 , 87 , 125,
  225,  96  ,18,
  73  ,139,  226,
  172 , 143 , 16,
  169 , 101 , 111,
  31 , 102 , 211,
  104 , 131 , 101,
  70  ,168  ,156,
  183 , 242 , 209,
  72  ,184 , 226]

bird_palette =  [255,255,255,
            220, 220, 0,
           220, 20, 60,
           190, 153, 153,
            250, 170, 30,
           220, 220, 0,
           107, 142, 35,
           102, 102, 156,
           152, 251, 152,
           119, 11, 32,
           244, 35, 232
          ]


bedroom_palette =[ 255,  255,  255,
  238,  229,  102,
  255, 72, 69,
  124,  99 , 34,
  193 , 127,  15,
  106,  177,  21,
  248  ,213 , 42,
  252 , 155,  83,
  220  ,147 , 77,
  99 , 83  , 3,
  116 , 116 , 138,
  63  ,182 , 24,
  200  ,226 , 37,
  225 , 184 , 161,
  233 ,  5  ,219,
  142 , 172  ,248,
  153 , 112 , 146,
  38  ,112 , 254,
  229 , 30  ,141,
   238, 229, 12,
   255, 72, 6,
   124, 9, 34,
   193, 17, 15,
   106, 17, 21,
   28, 213, 2,
   252, 155, 3,
   20, 147, 77,
   9, 83, 3,
   11, 16, 138,
   6, 12, 24,
   20, 22, 37,
   225, 14, 16,
   23, 5, 29,
   14, 12, 28,
   15, 11, 16,
   3, 12, 24,
   22, 3, 11
   ]


def trans_mask(mask):
    return mask

def trans_mask_stylegan_20classTo12(mask):
    final_mask = np.zeros(mask.shape)
    final_mask[(mask != 0)] = 1 # car
    final_mask[(mask == 4)] = 2 # head light
    final_mask[(mask == 14)] = 5 # tail light
    final_mask[(mask == 10)] = 3 # licence plate
    final_mask[ (mask == 19)] = 8 # wind shield
    final_mask[(mask == 15)] = 6 # wheel
    final_mask[(mask == 5)] = 9 # door
    final_mask[(mask == 8)] = 10 # handle
    final_mask[(mask == 17)] = 11 # wheelhub
    final_mask[(mask == 18)] = 7 # window
    final_mask[(mask == 11)] = 4 # mirror
    return final_mask

def trans_mask_stylegan_34classTo8(mask):
    mask_temp = np.zeros(mask.shape)
    
    #mask_temp[(mask == 25)] = 0 #neck -> background


    mask_temp[(mask == 1)] = 1 #head -> skin
    mask_temp[(mask == 2)] = 1 #cheek -> skin
    mask_temp[(mask == 3)] = 1 #chin -> skin
    mask_temp[(mask == 15)] = 1 #forehead -> skin
    mask_temp[(mask == 16)] = 1 #frown -> skin
    mask_temp[(mask == 19)] = 1 #jaw -> skin
    mask_temp[(mask == 20)] = 1 #mustache -> skin
    mask_temp[(mask == 31)] = 1 #philtrum -> skin
    mask_temp[(mask == 32)] = 1 #temples -> skin
    mask_temp[(mask == 33)] = 1 # wrinkles -> skin
    
    mask_temp[(mask == 26)] = 2 #nose -> nose
    mask_temp[(mask == 27)] = 2 #ala nose -> nose
    mask_temp[(mask == 28)] = 2 #bridgenose -> nose
    mask_temp[(mask == 29)] = 2 #tip nose -> nose
    mask_temp[(mask == 30)] = 2 #nostril nose -> nose

    mask_temp[(mask == 7)] = 3 #eye bottom -> eye
    mask_temp[(mask == 8)] = 3 #eyelashes -> eye
    mask_temp[(mask == 9)] = 3 #iris -> eye
    mask_temp[(mask == 10)] = 3 #pupil -> eye
    mask_temp[(mask == 11)] = 3 #sclera -> eye
    mask_temp[(mask == 12)] = 3 #tear duct -> eye
    mask_temp[(mask == 13)] = 3 #top lid -> eye

    mask_temp[(mask == 14)] = 4 #eyebrow -> eyebrow

    mask_temp[(mask == 4)] = 5 #ear -> ear
    mask_temp[(mask == 5)] = 5 #helix -> ear
    mask_temp[(mask == 6)] = 5 #lobule -> ear

    mask_temp[(mask == 21)] = 6 #l_lip -> mouth
    mask_temp[(mask == 22)] = 6 #mouth -> mouth
    mask_temp[(mask == 23)] = 6 #u_lip -> mouth
    mask_temp[(mask == 24)] = 6 #teeth -> mouth

    mask_temp[(mask == 17)] = 7 #hair -> hair
    mask_temp[(mask == 18)] = 7 #sideburns -> hair

    return mask_temp
    




def trans_mask_stylegan_19classTo9_hat(mask):
    mask_temp = np.zeros(mask.shape)

    #mask_temp[mask == 0] = 0 
    #mask_temp[mask == 15] = 0 #earring -> background
    #mask_temp[mask == 16] = 0 #necklace -> background
    #mask_temp[mask == 17] = 0 #neck -> background
    #mask_temp[mask == 18] = 0 #clothes -> background
    
    mask_temp[mask == 1] = 1 #skin -> skin
    mask_temp[mask == 2] = 2 #nose -> nose
    

    mask_temp[mask == 3] = 3 #glasses -> eye
    mask_temp[mask == 4] = 3 #l_eye -> eye
    mask_temp[mask == 5] = 3 #r_eye -> eye
    
    
    mask_temp[mask == 6] = 4 #l_brow -> brow
    mask_temp[mask == 7] = 4 #r_brow -> brow
    
    mask_temp[mask == 8] = 5 #l_ear -> ear
    mask_temp[mask == 9] = 5 #r_ear -> ear
    
    mask_temp[mask == 10] = 6 #mouth -> mouth
    mask_temp[mask == 11] = 6 #u_lip -> mouth
    mask_temp[mask == 12] = 6 #l_lip -> mouth
    
    mask_temp[mask == 13] = 7 #hair -> hair

    mask_temp[mask == 14] = 8 #hat -> hat

    return mask_temp

def trans_mask_stylegan_19classTo9_glasses(mask):
    mask_temp = np.zeros(mask.shape)

    #mask_temp[mask == 0] = 0 
    #mask_temp[mask == 14] = 0 #hat -> background
    #mask_temp[mask == 15] = 0 #earring -> background
    #mask_temp[mask == 16] = 0 #necklace -> background
    #mask_temp[mask == 17] = 0 #neck -> background
    #mask_temp[mask == 18] = 0 #clothes -> background
    
    mask_temp[mask == 1] = 1 #skin -> skin
    mask_temp[mask == 2] = 2 #nose -> nose
    

    #mask_temp[mask == 3] = 3 #glasses -> eye
    mask_temp[mask == 4] = 3 #l_eye -> eye
    mask_temp[mask == 5] = 3 #r_eye -> eye
    
    
    mask_temp[mask == 6] = 4 #l_brow -> brow
    mask_temp[mask == 7] = 4 #r_brow -> brow
    
    mask_temp[mask == 8] = 5 #l_ear -> ear
    mask_temp[mask == 9] = 5 #r_ear -> ear
    
    mask_temp[mask == 10] = 6 #mouth -> mouth
    mask_temp[mask == 11] = 6 #u_lip -> mouth
    mask_temp[mask == 12] = 6 #l_lip -> mouth
    
    mask_temp[mask == 13] = 7 #hair -> hair

    mask_temp[mask == 3] = 8 #glasses -> glasses

    return mask_temp

def trans_mask_stylegan_19classTo10(mask):
    mask_temp = np.zeros(mask.shape)

    #mask_temp[mask == 0] = 0 
    #mask_temp[mask == 15] = 0 #earring -> background
    #mask_temp[mask == 16] = 0 #necklace -> background
    #mask_temp[mask == 17] = 0 #neck -> background
    #mask_temp[mask == 18] = 0 #clothes -> background
    #mask_temp[mask == 16] = 0 #necklace -> background
    
    mask_temp[mask == 1] = 1 #skin -> skin
    mask_temp[mask == 2] = 2 #nose -> nose
    
    mask_temp[mask == 4] = 3 #l_eye -> eye
    mask_temp[mask == 5] = 3 #r_eye -> eye
    
    
    mask_temp[mask == 6] = 4 #l_brow -> brow
    mask_temp[mask == 7] = 4 #r_brow -> brow
    
    mask_temp[mask == 8] = 5 #l_ear -> ear
    mask_temp[mask == 9] = 5 #r_ear -> ear
    
    mask_temp[mask == 10] = 6 #mouth -> mouth
    mask_temp[mask == 11] = 6 #u_lip -> mouth
    mask_temp[mask == 12] = 6 #l_lip -> mouth
    
    mask_temp[mask == 13] = 7 #hair -> hair
    
    mask_temp[mask == 14] = 8 #hat -> hat

    mask_temp[mask == 3] = 9 #glasses -> glasses
    

    return mask_temp

def trans_mask_stylegan_19classTo8(mask):
    mask_temp = np.zeros(mask.shape)

    #mask_temp[mask == 0] = 0 
    #mask_temp[mask == 14] = 0 #hat -> background
    #mask_temp[mask == 15] = 0 #earring -> background
    #mask_temp[mask == 16] = 0 #necklace -> background
    #mask_temp[mask == 17] = 0 #neck -> background
    #mask_temp[mask == 18] = 0 #clothes -> background
    
    mask_temp[mask == 1] = 1 #skin -> skin
    mask_temp[mask == 2] = 2 #nose -> nose
    

    mask_temp[mask == 3] = 3 #glasses -> eye
    mask_temp[mask == 4] = 3 #l_eye -> eye
    mask_temp[mask == 5] = 3 #r_eye -> eye
    
    
    mask_temp[mask == 6] = 4 #l_brow -> brow
    mask_temp[mask == 7] = 4 #r_brow -> brow
    
    mask_temp[mask == 8] = 5 #l_ear -> ear
    mask_temp[mask == 9] = 5 #r_ear -> ear
    
    mask_temp[mask == 10] = 6 #mouth -> mouth
    mask_temp[mask == 11] = 6 #u_lip -> mouth
    mask_temp[mask == 12] = 6 #l_lip -> mouth
    
    mask_temp[mask == 13] = 7 #hair -> hair

    return mask_temp

'''
1	: _background_
2	: back_bumper
3	: back_glass
4	: back_left_door
5	: back_left_light
6	: back_right_door
7	: back_right_light
8	: front_bumper
9	: front_glass
10	: front_left_door
11	: front_left_light
12	: front_right_door
13	: front_right_light
14	: hood
15	: left_mirror
16	: right_mirror
17	: tailgate
18	: trunk
19	: wheel
'''

car_10_class = ['background', 'bumper', 'back window', 'door', 'light', 'windshield', 'hood', 'mirror', 'trunk', 'wheel']

def trans_mask_car_19classTo10(mask):
    mask_temp = np.zeros(mask.shape)

    mask_temp[mask==1] = 1 #back bumper -> bumper
    mask_temp[mask==7] = 1 #front bumper -> bumper

    mask_temp[mask==2] = 2 #back window
    
    mask_temp[mask==3] = 3 #back left door -> door
    mask_temp[mask==5] = 3 #back right door -> door
    mask_temp[mask==9] = 3 #front left door -> door
    mask_temp[mask==11] = 3 #front right door -> door

    mask_temp[mask==4] = 4 #back left light -> light
    mask_temp[mask==6] = 4 #back right light -> light
    mask_temp[mask==10] = 4 #front left light -> light
    mask_temp[mask==12] = 4 #front right light -> light

    mask_temp[mask==8] = 5 #front glass

    mask_temp[mask==13] = 6 #hood

    mask_temp[mask==14] = 7 #left mirror -> mirror
    mask_temp[mask==15] = 7 #right mirror -> mirror

    mask_temp[mask==16] = 8 #tailgate -> back of car
    mask_temp[mask==17] = 8 #trunk -> back of car

    mask_temp[mask==18] = 9 #wheel

    return mask_temp

car_11_class = ['background', 'front bumper', 'back window', 'door', 'light', 'windshield', 'hood', 'mirror', 'trunk', 'wheel', 'back bumper']


def trans_mask_car_19classTo11_bumper(mask):
    mask_temp = np.zeros(mask.shape)

    #mask_temp[mask==1] = 1 #back bumper -> bumper
    mask_temp[mask==7] = 1 #front bumper -> bumper

    mask_temp[mask==2] = 2 #back window
    
    mask_temp[mask==3] = 3 #back left door -> door
    mask_temp[mask==5] = 3 #back right door -> door
    mask_temp[mask==9] = 3 #front left door -> door
    mask_temp[mask==11] = 3 #front right door -> door

    mask_temp[mask==4] = 4 #back left light -> light
    mask_temp[mask==6] = 4 #back right light -> light
    mask_temp[mask==10] = 4 #front left light -> light
    mask_temp[mask==12] = 4 #front right light -> light

    mask_temp[mask==8] = 5 #front glass

    mask_temp[mask==13] = 6 #hood

    mask_temp[mask==14] = 7 #left mirror -> mirror
    mask_temp[mask==15] = 7 #right mirror -> mirror

    mask_temp[mask==16] = 8 #tailgate -> back of car
    mask_temp[mask==17] = 8 #trunk -> back of car

    mask_temp[mask==18] = 9 #wheel

    mask_temp[mask==1] = 10 #back bumper, long tail

    return mask_temp


cityscapes_class = ['void', 'road', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle']

def trans_mask_cityscapes_35_8(mask):
    mask_temp = np.zeros(mask.shape)
    #0: void: 0-6, 8, 9, 10
    vals_0 = list(range(7)) + [8,9,10]
    mask_temp[np.isin(mask, vals_0)] = 0
    
    #1: road: 7
    mask_temp[mask == 7] = 1
    
    #2: construction: 
    vals_2 = list(range(11,17))
    mask_temp[np.isin(mask, vals_2)] = 2
    
    #3: object
    vals_3 = list(range(17,21))
    mask_temp[np.isin(mask, vals_3)] = 3
    
    #4: nature
    mask_temp[np.isin(mask, [21,22])] = 4
    
    #5: sky
    mask_temp[mask == 23] = 5
    
    #6: human
    mask_temp[np.isin(mask, [24,25])] = 6
    
    #7: vehicle
    vals_7 = list(range(26,34)).append(-1)
    mask_temp[np.isin(mask, vals_7)] = 7
    
    return mask_temp
    
cityscapes_5_class = ['background', 'road', 'sky', 'human', 'vehicle']

def trans_mask_cityscapes_8_to_5(mask):

    mask_temp = np.zeros(mask.shape)
    mask_temp[mask == 0] = 0 #void -> bg
    mask_temp[mask == 3] = 0 #object -> bg
    mask_temp[mask == 2] = 0 #construction -> bg
    mask_temp[mask == 4] = 0 #nature -> bg

    mask_temp[mask == 1] = 1 #road
    mask_temp[mask == 5] = 2 #sky
    mask_temp[mask == 6] = 3 #human
    mask_temp[mask == 7] = 4 #vehicle
    
    return mask_temp
    
    

'''
def trans_mask_car_20classTo13(mask):
    mask_temp = np.zeros(mask.shape)

    mask_temp[mask==1] = 1 #back bumper
    mask_temp[mask==2] = 2 #back window
    
    mask_temp[mask==3] = 3 #back left door -> back door
    mask_temp[mask==5] = 3 #back right door -> back door

    mask_temp[mask==4] = 4 #back left light -> back light
    mask_temp[mask==6] = 4 #back right light -> back light
    
    mask_temp[mask==7] = 5 #front bumper -> front bumper

    mask_temp[mask==8] = 6 #front glass -> front glass

    mask_temp[mask==9] = 7 #front left door -> front door
    mask_temp[mask==11] = 7 #front right door -> front door

    mask_temp[mask==10] = 8 #front left light -> front light
    mask_temp[mask==12] = 8 #front right light -> front light

    mask_temp[mask==13] = 9 #hood

    mask_temp[mask==14] = 10 #left mirror -> mirror
    mask_temp[mask==15] = 10 #right mirror -> mirror

    mask_temp[mask==16] = 11 #tailgate -> back
    mask_temp[mask==17] = 11 #trunk -> back

    mask_temp[mask==18] = 12 #wheel

    return mask_temp

car_13_class = ['background', 'back bumper', 'back glass', 'back door', 'back light', 'front bumper', 'front glass', 
    'front door', 'front light', 'hood', 'mirror', 'trunk', 'wheel']


def trans_mask_car_20classTo9(mask):
    mask_temp = np.zeros(mask.shape)

    return mask_temp

def trans_mask_car_20classTo9(mask):
    mask_temp = np.zeros(mask.shape)

    mask_temp[mask==1] = 1 #back bumper -> bumper
    mask_temp[mask==7] = 1 #front bumper -> bumper

    mask_temp[mask==16] = 2 #tailgate -> back
    mask_temp[mask==17] = 2 #trunk -> back

    mask_temp[mask==2] = 3 #back glass
    
    mask_temp[mask==3] = 4 #back left door ->  door
    mask_temp[mask==5] = 4 #back right door ->  door
    mask_temp[mask==9] = 4 #front left door ->  door
    mask_temp[mask==11] = 4 #front right door ->  door

    mask_temp[mask==4] = 5 #back left light ->  light
    mask_temp[mask==6] = 5 #back right light ->  light
    mask_temp[mask==10] = 5 #front left light ->  light
    mask_temp[mask==12] = 5 #front right light ->  light
    
    mask_temp[mask==13] = 6 #hood -> hood

    mask_temp[mask==8] = 7 #front glass -> front glass

    mask_temp[mask==14] = 8 #left mirror -> mirror
    mask_temp[mask==15] = 8 #right mirror -> mirror



    mask_temp[mask==18] = 9 #wheel

    return mask_temp

car_9_class = ['background', 'back', 'back glass', 'door', 'light', 'front', 'front glass', 'mirror', 'wheel']


car_20_palette =[ 255,  255,  255, # 0 background
  238,  229,  102,# 1 back_bumper
  0, 0, 0,# 2 bumper
  124,  99 , 34, # 3 car
  193 , 127,  15,# 4 car_lights
  248  ,213 , 42, # 5 door
  220  ,147 , 77, # 6 fender
  99 , 83  , 3, # 7 grilles
  116 , 116 , 138,  # 8 handles
  200  ,226 , 37, # 9 hoods
  225 , 184 , 161, # 10 licensePlate
  142 , 172  ,248, # 11 mirror
  153 , 112 , 146, # 12 roof
  38  ,112 , 254, # 13 running_boards
  229 , 30  ,141, # 14 tailLight
  52 , 83  ,84, # 15 tire
  194 , 87 , 125, # 16 trunk_lids
  225,  96  ,18,  # 17 wheelhub
  31 , 102 , 211, # 18 window
  104 , 131 , 101# 19 windshield
         ]


def trans_mask_stylegan_car_20classTo9(mask):
    mask_temp = np.zeros(mask.shape)
    mask_temp[mask==6] = 0 #fender -> background
    mask_temp[mask==7] = 0 #grilles -> background
    mask_temp[mask==3] = 0 #car -> background
    mask_temp[mask==12] = 0 #roof -> background
    mask_temp[mask==13] = 0 #running board -> background
    

    mask_temp[mask==1] = 1
    mask_temp[mask==2] = 1

    mask_temp[mask==16] = 2
    

    mask_temp[mask==4] = 5 #lights -> lights
    mask_temp[mask==14] = 5 #taillight -> lights

    mask_temp[mask==5] = 4 #door -> door
    mask_temp[mask==8] = 4 #handles -> door
    mask_temp[mask==18] = 4 # window -> door
    
    mask_temp[mask==9] = 6 #hood -> hood

    mask_temp[mask==10] = 1 #license plate -> bumper
    
    mask_temp[mask==11] = 8 #mirror -> mirror
    

    mask_temp[mask==15] = 9
    mask_temp[mask==17] = 9

    mask_temp[mask==19] = 7 

    
    return mask_temp

'''