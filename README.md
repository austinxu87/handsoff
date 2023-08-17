# HandsOff: Labeled Dataset Generation With No Additional Human Annotations (CVPR 2023 Highlight)
> Recent work leverages the expressive power of generative adversarial networks (GANs) to generate labeled synthetic datasets. These dataset generation methods often require new annotations of synthetic images, which forces practitioners to seek out annotators, curate a set of synthetic images, and ensure the quality of generated labels. We introduce the HandsOff framework, a technique capable of producing an unlimited number of synthetic images and corresponding labels after being trained on less than 50 pre-existing labeled images. Our framework avoids the practical drawbacks of prior work by unifying the field of GAN inversion with dataset generation. We generate datasets with rich pixel-wise labels in multiple challenging domains such as faces, cars, full-body human poses, and urban driving scenes. Our method achieves state-of-the-art performance in semantic segmentation, keypoint detection, and depth estimation compared to prior dataset generation approaches and transfer learning baselines. We additionally showcase its ability to address broad challenges in model development which stem from fixed, hand-annotated datasets, such as the long-tail problem in semantic segmentation.

<a href="https://arxiv.org/abs/2212.12645"><img src="https://img.shields.io/badge/arXiv-2212.12645-b31b1b.svg" height=22.5></a>

The code is based on [EditGAN](https://github.com/nv-tlabs/editGAN_release).

## Updates
- [x] Initial code release
- [ ] Dataset split release
- [ ] Pretrained model release
- [ ] Additional domains/tasks

## Requirements
- **Note:** use `--recurse-submodules` when clone
- Alternatively, if you cloned without `--recurse-submodules`, run `git submodule update --init`
- Code is tested with CUDA 10.0 toolkit with PyTorch==1.3.1
- To set up conda environment:
```
conda env create --name handsoff_env --file requirements.yml
conda activate handsoff_env
```

## Datasets
- Faces: We train and evaluate on [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)
- Cars: We train and evaluate on [Car-Parts-Segmentation](https://github.com/dsmlr/Car-Parts-Segmentation)
- Full-body humans: We train and evaluate on a preprocessed [DeepFashion-MultiModal](https://github.com/yumingj/DeepFashion-MultiModal)
- Cityscapes: We train and evaluate on [Cityscapes](https://www.cityscapes-dataset.com)

## Data splits
- Faces:
    - Because of the CelebAMask-HQ dataset agreement, we cannot release our image splits directly. 
        - Download the image and annotation files from CelebAMask-HQ.
        - Utilize [g_mask.py](https://github.com/switchablenorms/CelebAMask-HQ/blob/master/face_parsing/Data_preprocessing/g_mask.py) provided by CelebAMask-HQ to construct segmentation masks.
        - Convert the png output of `g_mask.py` to a numpy array.
    - We map the original image numbers of CelebAMask-HQ to new image numbers based on the following mapping: [celeba_mapping](https://drive.google.com/file/d/1860THKCuktStkuCP_q71wx5e5_jlBu7d/view?usp=sharing). This `json` file has two keys:
        - `train`: a dict where the keys are the original image numbers in CelebAMask-HQ, and values are the image numbers that we use. These are the 50 images (and corresponding labels) we use to train HandsOff
        - `test`: a dict of the same structure as above.
        - Example: `celeba_mapping['train'][16569] : 0` means that 
            - `16569.jpg` in CelebAMask-HQ is `0.jpg` in the HandsOff train set
            - The segmentation mask corresponding to `16569.jpg` in CelebAMask-HQ is `image_mask0.npy` in the HandsOff train set
        - Example: `celeba_mapping['test][18698] : 29949` means that 
            - `18698.jpg` in CelebAMask-HQ is `29949.jpg` in the HandsOff test set
            - The segmentation mask corresponding to `18698.jpg` in CelebAMask-HQ is `image_mask29949.npy` in the HandsOff train set

## Pretrained models
### Pretrained GAN checkpoints
We use the following pretrained GAN checkpoints:
- Faces: [stylegan2-ffhq-config-f.pt](https://drive.google.com/file/d/1igxv6ZP4TFGe_392B-qnSqXnglTKH5yo/view?usp=share_link)
- Cars: [stylegan2_networks_stylegan2-car-config-f.pt](https://drive.google.com/file/d/1i-39ztut-VdUVUiFuUrwdsItR--HF81w/view?usp=share_link).
- Full-body humans: [stylegan_human_v2_1024.pt](https://drive.google.com/file/d/1FlAb1rYa0r_--Zj_ML8e6shmaF28hQb5/view?usp=sharing)
- Cityscapes: We contacted the authors of [this paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Gadde_Detail_Me_More_Improving_GANs_Photo-Realism_of_Complex_Scenes_ICCV_2021_paper.pdf).

### Pretrained ReStyle checkpoints
We use the following pretrained ReStyle checkpoints:
- Faces: [restyle_psp_ffhq_encode.pt](https://drive.google.com/file/d/1sw6I2lRIB0MpuJkpc8F5BJiSZrc0hjfE/view?usp=sharing)
- Cars: [restyle_e4e_cars_encode.pt](https://drive.google.com/file/d/1e2oXVeBPXMQoUoC_4TNwAWpOPpSEhE_e/view?usp=sharing)

### Pretrained Label Generators
Coming soon!

### GAN inversion latent codes
- Faces: Latent codes obtained via ReStyle and optimization refinement are located [here](https://drive.google.com/file/d/1O-VJhP5N5Rd3gabQ6OfS-ALlE2lgesRV/view?usp=sharing). 
    - :warning: These latent codes follow the number ordering of the HandsOff dataset split. See the Faces section in [Data splits](#data-splits) for our ordering.

## Training
:warning: Training HandsOff is RAM consuming, as all hypercolumn representations are kept in memory

:warning: Training HandsOff is GPU memory consuming. All experiments were run on Nvidia Tesla V100 GPUs with 32GB memory.

Examples of experimental configuration files available in `/experiments/` for face and car segmentation. More examples to come soon!

### Run GAN inversion
- Run ReStyle (or your GAN inversion method of choice)
    - Download ReStyle checkpoints or train ReStyle
```
cd restyle-encoder

python scripts/inference_iterative.py \
--exp_dir=/path/to/experiment \                             # path to output directory of ReStyle
--checkpoint_path=experiment/checkpoints/best_model.pt \    # pretrained ReStyle checkpoint path
--data_path=/path/to/test_data \                            # path to images to invert
--test_batch_size=4 \                                   
--test_workers=4 \
--n_iters_per_batch=5

cd ..
```
- Convert ReStyle outputs to format used to train label generator
```
python format_latents.py \
--latents_dir=/exp_dir/from/restyle \                       # path to `exp_dir` from inference_iterative.py (should contain `latents.npy`)
--latents_save_dir=/path/to/save/folder \                   # path to directory to save formatted latents
--latents_save_name=name_of_saved_latents.npy               # name of saved file (e.g., `latents_formatted.npy`)
```

- Optional: Run optimization. Script will update the latents path in `exp.json` automatically
    - Parameters for optimization found in `exp.json` (e.g., regularization parameter $\lambda$)
```
python optimize_latents.py \
--exp /path/to/handsoff/experiment/exp.json \               # path to exp.json for HandsOff (e.g., /experiments/face_seg.json)
--latents_path /path/to/initial/latents.npy                  # name of formatted outputs from format_latents.py
--latents_save_dir /path/to/save/folder \                   # path to save directory of refined latents
--latents_save_name name_of_saved_latents.npy \             # name of save file
--images_dir /path/to/images/to/refine \                    # path to images that were inverted
``` 
- If you don't optimize latents, format latents for training label generator

### Train the label generator
- Create an experiment config file (examples provided in `experiments/`)
- Train the label generator
```
python train_label_generator.py --exp experiments/exp.json 
```

## Generating Synthetic Datasets and Evaluation
### Generate data
```
python generate_data.py \                                   
--exp experiments/exp.json \                                # same config file as train_label_generator.py
--start_step start_step \                                   # int: random state to start dataset generation     
--resume path/to/dir/with/trained/label/generators \        # path to directory with label generator checkpoints 
--num_sample 10000 \                                        # number of image-label pairs to generate
--save_vis False                                            # whether to save colored images of generated labels
```

### Train DeepLabV3 on generated data
```
python train_deeplab.py \
--exp experiments/exp.json \                                        # same config file as train_label_generator.py
--data_path path/to/dir/with/trained/label/generators/samples \     # generate_data.py saves dataset to --resume/samples (if save_vis = False)
```

### Evaluate DeepLabV3 on real test set
```
python test_deeplab.py \
--exp experiments/exp.json \                                # same config file as train_label_generator.py
--resume path/to/dir/with/trained/deeplab/checkpoints \     # path to directory with trained DeepLabV3 checkpoints 
--validation_number val_number                              # Number of images used for validation. Takes the first val_number images for validation
```
