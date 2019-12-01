# ACDRNet
Official PyTorch implementation of "End to End Trainable Active Contours via Differentiable Rendering" ([link](https://openreview.net/pdf?id=rkxawlHKDr))

## Prerequisites
- python 3.6
- pytorch 1.1
- torchvision
- neural_renderer
- numpy
- scipy
- scikit-learn
- OpenCV
- tqdm
- h5py
- imageio
- pycocotools
- tensorboard

## Data preparation
### Buildings
The data folder should have the following format:
```
images/
    AAA.xxx
    BBB.xxx
masks/
    AAA.xxx
    BBB.xxx
```
Download links:

- Vaihingen : [link](https://drive.google.com/file/d/1nenpWH4BdplSiHdfXs0oYfiA5qL42plB/view)
- Bing : [link](https://drive.google.com/file/d/1Ta21c3jucWFoe5jwiVXXiAgozvdmnQKP/view)

### Cityscapes
Convert Cityscapes images to HDF5 for fast crop inference:
```
python data/make_hdf5.py [--images-path] [--outdir]
```
Generate Cityscapes instances .json files:
```
python data/generate_cityscapes_instances.py [--outdir] [--outfile] [--min-area]
```
Make sure you use the same path in both files for ```--outdir```. The data folder should have the following format:
```
cityscapes_instances/
    train/
        all_classes_instances.json
    train_val/
        all_classes_instances.json
    val/
        all_classes_instances.json
    all_images.hdf5    
```

Download links:
- Cityscapes : [link](https://www.cityscapes-dataset.com)
- Cityscapes splits from PolyRNN++ : [link](https://github.com/fidler-lab/polyrnn-pp)

## Train and Evaluate
Use the ```train.py``` for training. You can set the "validation" dataset to the "test" dataset 
for easy evaluation every ```--eval-rate``` epochs.

Training options:
```
# Training
--epochs        Number of epochs
--start-epoch   Starting epoch
--batch-size    Batch size
--lr            Learning rate
--lr-step       LR scheduler step

# Architecture
--arch          Network architecture. "unet" or "resnet"
--image-size    Neural Renderer output size
--dec-size      Spatial size of the decoder. Only relevant for ResNet
--enc-dim       Encoder dim(channels). Only relevant for UNet
--dec-dim       Decoder dim(channels)
--stages        ResNet skip connections
--drop          Dropout rate

# Active contour
--num-nodes     Number of nodes
--iter          AC number of iterations
--lmd-balloon   Balloon
--lmd-curve     Curvature
--lmd-dist      Distance

# Data
--train-dataset Training dataset
--ann-train     Split for training
--ann-val       Split tor evaluation

# Cityscapes Data
--inst-path     Path to Cityscapes instances directory
--ann-type      Type of annotation, full instance or only components. 
--class-name    Class for Cityscapes dataset
--loops         Data repetition in Cityscapes dataset

# Buildings Data
--data-path     Path to buildings dataset directory

# Misc
--eval-rate     Evaluate after "eval_rate" epochs
--save-rate     Save rate is "save_rate" * "eval_rate"
--checkname     Checkname
--resume        Resume file path
```

Cityscapes example:
```
python train.py --arch resnet --class-name train [--inst-path] --ann-type full --train-dataset cityscapes 
```

## Tunning
-   Add nodes by ```--num-node```
-   Sometimes fine-tunning with more AC iteration can yield better results, use ```--iter``` to set the number of iterations 
-   Train first on components ```--ann-type comp```  (Cityscapes)
-   In the data transformations switch between:
    - ```transforms.RandomResizedCrop((args.image_size, args.image_size), scale=(0.2, 2)),``` and ```transforms.RandomAffine(22, scale=(0.75, 1.25))```
    - ```transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])``` and ```transforms.NormalizeInstance()```

