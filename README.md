# Small Object Few-shot Segmentation for Vision-based Industrial Inspection

This is an official PyTorch implementation of the paper [Small Object Few-shot Segmentation for Vision-based Industrial Inspection](https://arxiv.org/abs/2407.21351).
```
@article{zhang2024small,
  title={Small Object Few-shot Segmentation for Vision-based Industrial Inspection},
  author={Zhang, Zilong and Niu, Chang and Zhao, Zhibin and Zhang, Xingwu and Chen, Xuefeng},
  journal={arXiv preprint arXiv:2407.21351},
  year={2024}
}
```

<p align="center">
  <img src=assets/paradigm.png width="100%">
</p>

<p align="center">
  <img src=assets/SOFS.jpg width="100%">
</p>

We present SOFS to solve problems that various and sufficient defects are difficult to obtain and anomaly detection cannot detect specific defects in Vision-based Industrial Inspection. 
SOFS can quickly adapt to unseen classes without retraining, achieving few-shot semantic segmentation (FSS) and few-shot anomaly detection (FAD).
SOFS can segment the small defects conditioned on the support sets, e.g., it segments the defects with area proportions less than 0.03%.
Some visualizations are shown in the figure below.

<p align="center">
  <img src=assets/vis.jpg width="100%">
</p>

## **Visualizations under Open Domain**
- We show the visualizations of SOFS for [Severstal: Steel Defect Detection](https://www.kaggle.com/competitions/severstal-steel-defect-detection/overview) under the open domain, where SOFS is trained on VISION V1.
<p align="center">
  <img src=assets/steel.png width="70%">
</p>

### Installation
1. The default python version is python 3.8.
2. Follow the installation of [DINO v2](https://github.com/facebookresearch/dinov2), such as xFormers.
3. Use the following commands:
```
pip install -r requirements.txt
```

### Train and test on VISION V1 dataset or Ds spectrum
- Pretrained model prepare: please download DINO v2 ViT-B/14 distilled (without registers) pre-trained model in [DINO v2](https://github.com/facebookresearch/dinov2).
- Dataset prepare: please download [VISION V1 dataset](https://huggingface.co/datasets/VISION-Workshop/VISION-Datasets), the corresponding reference is at [here](https://arxiv.org/abs/2306.07890).
- Dataset prepare: please download [Ds spectrum dataset](https://envision-research.github.io/Defect_Spectrum/), the corresponding reference is at [here](https://arxiv.org/abs/2310.17316).
- Replace TRAIN.dataset_path and TEST.dataset_path with your own VISION V1/Ds spectrum dataset path.
- For Ds spectrum dataset, please firstly replace the file name in VISION v1 of Ds spectrum dataset with Capacitor_VISION/Ring_VISION..., then put these folders together including the name in dataset split.
- Replace TRAIN.backbone_checkpoint with the path of pre-trained DINO v2 ViT-B/14 distilled.
- Prepare an empty folder, replace DATASET.vision_data_save_path with the corresponding path.
- Then run the following code:

```
bash train.sh
```

- The model trains in each train split and test in the corresponding test split. The result is at the ./log.
- After you run the above command for the first time, replace DATASET.vision_data_save with False and replace DATASET.vision_data_load with True.

## **Inference**
- We provide the model trained on VISION V1 and code for SOFS inference (open-domain test). You can put your own data for open-domain test.
- Please download SOFS model at [here](https://drive.google.com/file/d/1sI9varMvniDBxjxBwlWpLpMvTj5j0D_B/view?usp=sharing) (Google Drive) and place it at "./SOFS_model.pth".

### **Prepare for Your Own Data**
- You can refer to the data format in severstal_steel of Open_Domain_Data. The data in severstal_steel are from [Severstal: Steel Defect Detection](https://www.kaggle.com/competitions/severstal-steel-defect-detection/overview). Our training data do not contain this data, thus this is an open-domain test.
- Your own data should be organized as follows:

```
|-- Your Own Data (object name)
    |-- support
        |-- image
        |-- mask
    |-- query
        |-- image
```

- support contains image fold and mask fold, each image in mask fold contains {0, 255}, 255 indicates the target semantic. image fold in query contains the test image.

### Test on your own dataset
- You should replace "severstal_steel" with your own object in DATASET.open_domain_test_object of "./method_config/Open_Domain/SOFS.yaml".
- Then run the following code:

```
sh test.sh
```

## To-Do List
- [x] Task 1: Release inference code and model (open-domain test).
- [x] Task 2: Release training and test code for different datasets (part).
- [ ] Task 3: Release inference for a mixture of defective support samples and normal support samples.
- [ ] Task 4: Release online tools.

## Acknowledgement
We acknowledge the excellent implementation from [DINO v2](https://github.com/facebookresearch/dinov2), [HDMNet](https://github.com/Pbihao/HDMNet).

## License
The code is released under the CC BY-NC-SA 4.0 license.