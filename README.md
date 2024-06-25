# Small Object Few-shot Segmentation for Vision-based Industrial Inspection

<p align="center">
  <img src=assets/SOFS.jpg width="70%">
</p>

We present SOFS to solve problems that various and sufficient defects are difficult to obtain and anomaly detection cannot detect specific defects in Vision-based Industrial Inspection. 
SOFS can quickly adapt to unseen classes without retraining, achieving few-shot semantic segmentation (FSS) and few-shot anomaly detection (FAD).
SOFS can segment the small defects conditioned on the support sets, e.g., it segments the defects with area proportions less than 0.03%.
Some visualizations are shown in the figure below.

<p align="center">
  <img src=assets/vis1.jpg width="70%">
</p>

## **More Visualizations**
<p align="center">
  <img src=assets/more_vis.jpg width="100%">
</p>

## **Visualizations under Open Domain**
- We show the visualizations of SOFS for [Severstal: Steel Defect Detection](https://www.kaggle.com/competitions/severstal-steel-defect-detection/overview) under the open domain, where SOFS is trained on VISION V1.
<p align="center">
  <img src=assets/steel.png width="50%">
</p>

## To-Do List
- [ ] Task 1: Release inference code and model (open-domain test).
- [ ] Task 2: Release inference for a mixture of defective support samples and normal support samples.
- [ ] Task 3: Release training and test code for different datasets.
- [ ] Task 4: Release online tools.

**We will release the code before October 2024.**

## License
The code is released under the CC BY-NC-SA 4.0 license.