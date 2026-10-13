# Early Warning of Harmful Algal Blooms with Deep Learning on Imaging Flow Cytometry Images

APS360 (University of Toronto) individual project · Emily (Jiarui) Hu

A convolutional neural network (ImageNet-pretrained ResNet-18, fully fine-tuned) classifies phytoplankton images from an Imaging FlowCytobot (IFCB) into 38 expert-labelled SYKE classes. Each predicted class is grouped as a **HAB class** (potentially toxic cyanobacteria and dinoflagellates) or a **non-HAB class**, and predictions below a validation-tuned confidence threshold are labelled **uncertain**. A hand-crafted-feature random forest serves as the baseline.

## Data

Both datasets are public (CC BY 4.0) from the Finnish Environment Institute (SYKE) and are **not stored in this repository**; download them from B2SHARE:

| Dataset | Use | Contents | Link |
|---|---|---|---|
| SYKE-plankton_IFCB_2022 (images from 2016–2019) | Training / validation | 63,074 labelled images, 50 classes | [doi:10.23728/b2share.abf913e5a6ad47e6baa273ae0ed6617a](https://doi.org/10.23728/b2share.abf913e5a6ad47e6baa273ae0ed6617a) |
| SYKE-plankton_IFCB_Utö_2021 | Independent test set | 57,207 labelled + 94,028 unclassifiable images | [doi:10.23728/b2share.7c273b6f409c47e98a868d6517be3ae3](https://doi.org/10.23728/b2share.7c273b6f409c47e98a868d6517be3ae3) |

Classes with fewer than 100 images in 2016–2019, and the non-plankton Beads class, are excluded, leaving 38 SYKE classes.

## Repository contents

| Path | What it is |
|---|---|
| `EDA/APS360_EDA.ipynb` | Exploratory data analysis notebook (class counts, class distributions, imbalance, train vs test class mix) |
| `EDA/class_dist_2016_2019.png`, `EDA/class_dist_2021.png` | Class distributions of the training (2016–2019) and test (2021) data |
| `EDA/fig2_taxonomic_groups_2022.png` | Images per taxonomic group in the SYKE-plankton_IFCB_2022 release (2016–2019 images) |
| `EDA/class_counts_2021.csv` | Images per class folder in the 2021 test set |
| `EDA/resnet18_layers.jpg`, `EDA/resnet18_blocks.jpg` | ResNet-18 reference figures (see credits below) |
| `APS360 - Project Diagram - V2.pptx` | Project pipeline diagram |

## Running the EDA notebook

- **Google Colab:** open `EDA/APS360_EDA.ipynb` in Colab. The first cell mounts Google Drive; set `DATA_2021` to the extracted 2021 folder in your Drive. If the folder is not found, the notebook falls back to the published 2021 class counts.
- **Locally (Jupyter / VS Code):** set `DATA_2021` and `OUT` in the first cell and run all cells. Needs Python 3 and `matplotlib`.
- **Colab with a local runtime:** start Jupyter on your own computer and choose *Connect to a local runtime* in Colab to use local files and your own GPU.

## Figure credits

- `EDA/resnet18_layers.jpg`: F. Ramzan *et al.*, "A deep learning approach for automated diagnosis and multi-class classification of Alzheimer's disease stages using resting-state fMRI and residual neural networks," *Journal of Medical Systems*, vol. 44, p. 37, 2020, doi:10.1007/s10916-019-1475-2.
- `EDA/resnet18_blocks.jpg`: S. Zhao, S. Zhou, and R. Chen, "Research on radionuclide identification method based on GASF and deep residual network," *Applied Sciences*, vol. 15, no. 3, p. 1218, 2025, doi:10.3390/app15031218 (CC BY 4.0).
- Dataset: K. Kraft *et al.*, SYKE-plankton_IFCB_2022 and SYKE-plankton_IFCB_Utö_2021, B2SHARE, 2022.
