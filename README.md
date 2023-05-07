# Hybrid Deep Learning approach for Lane Detection (MSc thesis) 

# Introduction
Lane detection is a critical task in the field of autonomous driving and advanced driver assistance systems. This GitHub repository contains the code for the research work presented in a thesis that investigates the effectiveness of incorporating a Vision Transformer (ViT) to process feature maps extracted by a Convolutional Neural Network (CNN) for lane detection.
The research work explores the impact of incorporating temporal information from a road scene on a lane detection model's predictive performance. A post-processing technique that utilizes information from previous frames is proposed to improve the accuracy of the lane detection model.
The repository includes the implementation of a baseline CNN (SegNet) lane detection model, a hybrid CNN-ViT pipeline, and a post-processing mechanism that uses temporal information. The models are tested on the well-known TuSimple dataset, and the results are presented in the thesis report.

## Getting Started

# Installation and runtime steps

1. In order to train or evaluate the model, the TuSimple dataset should be downloaded from this directory: https://github.com/TuSimple/tusimple-benchmark/issues/3
2. After downloading the TuSimple dataset these directories need to be initialized by running the following commands from a terminal:

```shell
cd "PATH_TO_PROJECT"
mkdir datasets
mkdir models
mkdir plots
mkdir clips

```
3. After creating the necessary directories the TuSimple training and test datasets should be stored in the "datasets" folder as:
```bash
datasets/tusimple/
├─ train_set/
│  ├─ annotations/
│  │  ├─ 0313.json/
│  │  ├─ 0531.json/
│  │  ├─ 0601.json/
│  ├─ clips/
│  │  ├─ 0313_1/
│  │  ├─ 0313_2/
│  │  ├─ 0531/
│  │  ├─ 0601/
├─ test_set/
│  ├─ annotations/
│  │  ├─ test_label.json/
│  ├─ clips/
│  │  ├─ 0531/
│  │  ├─ 0530/
│  │  ├─ 0631/
```

In order to train our SegNet model from scratch, save the trained model in the "/models" directory and save the respective training plots you can run this cell:
```bash
cd "PATH_TO_PROJECT/resources"
python segnet_backbone.py
```

In order to train our pipeline model from scratch,save the trained model in the "/models" directory and save the respective training plots you can run this cell:
```bash
cd "PATH_TO_PROJECT/resources"
python pipeline.py
```
*** Running the pipeline is meant for reproducability reasons as according to our experiments, it seems to not perform well for lane detection.

Additionaly, two extra scripts are added in order to test our SegNet model with the post-processing module on your own unseen road scene data.

- In order to to this you should specify the "vid_path" variable in the clip2frame script and use mkdir to create an empty directory inside the "clips" folder.
This script turns your clip into sequential frames with 100ms difference between each of them
- In order to perform inference over your frames and turn those back into a videos similarly you need to use "mkdir pred_frames" to create an empty directory
in the clips folder to save the frames with the lane predictions overlay.
Then after changing the "frames_dir" variable to your previously defined frames directory you can run the pred2vid.py script to generate a video with the lane
predictions overlay.

# Results
Some results from our SegNet backbone model are depicted below:

<p align="left">
  <img src="https://iili.io/HSsTKEQ.jpg" alt="Original Image from the TuSimple dataset" width="400"/>
  <img src="https://iili.io/HSsTFCx.jpg" alt="Ground truth binary mask" width="400"/>
</p>
<p align="center">
  <em> a) Original Image from the TuSimple dataset</em> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; <em> b) Ground truth binary mask</em>
</p>

<p align="left">
  <img src="https://iili.io/HSsTB3B.png"alt="Prediction without temporal post-process" width="400"/>
  <img src="https://iili.io/HSsTf4V.png" alt="Prediction with temporal post-process" width="400"/>
</p>
<p align="center">
  <em>c) Prediction without temporal post-process</em> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; <em>d) Prediction with temporal post-process</em>
</p>






# Contributors
| Name | LinkedIn Profile |
| --- | --- |
| Dimitrios Zarogiannis | [https://www.linkedin.com/in/dimitrios-zarogiannis-2814a21b1/](https://www.linkedin.com/in/dimitrios-zarogiannis-2814a21b1/) |
| Stelio Bompai | [https://www.linkedin.com/in/stelio-bompai-4a96a3180/](https://www.linkedin.com/in/stelio-bompai-4a96a3180/) |

