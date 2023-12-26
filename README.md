## S2VT: Sequence to Sequence Video to Text
This is the PyTroch implementation of **Sequence to Sequence Video to Text** for training and testing model.

## Preparing Tools needed
clone coco-caption repository https://github.com/tylin/coco-caption.git
rename folder coco-caption to coco_caption

## Installs
I use Python 3.6 in this project. Recommend installing pytorch and python packages using Anaconda.
* PyTorch
* Numpy
* tqdm
* pretrainedmodels
* ffmpeg (can install using anaconda)


## Datasets
Download YouTubeClips dataset from :
https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar
Extract YouTubeClips in folder ./data
## Prepare Flow Data
Type in terminal (you can change with you directory path) :
```
python optical_flow.py --video_path [YouTubeClips path]
```

## Extract Features and Corpus
1. Extract RGB Features using VGG-16
    Type in terminal (you can change with you directory path) :
    ```
    python extract_features.py --video_path [YouTubeClips path]  --features_path [ ./data/msvd_vgg16_bn]
    ```

2. Extract Flow Features using Alexnet
    Type in terminal (you can change with you directory path) :
    ```
    python extract_features.py --video_path [OpticalFlow path] --features_path [ .\data\feats\msvd_alexnet_flow]
    ```

3. Preprocess Video Caption by run preproces_caption.py
Note : you need to adjust the path within the code
    ```
    python preproces_caption.py
    ```

## Training
1. Edit directory in training_rgb_flow.py , and you can adjust with your own directory
2. For training the model, you can just type this in terminal :
    ```
    python training_rgb_flow.py
    ```
    
## evaluation
1. Edit directory in training_rgb_flow.py , and you can adjust with your own directory
2. For tetsing or evaluation the model, you can just type this in terminal :
    ```
    python evaluation.py
    ```

## Results
|                Scheme              |        METEOR       |        Bleu_4       |       ROUGUE_L      |         CIDEr       |
|:----------------------------------:|:-------------------:|:-------------------:|:-------------------:|:-------------------:|
|           Original   Paper         |         0.298       |          N/A        |          N/A        |          N/A        |
|               Baseline             |     0.2965893042    |     0.3152643061    |     0.6666146162    |     0.5998173136    |

## Refferences
```
Sequence to Sequence - Video to Text
S. Venugopalan, M. Rohrbach, J. Donahue, T. Darrell, R. Mooney, K. Saenko
The IEEE International Conference on Computer Vision (ICCV) 2015
```
