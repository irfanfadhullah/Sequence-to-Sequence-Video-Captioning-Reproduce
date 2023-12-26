import pretrainedmodels
from pretrainedmodels import utils
import torch
import os
import subprocess
import shutil
import pathlib as plb
import numpy as np
from tqdm import tqdm
import json
import argparse


def extract_frames(video_path, output_destination):
    '''
    video_path =  path or directory of video
    output_destination = folder for image output that already extracted
    '''

    with open(os.devnull, 'w') as ffmpeg_log:
        if os.path.exists(output_destination):
            shutil.rmtree(output_destination) #remove previous folder if exist
        os.makedirs(output_destination) #make new folder
        #setting the parameters using ffmpeg library
        video_to_frames_params = [
            "ffmpeg", #call the ffmpeg function
            '-y', #Overwrite output files without asking.
            '-i', video_path, '-vf', 'scale=400:300', #"-i" for the input video, "-vf", and rescale the frame
            '-qscale:v', '2',#set 
            '{0}/%06d.jpg'.format(output_destination)#set the format and the directory of the extracted frames
        ]

        subprocess.call(video_to_frames_params, stdout=ffmpeg_log, stderr=ffmpeg_log) #start convert the video into frame

def extract_features(temp_path, features_path, frame_number, video_name, model):
    '''
    temp_path : Temporary path for extracted frame
    features_path : path to store the extracted features
    frame_number : get same number of frame for every video extracted
    '''

    #load model, in this extract features, I use inceptionv4 because the pretrained model dont use much GPU Resource
    C, H, W = 3,299,299 #parameters for image size and channel
    #choose the model that you want to use as the backbone to extract the features
    if model == 'inceptionv4':
        model = pretrainedmodels.inceptionv4(pretrained='imagenet')
    elif model == 'vgg16':
        model = pretrainedmodels.vgg16(pretrained='imagenet')
        C, H, W = 3, 224, 224
    elif model =='alexnet':
        model= pretrainedmodels.alexnet(pretrained='imagenet')
        C, H, W = 3, 224, 224
    model.last_linear = utils.Identity() #use the last linear function of the pretrained model

    model.to(device) #store the model to device (gpu or cpu)
    model.eval() #call the evaluation finction
    load_image_fn = utils.LoadTransformImage(model) #call the augmentation function yo convert all image into RGB format and doing some augmentation technique from pretrainedmodels library

    #load data
    image_list = sorted(temp_path.glob('*.jpg')) # get list of image in temporary path or directory
    #get index
    samples_index = np.linspace(0, len(image_list)-1, frame_number).astype(int)
    image_list = [image_list[i] for i in samples_index]
    # build tensor
    images = torch.zeros([len(image_list), C,H,W]) #create a tensor of scalar 0 with given dimension
    for i in range(len(image_list)):            #iterate over length of image
        image = load_image_fn(image_list[i])    #preprocess the image 
        images[i] = image                       #stroe the processed image into some list
    images = images.to(device)                  #stire image to the device that used
    with torch.no_grad():                       #disable the gradient calculation
        features = model(images)                #preprocess the image
    features = features.cpu().numpy()           #store the extracted feature to the cpu

    #save to npy format
    np.save(os.path.join(features_path, video_name+'.npy'), features) #save the extracted features

def extract(video_path, features_path,model, frame_number = 80):
    video_path = plb.Path(video_path)       #store the video path into new variable and convert the directory path into same standar (\ or /)
    features_path = plb.Path(features_path) #store the features path into new variable and convert the directory path into same standar (\ or /)
    temp_path = plb.Path(r"./_frames_out")  #store the temp path into new variable and convert the directory path into same standar (\ or /)
    assert video_path.is_dir()              #is used to debugging the code, when the condition is true, then nothing happen. If false, then AssertionError is raised
    if features_path.is_dir():              # check the feature driectory
        shutil.rmtree(features_path)        #if found, so remove the feature path
    os.mkdir(features_path)                 #and make a new one
    if temp_path.is_dir():                  # same like feature path
        shutil.rmtree(temp_path)            
    os.mkdir(temp_path)                     

    for video in tqdm(list(video_path.iterdir()), desc = 'Extracting...'):
        extract_frames(str(video), os.path.join(temp_path)) # extract video to frame
        extract_features(temp_path, features_path, frame_number, video.stem, model) # extract from frames to features


if __name__ =='__main__':
    parser = argparse.ArgumentParser()      #call the argument parser function
    parser.add_argument('--video_path', required=True, type=str)    #argument for the video path
    parser.add_argument('--features_path', required=True, type=str) #argument for the feature path
    parser.add_argument('--frames_number', default=80, type=int)    #argument for how much frame per video that want to extract
    parser.add_argument('--model', required=True,choices=['vgg16', 'inception_v4', 'alexnet'], type=str)    #define the model that want to use
    parser.add_argument('--gpu', default=0, type=int)   #specify the gpu device
    args = parser.parse_args()  #The parsed arguments are present as object attributes, so you can call the parser attribute (args.gpu)
    args = vars(args)           #It takes an object as a parameter which may be can a module, a class, an instance, or any object having __dict__ attribute. 

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args['gpu'])   #specify the environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   #define the device that want to use
    print("Extract Features with {}, device: {}".format(args['model'], str(device)))    #print the information about the model and device that used in extracting the feature
    #begin the extract feature process
    extract(
        video_path=args['video_path'],
        features_path=args['features_path'],
        frame_number=args['frames_number'],
        model=args['model']
        )