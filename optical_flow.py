# import pretrainedmodels
# from pretrainedmodels import utils
import torch
import os
import subprocess
import shutil
import pathlib as plb
import numpy as np
from tqdm import tqdm
import json
import argparse
import numpy as np
import cv2


def extract_frames(video_path, output_destination):
    '''
    video_path =  path or directory of video
    output_destination = folder for image output that already extracted
    '''

    with open(os.devnull, 'w') as ffmpeg_log:
        if os.path.exists(output_destination):
            shutil.rmtree(output_destination) #remove previous folder if exist
        os.makedirs(output_destination) #make new folder

        video_to_frames_params = [
            "ffmpeg",#call the ffmpeg function
            '-y',#Overwrite output files without asking.
            '-i', video_path, '-vf', 'scale=400:300',#"-i" for the input video, "-vf", and rescale the frame
            '-qscale:v', '2',
            '{0}/%06d.jpg'.format(output_destination)#set the format and the directory of the extracted frames
        ]

        subprocess.call(video_to_frames_params, stdout=ffmpeg_log, stderr=ffmpeg_log)#start convert the video into frame

def motion_vector(frame_2, magnitude, angle):
    '''
    Motion vector function is used to create a motion vector that needed as a part of making the optical flow video
    '''
    #motion vectors spaced 25x25 from each other
    for i in range(frame_2.shape[0]):        
        for j in range(frame_2.shape[1]):    
            if i % 25 == 0 and j % 25 == 0:  
                arrow_start = (j,i)               
                arrow_end = (i+(2.5 * magnitude[i,j] * np.sin(angle[i,j])), j+(2.5 * magnitude[i,j] * np.cos(angle[i,j]))) 
                #draw the arrows for vectors from start to end point
                frame_2 = cv2.arrowedLine(frame_2, arrow_start, (int(arrow_end[1]),int(arrow_end[0])),(0,0,255), 2)
                #window = cv2.arrowedLine(window, arrow_start, (int(arrow_end[1]),int(arrow_end[0])),(0,0,255), 2)
    return frame_2

def video_to_optical_flow_video(video_path, output_destination):
    name_file = video_path.split("\\")[-1] #get the name of the video file
    
    # save path for video that already convert to flow filter
    out_1 = cv2.VideoWriter(os.path.join(output_destination,name_file),cv2.VideoWriter_fourcc(*'mp4v'), 30, (640,480))

    cap = cv2.VideoCapture(video_path)#capture the video, frame by frame

    #reading the video
    ret, frame_1 = cap.read() #read the captured frame
    frame_1 = cv2.resize(frame_1,(640,480),fx=0,fy=0,interpolation=cv2.INTER_AREA) #resize the resolution and doing some interpolation process

    previous_frame = cv2.cvtColor(frame_1,cv2.COLOR_BGR2GRAY)#convert the image frame into grayscale
    hsv = np.zeros_like(frame_1) #Return an array of zeros with the same shape and type as a given array.
    hsv[:,:,1] = 255 #

    while True:
        ret_1, frame_2 = cap.read()#read the frame 2
        if frame_2 is None:
            break
        
        frame_2 = cv2.resize(frame_2,(640,480),fx=0,fy=0,interpolation=cv2.INTER_AREA) #resize the frame
        
        #blank canvas to plot the motion vectors
        window = np.zeros_like(frame_2)
        window_1 = np.zeros_like(frame_2)
        
        next_frame = cv2.cvtColor(frame_2,cv2.COLOR_BGR2GRAY)#convert the frame into grayscale
        
        #Farneback optical flow algorithm providing the gradients u and v
        optical = cv2.calcOpticalFlowFarneback(previous_frame,next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0) #compute a dense optical flow using the Gunnar Farneback's algorithm
        optical_1 = optical.copy()#copy the optical image
        
        gradient = 2 * (optical_1[:,:,0]) #get the gradient from the optical flow process
        #Return elements chosen from x or y depending on condition.
        where = np.where(gradient > 1)
        
        for i in range(len(where[0])):
            window_1[where[0][i],where[1][i],:] = frame_2[where[0][i],where[1][i],:] #store or fill the value from frame2 to the window 1 (zero array)
        
        #calculating the magnitude and direction of the gradients
        magnitude, angle = cv2.cartToPolar(optical[:,:,0], optical[:,:,1])
        
        #motion vector function to draw the vectors
        frame_2 = motion_vector(frame_2,magnitude, angle)
        #Sets image value according to the optical flow
        #direction put into the hus channel
        hsv[:,:,0] = angle * 180 / np.pi / 2
        
        #magnitude into the value channel 
        hsv[:,:,2] = cv2.normalize(magnitude,None,0,255,cv2.NORM_MINMAX)
        
        frame_BGR = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)#convert back from hsv to the BGR format
        out_1.write(frame_BGR)#save the frame

    out_1.release() 
    
def extract(video_path, frame_number = 80):

    video_path = plb.Path(video_path)#set the video path
    # features_path = plb.Path(features_path)
    # temp_path = plb.Path(r"D:/Irfan/hasil 1/")
    fix_path = plb.Path(r"D:/Irfan/hasil_fix/") # you must replace this path to your own path (this is the folder for store the extracted features)
    temp_path = plb.Path(r"D:/Irfan/hasil 2/") # you must replace this path to your own path (this is the folder for store the extracted features that currently process to avoid bad data when process stopped)
#same like 
    assert video_path.is_dir()#is used to debugging the code, when the condition is true, then nothing happen. If false, then AssertionError is raised
    # if features_path.is_dir():
    #     shutil.rmtree(features_path)
    # os.mkdir(features_path)
    if fix_path.is_dir(): #check the availability of the path
        pass #if True, doing nothing
    else:
        os.mkdir(fix_path) #if there is no existing path, then create a new path
    if temp_path.is_dir():      #check the availability of the path 
        shutil.rmtree(temp_path) #if there is an existing path, the remove it
    os.mkdir(temp_path)# and create a new path

    for video in tqdm(list(video_path.iterdir()), desc = 'Extracting...'):
        name_file = str(video).split("\\")[-1]#get the name of the video file
        path_save = plb.Path(os.path.join(fix_path,name_file)) #set the directory to save the extracted frame
        if path_save.is_file(): #pass the data that already extracted
            pass
        else:
            video_to_optical_flow_video(str(video), os.path.join(temp_path))#begin the process of converting from the normal into optical flow video
            os.rename(os.path.join(temp_path,name_file), path_save) #rename the temporary file into save file
        
        # extract_features(temp_path, features_path, frame_number, video.stem, model)


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', required=True, type=str)    #argument for defining the video path
    # parser.add_argument('--features_path', required=True, type=str)
    parser.add_argument('--frames_number', default=80, type=int)    #argument for defining the number of frame
    # parser.add_argument('--model', required=True,choices=['vgg16', 'inception_v4'], type=str)
    parser.add_argument('--gpu', default=0, type=int)   #defining the device that want to use to execute the process
    args = parser.parse_args()  #The parsed arguments are present as object attributes, so you can call the parser attribute (args.gpu)
    args = vars(args)           #It takes an object as a parameter which may be can a module, a class, an instance, or any object having __dict__ attribute. 

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args['gpu'])   #specify the environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   #define the device that want to use
    # print("Extract Features with {}, device: {}".format(args['model'], str(device)))
    
    #begin the extracted process
    extract(
        video_path=args['video_path'],
        # features_path=args['features_path'],
        frame_number=args['frames_number'],
        # model=args['model']
        )