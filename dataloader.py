from torch.utils.data import Dataset
import torch, json, random
import pathlib as plb
import numpy as np
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VideoDataset(Dataset): #custom DataLoader
    '''This dataloader load data from two feature that already extracted before (extracted from vgg16 rgb, and alexnet flow)'''
    def __init__(self, captions_file, features_path1, features_path2, max_len=80, mode = 'train'):
        with open(captions_file, encoding='utf-8') as f:
            data = json.load(f) #load the data
            self.word_to_index = data['word_to_index']  #store the word_to_index into a variable
            self.index_to_word = data['index_to_word'] #store the index_to_word into a variable
            self.captions = data['captions']    #store the caption data into a variable
            self.splits = data['splits']        #store the data_split into a variable

        features_path1 = [i for i in plb.Path(features_path1).glob('*.npy')] #load extracted feature from vgg16 rgb
        self.features_path1 = []    #create a new list to store the 
        for path in features_path1:
            if path.stem in self.splits[mode]: #check is the name file is in dataset that already splitted or not
                self.features_path1.append(path) #if True, append the path of the file into the features_path1

        features_path2 = [i for i in plb.Path(features_path2).glob('*.npy')] #load extracted feature from alexnet flow
        self.features_path2 = [] #same like features_path1
        for path in features_path2:
            if path.stem in self.splits[mode]:
                self.features_path2.append(path)

        self.max_len = max_len  

    def __getitem__(self, index):
        ID = self.features_path1[index].stem #get the name of the file
        alpha = 0.6 #set the weight
        features1 = np.load(str(self.features_path1[index]))    #load the feature
        features1 = torch.tensor(features1, dtype=torch.float, device = device, requires_grad=True) #convert to the tensor format
        features2 = np.load(str(self.features_path2[index]))    #load the feature
        features2 = torch.tensor(features2, dtype=torch.float, device = device, requires_grad=True) #convert to the tensor format
        features2 = features2*(1-alpha) # apply weighted with parameter alpha
        features1 = features2*alpha  # apply weighted with parameter alpha
        features = (features1.add(features2))/2  # combine two feature
        labels = self.captions[ID]  #store the caption to the label variable
        label = np.random.choice(labels, 1)[0] #get the random label from the dataset

        if len(label) > self.max_len:   #check the condition, is the label is greater than the maximum length or not
            label = label[:self.max_len]    #if Ture, then set the label length same as the maximum length
        pad_label = torch.zeros([self.max_len], dtype=torch.long, device = device)  #create a tensor contain of scalar zero
        pad_label[:len(label)] = torch.tensor(label, dtype=torch.long, device=device)   #replace the padding_label with the tensor of label data
        mask = torch.zeros([self.max_len], dtype=torch.float, device=device)    #create a tensor contain of scalar zero
        mask[:len(label)] = 1   #masking the label with one

        return features, pad_label, ID, mask

    def __len__(self):
        return len(self.features_path) #give the information about how much the feature file

if __name__ == '__main__':
    base_path = os.getcwd() #change to your base path
    #perform and making a dataset for trainset
    trainset = VideoDataset(os.path.join(base_path, '/data/captions.json'), os.path.join(base_path, '/data/feats/msvd_vgg16_bn'), os.path.join(base_path, '/data/feats/msvd_alexnet_flow'))
    train_loader = torch.utils.data.DataLoader(trainset) #
    # a = next(iter(train_loader))    #get the bacth of sample (not important)