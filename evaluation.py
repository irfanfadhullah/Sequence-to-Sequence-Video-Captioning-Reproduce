from sqlalchemy import except_all
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import json
import re, os
from tqdm import tqdm
from dataloader import VideoDataset
from coco_caption.pycocoevalcap.bleu.bleu import Bleu
from coco_caption.pycocoevalcap.rouge.rouge import Rouge
from coco_caption.pycocoevalcap.cider.cider import Cider
from coco_caption.pycocoevalcap.meteor.meteor import Meteor
from coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Opt: #The parameters
    # base_path = os.path.dirname(__file__)
    # print(base_path)
    # model_path = os.path.join(base_path, 'checkpoint/22_01_19_05_32_06-final.pth')
    # csv_file = os.path.join(base_path, '/data/video_corpus.csv')
    # train_source_file = os.path.join(base_path, '/data/annotation2016/train_val_videodatainfo.json')
    # caption_file = os.path.join(base_path, '/data/captions.json')
    # features_path = os.path.join(base_path, '/data/feats/msvd_vgg16_bn')
    model_path = 'vgg_rgb.pth'
    model_path1 = 'alexnet_flow.pth'
    csv_file = './data/video_corpus.csv'
    train_source_file = './data/annotation2016/train_val_videodatainfo.json'
    caption_file = './data/captions.json'
    features_path = './data/feats/msvd_vgg16_bn'
    # features_path1 = './data/feats/msvd_alexnet_flow'

    batch_size = 10

def eval():
    opt = Opt()

    #data
    #preparing the test set
    validation_set = VideoDataset(opt.caption_file, opt.features_path, mode='test') 
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size = opt.batch_size, shuffle=False) 
    word_to_index = validation_set.word_to_index #store the word_to_index data into the specific variable
    index_to_word = validation_set.index_to_word #store the word_to_index data into the specific variable
    voca_size = len(word_to_index) #count the length of the wor_to_index data or vocab

    #load_model and check if the computer has GPU or not
    try:
        model = torch.load(opt.model_path).to(device)
        # model1 = torch.load(opt.model_path).to(device)
    except:
        model = torch.load(opt.model_path,map_location=torch.device('cpu')).to(device)
        # model1 = torch.load(opt.model_path,map_location=torch.device('cpu')).to(device)
        
    #Testing
    predict_dict = {} #inititalize a dictionary to store the prediction data
    for index, (features, targets, IDs, masks) in enumerate(tqdm(validation_loader)): #iterate over the test dataset
        model.eval() #initialize the evaluation
        # model1.eval()
        with torch.no_grad(): #disable the gradient calculation
            # alpha = 0.6
            preds = model(features,mode = 'test')
            # preds1 = model1(features, mode = 'test')
            # print(preds)

            # print(preds.size())
            # print(preds1.size())

            # # print(preds)
            # preds = preds*alpha # multiplying with weight
            # print(preds)

            # preds1 = preds1*(1-alpha) # multiplying with weight
            # preds_all = preds.add(preds1) # addition of two tensors
            # preds_all = int(preds_all/2) # addition of two tensors

            # print(preds_all)
            # print(preds_all.size())

        #save
        for ID, pred_all in zip(IDs, preds): #iterate over the data for prediction
            word_pred = [index_to_word[str(i.item())] for i in pred_all] #
            if '<eos>' in word_pred:#remove the <eos> word
                word_pred = word_pred[:word_pred.index('<eos>')] #just use the word from beginning into before the <eos> word
            predict_dict[ID] = ' '.join(word_pred) #join all the word into a sentence
    return predict_dict #return the predict_dict data

def pred_to_coco(predict_dict, gts = None):
    samples = {} #create a new dictionary to store the samples
    IDs = [] #create a new list to store the IDs of the data
    for item in predict_dict.items(): #iterate over the data
        if gts is not None and item[0] in gts: #check the condition in the ground truth data
            IDs.append(item[0]) #if the condition is met, strore the ID to the IDs
            #store the image_id and caption into samples dictionary
            samples[item[0]] = [{
                u'image_id':item[0],
                u'caption': item[1]
            }]
    return samples, IDs #return the data

#this scorer is from the Microsoft COCO Caption Evaluation github
class coco_scorer(object):
    """
    codes from https://github.com/tylin/coco-caption
    Microsoft COCO Caption Evaluation
    """
    def __init__(self):
        print('init COCO Evaluation Scorer')
    
    def score(self, GT, RES, IDs):
        self.eval={}
        self.image_to_eval = {}
        gts = {}
        res = {}
        for ID in IDs:
            gts[ID] = GT[ID]
            res[ID] = RES[ID]
        
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Cider(), "CIDEr"),
            (Rouge(), "ROGUE_L")
        ]

        eval = {}
        for scorer, method in scorers:
            print('%s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method)==list:
                for sc, scs, m in zip(score, scores, method):
                    self.set_eval(sc, m)
                    self.set_image_to_eval_images(scs, IDs, m)
                    print("%s: %0.3f"%(method, score))
        return self.eval

    def set_eval(self, score, method):
        self.eval[method] = score

    def set_image_to_eval_images(self, scores, imgIds, method):
        for imgId, score, in zip(imgIds, scores):
            if imgId not in self.image_to_eval:
                self.image_to_eval[imgId] = {}
                self.image_to_eval[imgId]['image_id'] = imgId
            self.image_to_eval[imgId][method] = score

if __name__ == '__main__':
    prediction_dict = eval() #perform the evaluation step
    with open('./data/gts_3.json', encoding = 'utf-8') as f: #open the ground truth data
        gts = json.load(f)['gts']
    
    samples, IDs = pred_to_coco(prediction_dict, gts) #convert the format into coco scorer format

    scorer = coco_scorer() #call the scorer function
    scorer.score(gts, samples, IDs) #calculate the coco scorer

    #print the information
    print("===========================")
    print(scorer.set_eval)
    print("===========================")
    print(scorer.set_image_to_eval_images)