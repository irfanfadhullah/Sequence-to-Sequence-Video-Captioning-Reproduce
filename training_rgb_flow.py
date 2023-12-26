import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import time
import os
from tqdm import tqdm

from dataloader import VideoDataset
from s2vt_model import S2VT
from attention_model import Base_Attention
from helper_function import MaskCriterion, EarlyStopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# writer = SummaryWriter()


class Opt:
    """config class"""
    # - data config
    base_path = os.getcwd() #change to your base path
    caption_file = os.path.join(base_path,"data/captions.json")  # the file generated in prepare_captions.py
    features_path_rgb = os.path.join(base_path,"data/feats/msvd_vgg16_bn")  # the features extracted by extract_features.py
    features_path_flow = os.path.join(base_path,"data/feats/msvd_alexnet_flow")  # the features extracted by extract_features.py
    
    # - model config
    train_length = 80  # fix length during training, the feats length must be equal to this
    dim_hidden = 512 #the dimension of the hidden layer
    dim_embedding = 512 #the dimension of the embedding layer
    # feature_dim = 1536 #using inception4
    feature_dim=4096 #the dimension of the extracted feature
    feature_dropout = 0
    output_dropout = 0
    rnn_dropout = 0
    num_layers = 1
    bidirectional = False  # do not use True yet
    rnn_type = 'lstm'  # do not change to GRU yet
    # - data config
    batch_size = 16
    # - train config
    EPOCHS = 300 #how much the epochs do you want
    save_freq = 100  # every n epoch, save once
    save_path = os.path.join(base_path,"checkpoint") #the directory of the path to save the checkpoint
    # histogram_freq = 10
    start_time = time.strftime('%y_%m_%d_%H_%M_%S-', time.localtime()) #
    early_stopping_patience = 30 #the patience to stop the training when there are no improvement after the certain epochs 
    # - optimizer config
    lr = 0.0001 #the initial learning rate
    learning_rate_patience = 20 #the patience to reduce the learning rate when there are no improvement after the certain epochs 


def save_opt(opt):
    with open(os.path.join(opt.save_path, opt.start_time + 'opt.txt'), 'w+', encoding='utf-8') as f: #intiate the path and file to save the options or the parameters
        f.write(str(vars(Opt)))

def train_rgb_flow():
    opt = Opt() #call the parameters
    # write log
    save_opt(opt) #save the parameters

    # prepare data
    trainset = VideoDataset(opt.caption_file, opt.features_path_rgb, opt.features_path_flow) #preprocess the dataset, and split it into the train part 
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True)
    testset = VideoDataset(opt.caption_file, opt.features_path_rgb, opt.features_path_flow, mode='validation') #preprocess the dataset, and split it into the validation part 
    test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False)
    word_to_index = trainset.word_to_index #store the word_to_index data into the specific variable
    index_to_word = trainset.index_to_word #store the index_to_word data into the specific variable
    vocab_size = len(word_to_index) #count the length of the wor_to_index data or vocab

    #
    model = Base_Attention(vocab_size, opt.feature_dim, length=opt.train_length, dim_hidden=opt.dim_hidden, dim_embedding=opt.dim_embedding,
                         feature_dropout=opt.feature_dropout, output_dropout=opt.output_dropout, sos_index=3, eos_index=4).to(device)
    # intialize the optimizer that use for training phase
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt.lr,
    )
    # dynamic learning rate
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, verbose=True, patience=opt.learning_rate_patience
    )
    # initiate the early stopping function
    early_stopping = EarlyStopping(patience=opt.early_stopping_patience,
                                   verbose=True,
                                   path=os.path.join(opt.save_path, opt.start_time + 'stop.pth'))
    criterion = MaskCriterion() #initiate and call the loss function

    try:
        for epoch in range(opt.EPOCHS): #iterate over epochs
            # ****************************
            #            train
            # ****************************
            train_running_loss = 0.0 #initialize the initial loss
            loss_count = 0 #initialize the initial loss
            # iterate over the dataset and get the features, targets, IDsm and masks for each data in train dataset
            for index, (feats, targets, IDs, masks) in enumerate(
                    tqdm(train_loader, desc="epoch:{}".format(epoch))):
                optimizer.zero_grad() #for every mini-batch during the training phase, set the gradients to zero before starting to do backpropragation (because it is the mini-bacth process)
                model.train() #train the model

                # probs [B, L, vocab_size]
                probs = model(feats, targets=targets[:, :-1], mode='train') #calculate the probability

                loss = criterion(probs, targets, masks) #compute the loss

                loss.backward() #compute gradient of loss w.r.t all the parameters in loss that have requires_grad = True
                optimizer.step() # updates all the parameters based on parameter.grad

                train_running_loss += loss.item() #compute the current loss
                loss_count += 1 #increase the loss count

            train_running_loss /= loss_count #average the loss with the loss count

            # ****************************
            #           validate
            # ****************************
            valid_running_loss = 0.0 #initialize the initial validation loss
            loss_count = 0 #set the loss count into 0
            for index, (feats, targets, IDs, masks) in enumerate(test_loader): #iterate over the validation dataset
                model.eval() #call the evaluation function

                with torch.no_grad(): #disable the gradient calculation
                    probs = model(feats, targets=targets[:, :-1], mode='train') #calculate the probability
                    loss = criterion(probs, targets, masks) #calculate the loss

                valid_running_loss += loss.item() # sum the loss with the previous loss
                loss_count += 1 #increase the loss count for validation step

            valid_running_loss /= loss_count    # calculate the average loss

            print("train loss:{} valid loss: {}".format(train_running_loss, valid_running_loss))
            lr_scheduler.step(valid_running_loss) #check the loss and compare with the previous one to perform the learning rate scheduler algorithm

            # early stop
            early_stopping(valid_running_loss, model) #perform the stopping algorithm if the loss is not improve in several epochs
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # save checkpoint
            if epoch % opt.save_freq == 0: #perform the save checkpoint every certain epoch, based on the parameters
                print('epoch:{}, saving checkpoint'.format(epoch))
                torch.save(model, os.path.join(opt.save_path,
                                               opt.start_time + str(epoch) + '_flow.pth'))

    except KeyboardInterrupt as e:
        print(e)
        print("Training interruption, save tensorboard log...")

    # save model
    torch.save(model, os.path.join(opt.save_path, opt.start_time + 'final_flow.pth')) #save the final model (after the training is finished)

if __name__ == '__main__':
    print("============================================")
    print("===============RBG_Flow Training================")
    print("============================================")
    train_rgb_flow() #perform the training step

