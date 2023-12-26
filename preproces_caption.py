import pandas as pd
import numpy as np
import json, re
from collections import Counter
from tqdm import tqdm
import os

def build_vocab(all_words, min_freq=1):
    #returns a a list of tuples with each tuple having the word as the first member and the frequency as the second member.
    #The tuples are ordered by the frequency of the word.
    all_words = all_words.most_common() 
    word_to_index = {'<pad>': 0, '<unk>':1} #create a dictionary with <pad> has the key 0, and <unk> is 1
    for index, (word, freq) in enumerate(tqdm(all_words, desc='building vocab'), start=2): #iterate over all_words to build the vocab
        if freq<min_freq: #pass when the the frequency of the word is below to min_freq
            continue
        word_to_index[word] = index #create word to index data
    index_to_word = {v: k for k, v in word_to_index.items()} #reverse word to index to get the index to word data

    #output, printing the number of words and unknown word in the vocab
    print(f'number of words in vocab: {len(word_to_index)}')
    print(f'number of <unk> in vocab: {len(all_words)-len(word_to_index)}')

    return word_to_index, index_to_word

def parse_csv(csv_file, caption_file, gts_file, clean_only=False):
    file = pd.read_csv(csv_file, encoding='utf-8') #read the csv file (caption dataset)
    data = pd.DataFrame(file)   #create a new dataframe
    data = data.dropna(axis=0)  #drop the missing value
    eng_data = data[data['Language']=='English'] #set the data when the language is English
    if clean_only is True:  #if the clean_only is True then the algorith will choose only the clean category in the source column
        eng_data = eng_data[eng_data['Source']=='clean']
    print(f'There are totally {len(eng_data)} english descriptions') #print the information about the english data

    captions = []       #create a new list to store the caption
    counter = Counter() #call the Counter class
    filenames = []      #create a new list to store the filenames
    gts = {}            #create a new dictionary to store the ground truth
    max_cap_ids = {}    #create a new dictionary to store the max_cap_ids
    #iterate over dataset
    for _, name, start, end, sentence in tqdm(eng_data[['VideoID', 'Start', 'End', 'Description']].itertuples(), desc = 'reading captions'):

        file_name = name + '_' + str(start) + '_' +str(end)     #join the value to get the full filename
        filenames.append(file_name) #store the filename to the list

        tokenized = sentence.lower()    #set the sentence into lower (uncapitalize)
        tokenized = re.sub(r'[~\\/().!,;?:]', ' ', tokenized)   #remove the unwanted character
        gts_token = tokenized   #store the clean sentences into new variable
        tokenized = tokenized.split()   #split the sentence into word
        tokenized = ['<sos>'] + tokenized + ['<eos>'] #give the start of the sentences and end of sentences sign
        counter.update(tokenized)   #update the counter of the word with the clean tokenizer
        captions.append(tokenized)  #store the clean sentence into captions list

        if file_name in gts:        #check condition, if the file name is exist in the ground truth data
            max_cap_ids[file_name]+=1   #set the sigh into zero
            #make the dictionary format to store the filename, caption_id, caption, and clean ground truth
            gts[file_name].append({
                u'image_id':file_name,
                u'cap_id': 0,
                u'caption': sentence,
                u'tokenized': gts_token
            })
        else: #is not exist
            max_cap_ids[file_name] = 0 #set the sigh into zero
            #make the dictionary format to store the filename, caption_id, caption, and clean ground truth
            gts[file_name] = [{
                u'image_id': file_name,
                u'cap_id': 0,
                u'caption': sentence,
                u'tokenized': gts_token
            }]

    word_to_index, index_to_word = build_vocab(counter) #start the process of the build_vocab function to get tghe word_to_index and index_to_word data
    # 
    captions = [[word_to_index.get(w, word_to_index['<unk>']) for w in caption] for caption in tqdm(captions, desc = 'turing words into index')] 

    caption_dict = {}   #create a new variable to store the caption dict
    for name, cap in zip(filenames, captions): #iterate over the filename and caption
        if name not in caption_dict.keys(): #check if the file name is not in the caption
            caption_dict[name] = [] #if True, then set the name in the caption_dict as empty list
        caption_dict[name].append(cap)  #store the caption and file name into the caption dict

    #split dataset

    vid_names = list(caption_dict.keys())   #get all the video names
    # np.random.shuffle(vid_names)
    train_data = vid_names[:1200]       #split the data for the training
    valid_data = vid_names[1200: 1300]  #split the data for the validation
    test_data = vid_names[1300:1970]    #split the data for the testing
    data_split = {'train': train_data, 'validation':valid_data, 'test': test_data}  #combine all part into a dictionary
    with open(caption_file, 'w+',  encoding='utf-8') as f:  #save the caption file and other data with json format
        json.dump({
            'word_to_index' : word_to_index,
            'index_to_word': index_to_word,
            'captions': caption_dict,
            'splits': data_split
        }, f)

    with open(gts_file, 'w+', encoding='utf-8') as f:   #save the ground truth data
        json.dump({'gts': gts}, f)

if __name__ =='__main__':
    base_path = os.getcwd() #set the base path
    #perform the preprocess caption
    parse_csv(
        csv_file = os.path.join(base_path, 'data/video_corpus.csv'),
        caption_file = os.path.join(base_path, 'data/captions.json'),
        gts_file = os.path.join(base_path, 'data/gts_3.json'),
        clean_only=True
    )