import torch
from torch import nn
from tqdm import tqdm 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Base_Attention(nn.Module):
    def __init__(self, vocab_size, dim_feature, length, dim_hidden=500, dim_embedding=500, feature_dropout=0,output_dropout=0, sos_index = 3, eos_index = 4):
        super(Base_Attention, self).__init__()

        #Parameters
        self.dim_feature = dim_feature
        self.length = length
        self.dim_hidden = dim_hidden
        self.dim_embed = dim_embedding
        self.vocab_size = vocab_size
        self.sos_index = sos_index
        self.eos_index = eos_index

        #Layers
        self.encoder = nn.LSTM(dim_hidden, dim_hidden, batch_first = True, bidirectional=True)
        self.decoder = nn.LSTM(dim_hidden*2+dim_embedding, dim_hidden, batch_first = True)
        self.feature_linear = nn.Linear(dim_feature, dim_hidden)
        self.feature_dropout = nn.Dropout(p=feature_dropout)
        self.embedding = nn.Embedding(vocab_size, dim_embedding, padding_idx=0)
        self.out_linear = nn.Linear(dim_hidden, vocab_size)
        self.out_drop = nn.Dropout(output_dropout)

        #Attention Layer
        self.attention_encoder = nn.Linear(dim_hidden * 2, dim_hidden, bias=True)
        self.attention_prev_hidden = nn.Linear(dim_hidden, dim_hidden, bias=True)
        self.attention_apply = nn.Linear(dim_hidden, 1,bias=False)

    def attention(self, encoder_outputs, decoder_prev_hidden = None):
        """
        :param encoder_outputs : [B, L, dim_hidden*2]
        :param decoder_prev_hidden : [1, B, dim_hidden]
        """
        if decoder_prev_hidden is None:
            batch_size = encoder_outputs.shape[0]
            decoder_prev_hidden = torch.zeros([1, batch_size, self.dim_hidden], device=device)

        # encoder_outputs[B, L, dim_hidden * 2] -> enc_W_h[B, L, dim_hidden]
        enc_W_h = self.attention_encoder(encoder_outputs)

        # decoder_prev_hidden[1, B, dim_hiddern] --repeat--> repeat_hidden[B, L, dim_hidden]
        repeat_hidden = decoder_prev_hidden.transpose(dim0=1, dim1=0)
        repeat_hidden = repeat_hidden.repeat([1, self.length, 1])
        decoder_W_h = self.attention_prev_hidden(repeat_hidden)

        # et[B,L,1]
        et = self.attention_apply(torch.tanh(torch.add(enc_W_h, decoder_W_h)))
        
        # at[B,1,L]
        at = torch.softmax(et, dim=2).squeeze(2).unsqueeze(1)

        #ct = sum[at_i * h_i] context[B,1, dim_hidden *2]
        context = torch.bmm(at, encoder_outputs)
        return context
        
    def forward(self, features, targets = None, mode = 'train'):
        #save the information
        batch_size = features.shape[0]

        #Encoding Stage
        features = self.feature_linear(self.feature_dropout(features))

        # features[B,L,dim_hidden] -> encoder_output[B,L, dim_hidden*2]
        encoder_outputs, _ = self.encoder(features)

        #Decoding Stage
        if mode == 'train': #for training step
            context = self.attention(encoder_outputs) #[B,1,dim_hidden *2] #apply attention function
            embed_targets = self.embedding(targets) #[B,L-1, dim_embedding] #create embedding for target sentence
            state = None #intialize the state into None
            probs = [] # create a new list for storing the probability
            for i in range(self.length - 1):
                current_word = embed_targets[:,i,:].unsqueeze(1) #change the dimension of the embedding target for second column
                decoder_input = torch.cat([current_word, context], dim=2)#concatenate the current word with the context

                #output[B,1,dim_hidden*2] hidden[2.b,dim_hidden]
                decoder_output, state = self.decoder(decoder_input, state) #decode the concatenate tensor of current word and context
                context = self.attention(encoder_outputs, state[0]) #get the context of the feature from encoder with attention function

                # prob[B,1, vocab_size]
                prob = self.out_linear(self.out_drop(decoder_output)) #apply linear layer to get the probability from the output of decoder
                probs.append(prob)  #store the probability into probs list
            return torch.cat(probs, dim=1) #concatenate the probability tensor along the first dimension
        elif mode =='test': #for the test step
            #current_word[B,1,dim_embedding]
            current_word = (self.sos_index * torch.ones([batch_size], dtype=torch.long)).to(device)  #prepare the current word
            current_word = self.embedding(current_word).view([batch_size,1,-1]) #make an embedding of the word
            context = self.attention(encoder_outputs) # [B,1,dom_hidden*2]
            state = None
            preds = [] #create a new list to store the prediction result
            for i in range(self.length): #iterate over length of test data
                decoder_input = torch.cat([current_word, context], dim=2) #create a new tensor for the input of decoder by concate the current word with context tensor

                #output[B,1,dim_hidden*2] hidden[2,b,dim_hidden]
                decoder_output, state = self.decoder(decoder_input, state) #preprocess the decoder
                context = self.attention(encoder_outputs, state[0]) #get the context from the encoder output

                #prob[B,1,voab_size]
                prob = self.out_linear(self.out_drop(decoder_output)) #compute the probability of the decoder output by linear layer
                pred = torch.argmax(prob, dim=2) #[B,1] #get the prediction
                current_word = self.embedding(pred) # [B,1,dim_embedding] #get the word from embedding
                preds.append(pred)#store each prediction into the list
            return torch.cat(preds, dim=1) #[B, L]