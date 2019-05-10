# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:34:50 2019

@author: qianliang

pytorch version char rnn generator
"""

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
from datetime import datetime
from utils import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import init
#import torchvision.datasets as dsets
#import torchvision.transforms as transforms
from torch.autograd import Variable



USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")



vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print( "Reading CSV file...")
with open('data/1.txt', 'r',encoding='UTF-8') as f:
    reader = csv.reader(f, skipinitialspace=True)
    next(reader)
#    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
#    for x in reader:
#        sents = sent_detector.tokenize(x[0])
#        print(sents)
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print( "Parsed %d sentences." % (len(sentences)))

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print( "Found %d unique words tokens." % len(word_freq.items()))

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print ("Using vocabulary size %d." % vocabulary_size)
print ("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

print ("\nExample sentence: '%s'" % sentences[0])
print ("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])


#V : (word_dim, hidden_dim)
#W: (hidden_dim, hidden_dim)
#U: (hidden_dim, word_dim)

class GRU(nn.Module):
    
    def __init__(self, word_dim = vocabulary_size, hidden_dim = 100, output_dim = vocabulary_size):
        super(GRU, self).__init__()
        self.input_dim = word_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.Uz = nn.Linear(self.input_dim, self.hidden_dim)
        self.Wz = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.Ur = nn.Linear(self.input_dim, self.hidden_dim)
        self.Wr = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.Uh = nn.Linear(self.input_dim, self.hidden_dim)
        self.Wh = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.V = nn.Linear(self.hidden_dim, self.output_dim)

        self.Tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)
        self.Sigmoid = nn.Sigmoid()
        
        init.xavier_normal(self.Wz.weight)
        init.xavier_normal(self.Uz.weight)
        
        init.xavier_normal(self.Wr.weight)
        init.xavier_normal(self.Ur.weight)
        
        init.xavier_normal(self.Wh.weight)
        init.xavier_normal(self.Uh.weight)
        
        init.xavier_normal(self.V.weight)
        
        
        
    def forward(self, x, hidden=None):

        if hidden is None:
            
            hidden = torch.zeros(1, self.hidden_dim).cuda()
            hidden.requires_grad_(True)
 
        # GRU Layer 1
        z_t1 = self.Sigmoid(self.Uz(x) + self.Wz(hidden))
        r_t1 = self.Sigmoid(self.Ur(x) + self.Wr(hidden))
        c_t1 = self.Tanh(self.Uh(x) + self.Wh(hidden*r_t1))
        print(z_t1.shape,hidden.shape)
        print((z_t1 * hidden).shape)
        hidden = (torch.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * hidden
        
        
        output = self.V(hidden)
        output = self.softmax(output)
        
        return output,hidden


class LSTM(nn.Module):
    
    def __init__(self, word_dim = vocabulary_size, hidden_dim = 100, output_dim = vocabulary_size):
        super(LSTM, self).__init__()
        
        self.input_dim = word_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.Ui = nn.Linear(self.input_dim, self.hidden_dim)
        self.Wi = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.Uf = nn.Linear(self.input_dim, self.hidden_dim)
        self.Wf = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.Uo = nn.Linear(self.input_dim, self.hidden_dim)
        self.Wo = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.Ug = nn.Linear(self.input_dim, self.hidden_dim)
        self.Wg = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.V = nn.Linear(self.hidden_dim, self.output_dim)

        init.xavier_normal(self.Wi.weight)
        init.xavier_normal(self.Ui.weight)
        
        init.xavier_normal(self.Wf.weight)
        init.xavier_normal(self.Uf.weight)
        
        init.xavier_normal(self.Wo.weight)
        init.xavier_normal(self.Uo.weight)
        
        init.xavier_normal(self.Wg.weight)
        init.xavier_normal(self.Ug.weight)

        init.xavier_normal(self.V.weight)

        self.Tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)
        self.Sigmoid = nn.Sigmoid()


        
    def forward(self, x, hidden = None, c = None):

        if hidden is None:
            
            hidden = torch.zeros(1, self.hidden_dim).cuda()
            hidden.requires_grad_(True)
        if c is None:
            c = torch.zeros(1, self.hidden_dim).cuda()
            hidden.requires_grad_(True)
            
            
 
        i_t1 = self.Sigmoid(self.Ui(x) + self.Wi(hidden))
        f_t1 = self.Sigmoid(self.Uf(x) + self.Wf(hidden))
        o_t1 = self.Sigmoid(self.Uo(x) + self.Wo(hidden))
        g = self.Tanh(self.Ug(x) + self.Wg(hidden))
        # vector dot product
        c_t1 = c * f_t1 + g * i_t1
        hidden = self.Tanh(c_t1) * o_t1
        
        output = self.V(hidden)
        output = self.softmax(output)
        
        return output,hidden,c_t1


class RNN(nn.Module):
    
    def __init__(self, word_dim = vocabulary_size, hidden_dim = 100, output_dim = vocabulary_size):
        super(RNN, self).__init__()
        self.input_dim = word_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.W = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.U = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.V = nn.Linear(self.hidden_dim, self.output_dim, bias=False)

        self.Tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

        init.xavier_normal(self.W.weight)
        init.xavier_normal(self.U.weight)
        init.xavier_normal(self.V.weight)
        
    def forward(self, x, hidden=None):

        if hidden is None:
            
            hidden = torch.zeros(1, self.hidden_dim).cuda()
            hidden.requires_grad_(True)
 
        hidden = self.Tanh(torch.add(self.U(hidden),self.W(x)))
        output = self.V(hidden)
        output = self.softmax(output)
        return output,hidden

    
# one-hot matrix of first to last words (including end tag) for input
def inputTensor(line):
#    print('inputTensor',line)
    tensor = torch.zeros(len(line),1, vocabulary_size).cuda()
    for li in range(len(line)):
        index = line[li]
        tensor[li][0][index]=1
    return tensor

def inputTensorSample(index):
    tensor = torch.zeros(1,1, vocabulary_size).cuda()
    tensor[0][0][index]=1
    return tensor


learning_rate = 0.005


use_lstm = True

if use_lstm:
    rnn = LSTM(vocabulary_size, 100, vocabulary_size)
else:
    rnn = GRU(vocabulary_size, 100, vocabulary_size)
    
rnn = rnn.cuda()
opt = torch.optim.RMSprop(rnn.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
#opt = torch.optim.SGD(rnn.parameters(),lr = learning_rate,momentum=0.9)
print(rnn)

#criterion = nn.NLLLoss()
criterion = nn.NLLLoss()



def train(input_line, target_line):
    
    rnn.zero_grad()
    
    loss = 0
    hidden = None
    
    if use_lstm:
        c= None
    
    input_line = inputTensor(input_line)
    target_line = torch.LongTensor(target_line).cuda()
#    print('target_line',target_line)
    target_line.unsqueeze_(1) # column array
    
    for i in range(input_line.size(0)):
        
        if use_lstm:
            output, hidden, c = rnn(input_line[i], hidden, c)
        else:
            output, hidden = rnn(input_line[i], hidden)
            
        l = criterion(output, target_line[i])
        loss+=l
        
    loss.backward()
    opt.step()
    
#    for p in rnn.parameters():
#        p.data.add_(-learning_rate, p.grad.data)
        
    return output, loss.item() / input_line.size(0)


import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s/60)
    s -= m*60
    return '%dm %ds' %(m,s)


max_length = 20
# sample from a category and starting letter
def sample(start_word='SENTENCE_START',hidden_of_model = None):
    
    with torch.no_grad(): # no need to track history in sampling
        input = inputTensorSample(word_to_index[start_word])
        output_sen = []
        output_sen.append(start_word)
        hidden = None
        
        if use_lstm:
            c = None

        for i in range(max_length):
          
            if use_lstm:
                output, hidden,c  = rnn(input[0], hidden, c)
            else:
                output, hidden  = rnn(input[0], hidden)
                
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == vocabulary_size -1:
                break
            else:
                output_sen.append(index_to_word[topi])
            input = inputTensorSample(topi)
        print('sample sentence:',output_sen)
        return output_sen,hidden_of_model
    
    

print_every = 50
plot_every = 2000
all_losses = []
total_loss = 0 # Reset every plot_every iters

start = time.time()

#print(len(X_train),len(y_train))
nepoch = 20
num_examples_seen = 0

for epoch in range(nepoch):

    total_loss = 0

    for i in range(len(y_train[:200])):
    
        output, loss = train(X_train[i], y_train[i])
        num_examples_seen += 1
        #print(output.shape)
        total_loss += loss
#        print('%s (%d %d%%) %.4f' % (timeSince(start), i, i / len(y_train) * 200, loss))
        
#        if i % print_every == 0:
#            print('%s (%d %d%%) %.4f' % (timeSince(start), i, i / len(y_train) * 100, loss))
    
    all_losses.append(total_loss/200)
            
    print('num_examples_seen',num_examples_seen,'epoch:',epoch)
    print('all_losses',all_losses)
    if (len(all_losses) > 1 and all_losses[-1] > all_losses[-2]):
        learning_rate = learning_rate * 0.8
        if 0.00005 > learning_rate:
            learning_rate = 0.00005
        print ("Setting learning rate to %f" % learning_rate)
    sample()
        
    
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)




    






