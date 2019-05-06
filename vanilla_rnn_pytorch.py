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
import torchvision.datasets as dsets
import torchvision.transforms as transforms
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


class RNN(nn.Module):
    
    def __init__(self, word_dim = vocabulary_size, hidden_dim = 100, output_dim = vocabulary_size):
        super(RNN, self).__init__()
        self.input_dim = word_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.i2h = nn.Linear(self.input_dim, self.hidden_dim)
        self.h2h = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.h2o = nn.Linear(self.hidden_dim, self.output_dim)

        self.Tanh = nn.Tanh()
        
    def forward(self, x, hidden=None):

        if hidden is None:
            # only one hidden unit
            hidden = torch.zeros(1, self.hidden_dim,device=device)
#        print('input_',x.shape)
        WXt = self.i2h(x) 
        UHt_1 = self.h2h(hidden) 
        St = torch.add(UHt_1,WXt)
        hidden = self.Tanh(St)
        self.hidden = hidden
        output = self.h2o(hidden)
        #output = nn.Softmax(output)
        
        return output,hidden

    
# one-hot matrix of first to last words (including end tag) for input
def inputTensor(line):
    tensor = torch.zeros(len(line),1, vocabulary_size).cuda()
    for li in range(len(line)):
        index = line[li]
        tensor[li][0][index]=1
    return tensor

def inputTensorSample(index):
    tensor = torch.zeros(1,1, vocabulary_size).cuda()
    tensor[0][0][index]=1
    return tensor

criterion = nn.CrossEntropyLoss()

learning_rate = 0.0005

def train(input_line, target_line):
    
    rnn.zero_grad()
    
    loss = 0
    hidden = None
    
    input_line = inputTensor(input_line)
    target_line = torch.LongTensor(target_line).cuda()
#    print('target_line',target_line)
    target_line.unsqueeze_(1) # column array
    
    for i in range(input_line.size(0)):
        output, hidden = rnn(input_line[i], hidden)
        
        l = criterion(output, target_line[i])
        loss+=l
        
    loss.backward()
    
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)
        
    return output, loss.item() / input_line.size(0)

# sample from a category and starting letter
def sample(start_word='SENTENCE_START',hidden_of_model = None):
    
    with torch.no_grad(): # no need to track history in sampling
        input = inputTensorSample(word_to_index[start_word])
        print('sample',input.shape)
        output_sen = []
        output_sen.append(start_word)
        hidden = None
        for i in range(max_length):
          
            output, hidden = rnn(input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == vocabulary_size -1:
                break
            else:
                output_sen.append(index_to_word[topi])
            input = inputTensorSample(topi)
        print('sample sentence:',output_sen)
        return output_sen,hidden_of_model


import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s/60)
    s -= m*60
    return '%dm %ds' %(m,s)


rnn = RNN(vocabulary_size, 100, vocabulary_size)
rnn = rnn.cuda()

print_every = 50
plot_every = 50
all_losses = []
total_loss = 0 # Reset every plot_every iters

start = time.time()

#print(len(X_train),len(y_train))

for i in range(len(y_train)):

    output, loss = train(X_train[i], y_train[i])
    #print(output.shape)
    total_loss += loss

    if i % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), i, i / len(y_train) * 100, loss))

    if i % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        print('all_losses',all_losses)
        total_loss = 0
        _, hidden_of_model_ret = sample(start_word='SENTENCE_START',hidden_of_model = rnn.hidden)
        rnn.hidden = hidden_of_model_ret

max_length = 20


    






