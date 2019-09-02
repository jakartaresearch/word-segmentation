import os
import pickle
import pandas as pd
import numpy as np

def get_label_index(text_list):
    text = "".join(text_list)
    chars = sorted(list(set(text)))
    
    word2idx = dict((w, i) for i, w in enumerate(chars))
    idx2word = dict((i, w) for i, w in enumerate(chars))
    
    return word2idx, idx2word

def get_flag_space(sentence):
    no_space = []
    flag_space = []
    sentence = str(sentence)
    for char in sentence: 
        if char != ' ':
            no_space.append(char)
            flag_space.append('0')
        elif char == ' ':
            flag_space[-1] = '1'
            
    no_space = ''.join(no_space)
    flag_space = ''.join(flag_space)
    return flag_space

def char_vectorizer(list_inputs, char_indices, MAX_LENGTH):
    x = np.zeros((len(list_inputs), MAX_LENGTH, len(char_indices)))
    for i, input_ in enumerate(list_inputs):
        for t, char in enumerate(input_):
            x[i, t, char_indices[char]] = 1
    
    return x

# flag_space example = '000010000010000'
def get_source_with_space(input,flag_space):
    source_space=''
    for a,b in zip(input,flag_space):
        if b=='1':
            a0=a+' '
            source_space+=a0
        else:
            source_space+=a
    return source_space