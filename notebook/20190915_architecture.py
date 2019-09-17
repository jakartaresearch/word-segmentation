#!/usr/bin/env python
# coding: utf-8

# experiment 1

# In[ ]:


import warnings
warnings.simplefilter('ignore')

import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm

from sklearn.metrics import hamming_loss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# In[ ]:


device = "cuda" if torch.cuda.is_available() else "cpu"


# In[ ]:


tanggal = datetime.datetime.today().strftime('%Y-%m-%d_%H_%M_%S')


# In[ ]:


MAX_LENGTH = 25
BATCH_SIZE = 10000


# In[ ]:


print("reading data...")


# In[ ]:


data = pickle.load(open(os.path.join('../data/clean/data_clean_100k.res'), 'rb'))


# In[ ]:


d_data = data


# In[ ]:


d_data['source_len']= d_data.source.str.len()


# In[ ]:


d_data_selected = d_data[d_data['source_len'] == MAX_LENGTH]


# In[ ]:


d_data_selected.reset_index(drop=True, inplace=True)


# In[ ]:


idx2char = dict((idx, c) for idx, c in enumerate("abcdefghijklmnopqrstuvwxyz", 1))
char2idx = dict((c, idx) for idx, c in enumerate("abcdefghijklmnopqrstuvwxyz", 1))


# In[ ]:


idx2char[0] = '<UNK>'
char2idx['<UNK>'] = 0


# In[ ]:


def char_vectorizer(list_inputs, char_indices):
    x = np.zeros((len([list_inputs]), MAX_LENGTH, len(char_indices)))
    for i, input_ in enumerate([list_inputs]):
        for t, char in enumerate(input_):
            x[i, t, char_indices[char]] = 1
    
    return x


# In[ ]:


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


# In[ ]:


def char_vectorizer(list_inputs, char_indices):
    x = np.zeros((len([list_inputs]), MAX_LENGTH, len(char_indices)))
    for i, input_ in enumerate([list_inputs]):
        for t, char in enumerate(input_):
            try:
                x[i, t, char_indices[char]] = 1
            except:
                x[i, t, 0] = 1
    
    return x


# In[ ]:


def flag_space_to_list(flag):
    return np.array(list(flag)).astype(int)


# In[ ]:


d_data_selected['flag_space'] = d_data_selected['target'].apply(get_flag_space)


# In[ ]:


d_data_selected.loc[:, 'matrix'] = d_data_selected.loc[:, 'source'].apply(char_vectorizer, args=(char2idx,))


# In[ ]:


d_data_selected['flag_space_array'] = d_data_selected.flag_space.apply(flag_space_to_list)


# In[ ]:


d_data_selected['flag_space_sum'] = d_data_selected.flag_space_array.apply(np.sum)


# In[ ]:


class Dataset():
    def __init__(self, data):
        row, col = data.shape
        train = data.loc[:int(row*.8)]
        test = data.loc[int(row*.8):]
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        
        self.lookup = {
            'train': (train, len(train)),
            'test': (test, len(test))
        }
        
        self.set_split('train')
        
    def set_split(self, split = 'train'):
        self.data, self.length = self.lookup[split]
    
    def __getitem__(self, index):
        X = self.data.loc[index, 'matrix']
        X = torch.Tensor(X).squeeze(0)
        
        y = np.array(list(self.data.loc[index, 'flag_space'])).astype(int)
        y = torch.Tensor(y).squeeze(0)
        
        return {'x': X,
               'y': y}
    
    def __len__(self):
        return self.length


# In[ ]:


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.lstm = nn.LSTM(len(char2idx), 256)
        
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)
        
    def forward(self, input_, apply_sigmoid=False):
        
        y_pred, _ = self.lstm(input_)
        y_pred, _ = self.lstm(input_, _)
        y_pred = self.fc1(y_pred)
        y_pred = self.fc2(y_pred)
        y_pred = self.fc3(y_pred)
        y_pred = self.fc4(y_pred)
        y_pred = self.fc5(y_pred)
        
        if apply_sigmoid:
            y_pred = torch.sigmoid(y_pred)
        
        y_pred = y_pred.squeeze(2)    
        return y_pred


# In[ ]:


def compute_accuracy(y_true, y_pred):
    y_true = y_true.long().cpu().numpy()
    y_pred = (y_pred > 0.5).long().cpu().numpy()
    try:
#         hamming_score = hamming_loss(y_true, y_pred)
#         return 1 - hamming_score
        return (y_true == y_pred).all(axis = 1).mean()
    except:
        print("y_true", y_true, "y_pred", y_pred)


# In[ ]:


dataset = Dataset(d_data_selected)
classifier = Classifier().to(device)


# In[ ]:


loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(classifier.parameters(), lr = 0.001, weight_decay=0.002)


# In[ ]:


history_dict ={
    'acc_train': [],
    'acc_test': [],
    'loss_train': [],
    'loss_test': []
}


# In[ ]:


print("start training...")


# In[ ]:


try:
    for epoch in range(1,101):

        running_loss = 0
        running_acc = 0
        running_loss_val = 0
        running_acc_val = 0

        start = time.time()

        classifier.train()
        dataset.set_split('train')
        data_generator = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
        for batch_index, batch_dict in enumerate(data_generator, 1):
            optimizer.zero_grad()

            y_pred = classifier(batch_dict['x'].to(device))

            loss_train = loss_func(y_pred, batch_dict['y'].to(device))
            loss_item = loss_train.item()
            running_loss += (loss_item - running_loss) / batch_index

            loss_train.backward()

            accuracy_score = compute_accuracy(batch_dict['y'], y_pred)
            running_acc += (accuracy_score - running_acc) / batch_index

            optimizer.step()

        classifier.eval()
        dataset.set_split('test')
        data_generator = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
        for batch_index, batch_dict in enumerate(data_generator, 1):

            y_pred = classifier(batch_dict['x'].to(device))

            loss_train_val = loss_func(y_pred, batch_dict['y'].to(device))
            loss_item_val = loss_train_val.item()
            running_loss_val += (loss_item_val - running_loss_val) / batch_index

            accuracy_score_val = compute_accuracy(batch_dict['y'], y_pred)
            running_acc_val += (accuracy_score_val - running_acc_val) / batch_index
            
        history_dict['acc_train'].append(running_acc)
        history_dict['acc_test'].append(running_acc_val)
        history_dict['loss_train'].append(running_loss)
        history_dict['loss_test'].append(running_loss_val)

        print("{:.2f} sec | epoch {} loss train: {:.2f} accuracy train: {:.2f} loss val {:.2f} accuracy val {:.2f}".format(
            time.time() - start, epoch, running_loss, running_acc, running_loss_val, running_acc_val
        ))
except KeyboardInterrupt:
    print("exit loop")


# In[ ]:


if not os.path.exists('../reports/20190915_architecture/{}'.format(tanggal)):
    os.makedirs("../reports/20190915_architecture/{}".format(tanggal))

plt.plot(history_dict['loss_train'])
plt.plot(history_dict['loss_test'])
plt.ylim(0.0, 1.0)
plt.savefig('../reports/20190915_architecture/{}/loss.png'.format(tanggal))

plt.plot(history_dict['acc_train'])
plt.plot(history_dict['acc_test'])
plt.ylim(0.0, 1.0)
plt.savefig('../reports/20190915_architecture/{}/accuracy.png'.format(tanggal))


pickle.dump(history_dict,open("../reports/20190915_architecture/{}/history.pkl".format(tanggal), 'wb'))


# In[ ]:





# In[ ]:




