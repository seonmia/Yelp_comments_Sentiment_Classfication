# -*- coding: utf-8 -*-
"""bert_base_cased



# Hyperparameters
"""

used_model = 'bert-base-cased'
cased = False if 'uncased' in used_model else True

train_batch_size = 32
eval_batch_size = 32
test_batch_size = 32

learning_rate = 5e-5
train_epoch = 4
weight_decay = 0.1

wandb_project = "final_project1" # WandB에 넣어둘 프로젝트 이름 
wandb_team = "goorm-project-nlp-team-1" # WandB 팀명


"""# Import requirements"""

import os
import pdb
import argparse
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
import wandb
from time import time

import torch
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from tqdm import tqdm, trange
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    AutoConfig,
    AdamW,
)

wandb.login()



"""# 1. Preprocess"""

def make_id_file(task, tokenizer, cased):
    def make_data_strings(file_name, cased):
        data_strings = []
        with open(os.path.join(file_name), 'r', encoding='utf-8') as f:
            id_file_data = [tokenizer.encode(line if cased else line.lower()) for line in f.readlines()]
        for item in id_file_data:
            data_strings.append(' '.join([str(k) for k in item]))
        return data_strings
  
    print('it will take some times...')
    train_pos = make_data_strings('sentiment.train.1', cased)
    train_neg = make_data_strings('sentiment.train.0', cased)
    dev_pos = make_data_strings('sentiment.dev.1', cased)
    dev_neg = make_data_strings('sentiment.dev.0', cased)

    print('make id file finished!')
    return train_pos, train_neg, dev_pos, dev_neg

tokenizer = BertTokenizer.from_pretrained(used_model)

train_pos, train_neg, dev_pos, dev_neg = make_id_file('yelp', tokenizer, cased)

class SentimentDataset(object):
    #  def __init__(self, pos, neg):
    def __init__(self, tokenizer, pos, neg):
        self.tokenizer = tokenizer
        self.data = []
        self.label = []

        for pos_sent in pos:
            self.data += [self._cast_to_int(pos_sent.strip().split())]
            self.label += [[1]]
        for neg_sent in neg:
            self.data += [self._cast_to_int(neg_sent.strip().split())]
            self.label += [[0]]
    def _cast_to_int(self, sample):
        return [int(word_id) for word_id in sample]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
         sample = self.data[index]
         return np.array(sample), np.array(self.label[index])

train_dataset = SentimentDataset(tokenizer, train_pos, train_neg)
dev_dataset = SentimentDataset(tokenizer, dev_pos, dev_neg)

for i, item in enumerate(train_dataset):
    print(item)
    if i == 10:
        break

def collate_fn_style(samples):
    input_ids, labels = zip(*samples)
    max_len = max(len(input_id) for input_id in input_ids)
    attention_mask = torch.tensor([[1] * len(input_id) + [0] * (max_len - len(input_id)) for input_id in input_ids])
    input_ids = pad_sequence([torch.tensor(input_id) for input_id in input_ids],
                             batch_first=True)
    
    token_type_ids = torch.tensor([[0] * len(input_id) for input_id in input_ids])
    position_ids = torch.tensor([list(range(len(input_id))) for input_id in input_ids])
    labels = torch.tensor(np.stack(labels, axis=0))

    return input_ids, attention_mask, token_type_ids, position_ids, labels

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=train_batch_size,
                                           shuffle=True, collate_fn=collate_fn_style,
                                           pin_memory=True, num_workers=2)
dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=eval_batch_size,
                                         shuffle=False, collate_fn=collate_fn_style,
                                         num_workers=2)

"""# 2. Train"""

random_seed=42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BertForSequenceClassification.from_pretrained(used_model)
model.to(device)

model.train()

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def compute_acc(predictions, target_labels):
    return (np.array(predictions) == np.array(target_labels)).mean()

wandb.init(project=wandb_project, name=used_model+' '+str(int(time()))[-3:], entity=wandb_team)

init_time=time()
lowest_valid_loss = 9999.

train_acc = []
train_loss = []
valid_acc = []
valid_loss = []


curr_train_loss = [] 
curr_train_acc = [] 

report_to ="wandb" 



for epoch in range(train_epoch):
    with tqdm(train_loader, unit="batch") as tepoch:
        for iteration, (input_ids, attention_mask, token_type_ids, position_ids, labels) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            position_ids = position_ids.to(device)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            
            output = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           position_ids=position_ids, 
                           labels=labels)
            loss = output.loss


            logits = output.logits
            batch_predictions = [0 if example[0] > example[1] else 1 for example in logits]
            batch_labels = [int(example) for example in labels]
            
            acc = compute_acc(batch_predictions, batch_labels)
            



            loss.backward()

            optimizer.step()


            curr_train_loss.append(loss.item())
            curr_train_acc.append(acc)



            tepoch.set_postfix(loss=loss.item())
            if iteration != 0 and iteration % int(len(train_loader) / 5) == 0:
                # Evaluate the model five times per epoch
                with torch.no_grad():
                    model.eval()
                    curr_valid_loss = []   # valid_losses 수정 
                    curr_valid_acc = []  
                     

                    for input_ids, attention_mask, token_type_ids, position_ids, labels in tqdm(dev_loader,
                                                                                                desc='Eval',
                                                                                                position=1,
                                                                                                leave=None):
                        input_ids = input_ids.to(device)
                        attention_mask = attention_mask.to(device)
                        token_type_ids = token_type_ids.to(device)
                        position_ids = position_ids.to(device)
                        labels = labels.to(device, dtype=torch.long)

                        output = model(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       position_ids=position_ids,
                                       labels=labels)

                        logits = output.logits
                        loss = output.loss

                        batch_predictions = [0 if example[0] > example[1] else 1 for example in logits]
                        batch_labels = [int(example) for example in labels]


                        curr_valid_loss.append(loss.item())
                        curr_valid_acc.append(compute_acc(batch_predictions, batch_labels))

                

                # loss /acc 계산
                mean_train_acc = sum(curr_train_acc) / len(curr_train_acc)
                mean_train_loss = sum(curr_train_loss) / len(curr_train_loss)
                mean_valid_acc = sum(curr_valid_acc) / len(curr_valid_acc)
                mean_valid_loss = sum(curr_valid_loss) / len(curr_valid_loss)

                train_acc.append(mean_train_acc)
                train_loss.append(mean_train_loss)
                valid_acc.append(mean_valid_acc)
                valid_loss.append(mean_valid_loss)
                
                curr_train_acc = [] 
                curr_train_loss = [] 

                # wandb log 수집 

                wandb.log({ 
                        "Train Loss": mean_train_loss,
                        "Train Accuracy": mean_train_acc,
                        "Valid Loss" : mean_valid_loss, 
                        "Valid Accuracy" : mean_valid_acc

                        })


                ###############

                if lowest_valid_loss > mean_valid_loss:
                    lowest_valid_loss = mean_valid_loss
                    print('Acc for model which have lower valid loss: ', mean_valid_acc)
                    torch.save(model.state_dict(), "./pytorch_model.bin")



fin_time=time()
print('Time:',fin_time-init_time)


"""# 3. Test"""

import pandas as pd
test_df = pd.read_csv('test_no_label.csv')

test_dataset = test_df['Id']

def make_id_file_test(tokenizer, test_dataset, cased):
    data_strings = []
    id_file_data = [tokenizer.encode(sent if cased else sent.lower()) for sent in test_dataset]
    for item in id_file_data:
        data_strings.append(' '.join([str(k) for k in item]))
    return data_strings

test = make_id_file_test(tokenizer, test_dataset, cased)

class SentimentTestDataset(object):
    def __init__(self, tokenizer, test):
        self.tokenizer = tokenizer
        self.data = []

        for sent in test:
            self.data += [self._cast_to_int(sent.strip().split())]

    def _cast_to_int(self, sample):
        return [int(word_id) for word_id in sample]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return np.array(sample)

test_dataset = SentimentTestDataset(tokenizer, test)

def collate_fn_style_test(samples):
    input_ids = samples
    max_len = max(len(input_id) for input_id in input_ids)
    attention_mask = torch.tensor(
        [[1] * len(input_id) + [0] * (max_len - len(input_id)) for input_id in
         input_ids])
    input_ids = pad_sequence([torch.tensor(input_id) for input_id in input_ids],
                             batch_first=True)
    
    
    token_type_ids = torch.tensor([[0] * len(input_id) for input_id in input_ids])
    position_ids = torch.tensor([list(range(len(input_id))) for input_id in input_ids])

    return input_ids, attention_mask, token_type_ids, position_ids

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,
                                          shuffle=False, collate_fn=collate_fn_style_test,
                                          num_workers=2)

with torch.no_grad():
    model.eval()
    predictions = []
    for input_ids, attention_mask, token_type_ids, position_ids in tqdm(test_loader,
                                                                        desc='Test',
                                                                        position=1,
                                                                        leave=None):

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        position_ids = position_ids.to(device)

        output = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids,
                       position_ids=position_ids)

        logits = output.logits
        batch_predictions = [0 if example[0] > example[1] else 1 for example in logits]
        predictions += batch_predictions

test_df['Category'] = predictions

test_df.to_csv('submission_uncased.csv', index=False)

test_df['Category'].value_counts()