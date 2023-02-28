# -*- coding: utf-8 -*-
import argparse
from easydict import EasyDict
import pprint
import yaml
import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch.optim as optim
from Utilities import RE_Dataset, RE_Trainer
from datetime import datetime

def main():
  print('Training pipeline starts:')
  print(datetime.now())
  """ load config """
  parser = argparse.ArgumentParser()
  add_arg = parser.add_argument
  add_arg("-c", "--config", help='path to config file', type=str)
  args = parser.parse_known_args()[0]
  
  with open(args.config) as yaml_file:
    config = EasyDict(yaml.safe_load(yaml_file))
  
  print('Config loaded:')
  pprint.pprint(config)
  print(datetime.now())
  """ Load development id """
  with open(config['deve_id_file']) as f:
    lines = f.readlines()
  dev_id = [line.strip() for line in lines]
  """ Train-valid split """
  valid_id = np.random.choice(dev_id, int(len(dev_id) * config['valid_ratio']), replace=False).tolist()
  train_id = [i for i in dev_id if i not in valid_id]
  
  rel = pd.read_pickle(os.path.join(config['entity_pairs_path'], config['run_name']+'.pickle'))
  """ Load parameters for dataset """
  files = os.listdir(config['txt_dir'])
  doc_dict = {}
  for file in files:
    with open(os.path.join(config['txt_dir'], file), 'r') as f:
      doc_dict[file.replace('.txt', '')] = f.read()
    
  label_map = config['label_map']
  
  tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
  
  train_dataset = RE_Dataset(rel_df=rel.loc[rel['document_id'].isin(train_id)], 
                              doc_dict=doc_dict,
                              tokenizer=tokenizer, 
                              label_map=label_map, 
                              word_seq_lenght=config['word_token_length'], 
                              token_seq_length=config['wordpiece_token_length'],
                              has_label=True)
  
  valid_dataset = RE_Dataset(rel_df=rel.loc[rel['document_id'].isin(valid_id)], 
                              doc_dict=doc_dict,
                              tokenizer=tokenizer, 
                              label_map=label_map, 
                              word_seq_lenght=config['word_token_length'], 
                              token_seq_length=config['wordpiece_token_length'],
                              has_label=True)
  print('Training dataset created')
  print(datetime.now())
  
  """ define model """
  model = AutoModelForSequenceClassification.from_pretrained(config['base_model'], num_labels=len(label_map))
  optimizer = optim.Adam(model.parameters(), lr=float(config['lr']))
  
  print('Model loaded')
  print(datetime.now())
  """ Training """
  trainer = RE_Trainer(run_name=config['run_name'], 
                        model=model,
                        n_epochs=config['n_epochs'],
                        train_dataset=train_dataset,
                        batch_size=config['batch_size'],
                        optimizer=optimizer,
                        valid_dataset=valid_dataset,
                        save_model_mode='best',
                        save_model_path=os.path.join(config['out_path'], 'checkpoints'),
                        log_path=os.path.join(config['out_path'], 'logs'))
  
  trainer.train()

if __name__ == '__main__':
  main()