# -*- coding: utf-8 -*-
import os
import re
import argparse
from easydict import EasyDict
import yaml
import pandas as pd
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from Utilities import Entity_pair_maker, RE_Dataset, RE_Predictor
from datetime import datetime
import pprint

def main():
  print('Prediction pipeline starts:')
  print(datetime.now())
  parser = argparse.ArgumentParser()
  add_arg = parser.add_argument
  add_arg("-c", "--config", help='path to config file', type=str)
  args = parser.parse_known_args()[0]
  
  with open(args.config) as yaml_file:
    config = EasyDict(yaml.safe_load(yaml_file))

  print('Config loaded:')
  pprint.pprint(config)
  print(datetime.now())
  
  label_map = config['label_map']
  tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
  """ load entity-pairs to predict """
  entities = pd.read_pickle(config['predict_entity_file'])
  maker = Entity_pair_maker(entities=entities,
                            max_dist=config['max_dist'],
                            constrains=eval(config['constrains']))
  entity_pairs = maker.get_entity_pairs()
  """ Load parameters for dataset """
  text = pd.read_pickle(config['predict_doc_text_file'])
  doc_dict = {i:t for i, t in zip(text['document_id'], text['text'])}
    
  label_map = config['label_map']
  
  tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
  
  pred_dataset = RE_Dataset(rel_df=entity_pairs, 
                            doc_dict=doc_dict,
                            tokenizer=tokenizer, 
                            label_map=label_map, 
                            word_seq_lenght=config['word_token_length'], 
                            token_seq_length=config['wordpiece_token_length'],
                            has_label=False)
  """ load model """
  best = AutoModelForSequenceClassification.from_pretrained(config['predict_model'], num_labels=len(label_map))
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  best.to(device)
  best.eval()

  """ Prediction """
  predictor = RE_Predictor(model=best,
                           tokenizer=tokenizer,
                           dataset=pred_dataset,
                           label_map=label_map,
                           batch_size=config['batch_size'])
  
  pred_df = predictor.predict()
  
  pred_df = pd.concat([entity_pairs, pred_df], axis=1)
  pred_df.to_pickle(config['predict_outfile'])

if __name__ == '__main__':
  main()