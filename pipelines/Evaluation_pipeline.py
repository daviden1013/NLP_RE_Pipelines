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
from Utilities import RE_Dataset, RE_Predictor, evaluate
from datetime import datetime
import pprint

def main():
  print('Evaluation pipeline starts:')
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
  """ load test_id """
  with open(config['test_id_file']) as f:
    lines = f.readlines()
  test_id = [line.strip() for line in lines]
  
  rel = pd.read_pickle(os.path.join(config['entity_pairs_path'], config['run_name']+'.pickle'))
  """ Load parameters for dataset """
  files = os.listdir(config['txt_dir'])
  doc_dict = {}
  for file in files:
    with open(os.path.join(config['txt_dir'], file), 'r') as f:
      doc_dict[file.replace('.txt', '')] = f.read()
    
  label_map = config['label_map']
  
  tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
  test_rel = rel.loc[rel['document_id'].isin(test_id)].reset_index()
  
  test_dataset = RE_Dataset(rel_df=test_rel, 
                              doc_dict=doc_dict,
                              tokenizer=tokenizer, 
                              label_map=label_map, 
                              word_seq_lenght=config['word_token_length'], 
                              token_seq_length=config['wordpiece_token_length'],
                              has_label=False)
  """ load model """
  best = AutoModelForSequenceClassification.from_pretrained(config['base_model'], num_labels=len(label_map))
  
  if config['checkpoint'] == 'best':
    model_names = [f for f in os.listdir(config['checkpoint_dir']) if '.pth' in f]
    best_model_name = sorted(model_names, key=lambda x:int(re.search("-(.*?)_", x).group(1)))[-1]
    print(f'Evaluate model: {best_model_name}')
    print(datetime.now())
    best.load_state_dict(torch.load(os.path.join(config['checkpoint_dir'], best_model_name), 
                                    map_location=torch.device('cpu')))
  
  else:
    print(f"Evaluate model: {config['checkpoint']}")
    print(datetime.now())
    best.load_state_dict(torch.load(os.path.join(config['checkpoint_dir'], config['checkpoint']), 
                                    map_location=torch.device('cpu')))

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  best.to(device)
  best.eval()

  """ Prediction """
  predictor = RE_Predictor(model=best,
                           tokenizer=tokenizer,
                           dataset=test_dataset,
                           label_map=label_map,
                           batch_size=config['batch_size'])
  
  pred_df = predictor.predict()

  """ Evaluate """
  test_rel = pd.concat([test_rel, pred_df], axis=1)
  evaluation = evaluate(test_rel, label_map)
  evaluation = pd.DataFrame(evaluation).transpose()

  """ Output evaluation """
  eval_dir = os.path.join(config['out_path'], 'evaluations', config['run_name'])
  if not os.path.isdir(eval_dir):
    os.makedirs(eval_dir)
    
  evaluation.to_csv(os.path.join(eval_dir, f"{config['run_name']} evaluation.csv"), index=True)
  

if __name__ == '__main__':
  main()