# -*- coding: utf-8 -*-
import argparse
from easydict import EasyDict
import pprint
import yaml
import os
import torch.optim as optim
from Utilities import Annotation_loader
from datetime import datetime

def main():
  print('Entity-pairs pipeline starts:')
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
  """ make training datasets """
  loader = BRAT_annotation_loader(max_dist=config['max_dist'], 
                                  ann_dir=config['ann_dir'],
                                  constrains=eval(config['constrains']))
  rel = loader.get_relations()
  """ Save """
  if not os.path.isdir(config['entity_pairs_path']):
    os.mkdir(config['entity_pairs_path'])
  rel.to_pickle(os.path.join(config['entity_pairs_path'], config['run_name']+'.pickle'))
  print('Entity-pairs saved.')
  print(datetime.now())
  
if __name__ == '__main__':
  main()
