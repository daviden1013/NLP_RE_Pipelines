# -*- coding: utf-8 -*-
import abc
from typing import List, Dict, Tuple
import os
import re
import numpy as np
import pandas as pd
import json
from itertools import combinations
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm


class Annotation_loader:
  def __init__(self, max_dist:int, constrains:List[Tuple[str, str]]=None):
    """
    This class inputs annotation and constrains, outputs a pd.DataFrame of eneity-pairs
    to train model for relation extraction.

    Parameters
    ----------
    max_dist : int
      Maximum char distance to consider relations. Any entity pairs with > max_dist 
      will be assumed no relation, thus NOT included in output.
    constrains : List[Tuple[str, str]], optional
      List of valid entity-type pairs to model. The default is None.
    """
    self.max_dist = max_dist
    self.constrains = constrains

  def _apply_constrains(self, entity_1_type:str, entity_2_type:str) -> bool:
    """
    This method checks if 2 entity types need to be modeled based on constrains.
    """
    for t in self.constrains:
      if (entity_1_type == t[0] and entity_2_type == t[1]) or (entity_1_type == t[1] and entity_2_type == t[0]):
        return True
    return False
  
  @abc.abstractmethod
  def get_relations(self) -> pd.DataFrame:
    """
    Call _get_relations() on each annotation

    Returns
    -------
    pd.DataFrame
      DataFrame with document_id, entity_1 (entity ID), entity_2 (entity ID), 
      entity_1_type, entity_1_start, entity_1_end,
      entity_2_type, entity_2_start, entity_2_end,
      relation_id (optional), relation_type
    """
    return NotImplemented


class BRAT_annotation_loader(Annotation_loader):
  def __init__(self, max_dist:int, ann_dir:str, constrains:List[Tuple[str, str]]=None):
    """ This is a child annotation loader for BRAT """
    super().__init__(max_dist, constrains)
    self.ann_dir = ann_dir
    
  def _get_relations(self, ann:List[str]) -> pd.DataFrame:
    """
    This method inputs a list of annotation lines and outputs a DataFrame of 
    entity combinations. Some combo has negative (no) relation.
    entity_1's ID is always < entity_2's ID.
    When entity_1's start and entity_2's start > max_dist, we do not include in output 
    (default to no relation).

    Parameters
    ----------
    ann : List
      line of annotation
    constrains : List
      List of possible entity type combo to consider relation. If not None, it 
      will only reture the combo in list.

    Returns
    -------
    pd.DataFrame
      a DataFrame with relation_id, relation_type, entity_1, entity_2, 
      entity_1_type, entity_1_start, entity_1_end
      entity_2_type, entity_2_start, entity_2_end
    """
    rel_list = []
    entity_list = []
    for l in ann:
      if l[0] == 'R':
        res = {}
        vec = l.split()
        res['relation_id'] = vec[0]
        res['relation_type'] = vec[1]
        res['entity_1'] = vec[2].replace('Arg1:', '')
        res['entity_2'] = vec[3].replace('Arg2:', '')
        if res['entity_1'] > res['entity_2']:
          res['entity_1'], res['entity_2'] = res['entity_2'], res['entity_1']
          
        rel_list.append(res)
      elif l[0] == 'T':
        res = {}
        vec = l.split()
        res['entity_id'] = vec[0]
        res['entity_type'] = vec[1]
        res['start'] = int(vec[2])
        i = 3
        while True:
          if ';' in vec[i]:
            i += 1
          else:
            res['end'] = int(vec[i])
            break
        entity_list.append(res)
    
    if len(rel_list) == 0:
      return None
    
    rel_df = pd.DataFrame(rel_list)
    entity_df = pd.DataFrame(entity_list)

    comb = [sorted(c) for c in combinations(entity_df['entity_id'], 2)]
    comb = pd.DataFrame(comb, columns=['entity_1', 'entity_2'])
    
    rel = pd.merge(comb, rel_df, on=['entity_1', 'entity_2'], how='left')
    for i in [1,2]:
      rel = pd.merge(rel, entity_df, left_on=f'entity_{i}', right_on='entity_id', how='left')
      rel.drop(columns=['entity_id'], inplace=True)
      rel.rename(columns={'entity_type':f'entity_{i}_type', 
                          'start':f'entity_{i}_start', 
                          'end':f'entity_{i}_end'}, inplace=True)
    
    rel = rel.loc[abs(rel['entity_2_start'] - rel['entity_1_start']) <= self.max_dist]
    rel['relation_type'] = rel['relation_type'].fillna('No_relation')
    if self.constrains is not None:
      rel = rel.loc[rel.apply(lambda x:self._apply_constrains(x.entity_1_type, x.entity_2_type), axis=1)]
    
    return rel
  
  
  def get_relations(self) -> pd.DataFrame:
    """
    Call _get_relations() on each annotation

    Returns
    -------
    pd.DataFrame
      DataFrame with document_id, entity_1 (entity ID), entity_2 (entity ID), 
      entity_1_type, entity_1_start, entity_1_end,
      entity_2_type, entity_2_start, entity_2_end,
      relation_id (optional), relation_type
    """
    files = os.listdir(self.ann_dir)
    loop = tqdm(files, total=len(files), leave=True)
    rel_list = []
    for file in loop:
      loop.set_description(f"Processing {file}")
      with open(os.path.join(self.ann_dir, file), 'r') as f:
        ann = f.readlines()
      rel = self._get_relations(ann)
      if rel is not None:
        rel['document_id'] = file.replace('.ann', '')
        rel_list.append(rel)
    
    rel_df = pd.concat(rel_list)[['document_id', 'entity_1', 'entity_2', 
                                  'entity_1_type', 'entity_1_start', 'entity_1_end',
                                  'entity_2_type', 'entity_2_start', 'entity_2_end',
                                  'relation_id', 'relation_type']]
    
    return rel_df
  

class Label_studio_annotation_loader(Annotation_loader):
  """ 
  This is a child annotation loader for Label-studio 
  Since Label-studio doesn't allow specify relation type during annotation,
  The rel_type (pull from config) defines it. 
  
  rel_type : Dict[Tuple[str], str]
    Definition of relation types. Dictionary key is a 2-tuple of entity types. 
    Value is relation type.
  """
  def __init__(self, max_dist:int, ann_file:str, ID:str, rel_type:Dict[Tuple[str], str], 
               constrains:List[Tuple[str, str]]=None):
    super().__init__(max_dist, constrains)
    self.ann_file = ann_file
    self.ID = ID
    self.rel_type = {}
    for (e1, e2), rel in rel_type.items():
      self.rel_type[(e1, e2)] = rel
      self.rel_type[(e2, e1)] = rel
    
  def _get_relations(self, ann:Dict) -> pd.DataFrame:
    """
    This method inputs a json of 1 document's annotation and outputs a DataFrame of 
    entity combinations. Some combo has negative (no) relation.
    entity_1's ID is always < entity_2's ID.
    When entity_1's start and entity_2's start > max_dist, we do not include in output 
    (default to no relation).

    Parameters
    ----------
    ann : List
      json of annotation
    constrains : List
      List of possible entity type combo to consider relation. If not None, it 
      will only reture the combo in list.

    Returns
    -------
    pd.DataFrame
      a DataFrame with relation_id, relation_type, entity_1, entity_2, 
      entity_1_type, entity_1_start, entity_1_end
      entity_2_type, entity_2_start, entity_2_end
    """
    entity_list = []
    rel_list = []
    for r in ann['annotations'][0]['result']:
      if r['type']=='labels':
        entity_list.append({'entity_id':r['id'], 
                            'entity_type':r['value']['labels'][0], 
                            'start':r['value']['start'], 
                            'end':r['value']['end']})
      elif r['type']=='relation':
        if r['from_id'] < r['to_id']:
          rel_list.append({'entity_1':r['from_id'], 'entity_2':r['to_id'], 'relation_type':'Related'})
        else:
          rel_list.append({'entity_1':r['to_id'], 'entity_2':r['from_id'], 'relation_type':'Related'})
                
    if len(rel_list) == 0:
      return None
        
    rel_df = pd.DataFrame(rel_list)
    entity_df = pd.DataFrame(entity_list)

    comb = [sorted(c) for c in combinations(entity_df['entity_id'], 2)]
    comb = pd.DataFrame(comb, columns=['entity_1', 'entity_2'])
    rel = pd.merge(comb, rel_df, on=['entity_1', 'entity_2'], how='left')
    for i in [1,2]:
      rel = pd.merge(rel, entity_df, left_on=f'entity_{i}', right_on='entity_id', how='left')
      rel.drop(columns=['entity_id'], inplace=True)
      rel.rename(columns={'entity_type':f'entity_{i}_type', 
                          'start':f'entity_{i}_start', 
                          'end':f'entity_{i}_end'}, inplace=True)
    
    # Filter by max_dist
    rel = rel.loc[abs(rel['entity_2_start'] - rel['entity_1_start']) <= self.max_dist]
    
    # Filter by constrains
    if self.constrains is not None:
      rel = rel.loc[rel.apply(lambda x:self._apply_constrains(x.entity_1_type, x.entity_2_type), axis=1)]
    
    # Assign relation types
    def assign_relation(e1:str, e2:str, r:str) -> str:
      if r == 'Related' and (e1, e2) in self.rel_type:
        return self.rel_type[(e1, e2)]
      return 'No_relation'
      
    rel['relation_type'] = rel.apply(lambda x:assign_relation(x.entity_1_type, x.entity_2_type, x.relation_type), axis=1)
    
    return rel
  
  
  def get_relations(self) -> pd.DataFrame:
    """
    Call _get_relations() on each annotation

    Returns
    -------
    pd.DataFrame
      DataFrame with document_id, entity_1 (entity ID), entity_2 (entity ID), 
      entity_1_type, entity_1_start, entity_1_end,
      entity_2_type, entity_2_start, entity_2_end,
      relation_id (optional), relation_type
    """
    with open(self.ann_file, encoding='utf-8') as f:
      annotation = json.loads(f.read())
    loop = tqdm(annotation, total=len(annotation), leave=True)
    rel_list = []
    for ann in loop:
      loop.set_description(f"Processing {ann['data'][self.ID]}")
      rel = self._get_relations(ann)
      if rel is not None:
        rel['document_id'] = ann['data'][self.ID]
        rel_list.append(rel)        
    
    rel_df = pd.concat(rel_list)[['document_id', 'entity_1', 'entity_2', 
                                  'entity_1_type', 'entity_1_start', 'entity_1_end',
                                  'entity_2_type', 'entity_2_start', 'entity_2_end',
                                  'relation_type']]
    
    return rel_df
  
  def get_text(self) -> pd.DataFrame:
    with open(self.ann_file, encoding='utf-8') as f:
      annotation = json.loads(f.read())
    loop = tqdm(annotation, total=len(annotation), leave=True)
    text_list = []
    for ann in loop:
      loop.set_description(f"Processing {ann['data'][self.ID]}")
      text = {}
      text['document_id'] = ann['data'][self.ID]
      text['text'] = ann['data']['text']
      text_list.append(text)
  
    text_df = pd.DataFrame(text_list)
    return text_df
  

class RE_Dataset(Dataset):
  def __init__(self, 
               rel_df: pd.DataFrame,
               doc_dict: Dict[str,str], 
               tokenizer: AutoTokenizer, 
               label_map: Dict[str,int],
               word_seq_lenght: int=32, 
               token_seq_length: int=64,
               has_label: bool=True):
    """
    This class inputs a DataFrame of relations and outputs a tuple of tensors
    for model training, evaluation and prediction.

    Parameters
    ----------
    rel_df : pd.DataFrame
      DataFarme with document_id, entity_1 (entity ID), entity_2 (entity ID), 
                      entity_1_type, entity_1_start, entity_1_end,
                      entity_2_type, entity_2_start, entity_2_end,
                      relation_type.
      Note: this is all comination of entities. Some entity-pairs have 
      no relation (relation_type="No_relation")                     
    doc_dict : Dict[str:str]
      dict of {document_id: content} for feature creation.
    tokenizer : AutoTokenizer
      tokenizer.
    label_map : Dict[str:int]
      dict of {relation_type: int} defined as in config.
    word_seq_lenght : int, optional
      number of words to include in input text. The default is 32.
    token_seq_length : int, optional
      number of wordpiece tokens to input. The default is 64.
    has_label : bool, optional
      Indicator, for training and evaluation should be True; for prediction False.
      The default is True.
    """
    self.rel_df = rel_df
    self.doc_dict = doc_dict
    self.tokenizer = tokenizer
    self.label_map = label_map
    self.word_seq_lenght = word_seq_lenght
    self.token_seq_length = token_seq_length
    self.has_label = has_label
    
    self.doc_word_list = {}
    self.doc_pos_list = {}
    for doc_id, text in self.doc_dict.items():
      self.doc_word_list[doc_id] = []
      self.doc_pos_list[doc_id] = []
      for it in re.finditer('\S+', text):
        self.doc_word_list[doc_id].append(it.group())
        self.doc_pos_list[doc_id].append(it.start())
      
      self.doc_pos_list[doc_id] = np.array(self.doc_pos_list[doc_id])
    
  def __len__(self) -> int:
    """
    This method returns the number of instances to feed in the model.

    Returns
    -------
    int
      number of instances.
    """
    return self.rel_df.shape[0]
    
  def __getitem__(self, idx:int) -> tuple:
    """
    This method inputs an index and output the record. For torch.DataLoader 
    internal use. 
    """
    rec = self.rel_df.iloc[idx]
    feature = self._get_text(rec['entity_1_start'], rec['entity_1_end'], 
                             rec['entity_2_start'], rec['entity_2_end'],
                             self.doc_word_list[rec['document_id']], 
                             self.doc_pos_list[rec['document_id']])
    
    tokens = self.tokenizer(feature, add_special_tokens=False, padding='max_length', truncation=True, 
                       max_length=self.token_seq_length)
    
    if self.has_label:
      tokens['label'] = self.label_map[rec['relation_type']]
    
    for v in tokens.keys():
      tokens[v] = torch.tensor(tokens[v])
    
    return tokens
    
  def _get_text(self, entity_1_start:int, entity_1_end:int, 
                entity_2_start:int, entity_2_end:int, 
                word_list:List[str], pos_arr:np.array) -> str:
    """
    This method inputs a start entity pos and end entity pos, calculates the midpoint
    and return a string that has self.word_seq_lenght words around the midpoint

    Parameters
    ----------
    entity_1_start : int
      char pos of entity's start.
    entity_1_end : int
      char pos of entity's end.
    entity_2_start : int
      char pos of entity's start.
    entity_2_end : int
      char pos of entity's end.
    word_list : List[str]
      List of words for a given document text
    pos_arr : np.array
      array of start positions for a given document text. Correspoind to word_list.

    Returns
    -------
    str
      sequence that centered with midpoint of two entities.
    """
    if entity_1_start > entity_2_start:
      entity_1_start, entity_2_start = entity_2_start, entity_1_start
    if entity_1_end > entity_2_end:
      entity_1_end, entity_2_end = entity_2_end, entity_1_end
    
    
    entity_1_start_idx = np.abs((pos_arr - entity_1_start)).argmin()
    entity_1_end_idx = np.abs((pos_arr - entity_1_end)).argmin()
    entity_2_start_idx = np.abs((pos_arr - entity_2_start)).argmin()
    entity_2_end_idx = np.abs((pos_arr - entity_2_end)).argmin()
    
    span = int((self.word_seq_lenght - (entity_2_end_idx - entity_1_start_idx))/2)
    start_idx = max(entity_1_start_idx - span, 0)
    end_idx = min(entity_2_end_idx + span, len(word_list)-1)
    
    text_list = word_list[start_idx:entity_1_start_idx] + ['[E]'] + \
                  word_list[entity_1_start_idx:entity_1_end_idx] + ['[\E]'] + \
                    word_list[entity_1_end_idx:entity_2_start_idx] + ['[E]'] + \
                      word_list[entity_2_start_idx:entity_2_end_idx] + ['[\E]'] + \
                        word_list[entity_2_end_idx:end_idx]
    return ' '.join(text_list)
    

class RE_Trainer():
  def __init__(self, run_name: str, model, n_epochs: int, train_dataset: Dataset, 
               batch_size: int, optimizer, 
               valid_dataset: Dataset=None, shuffle: bool=True, drop_last: bool=True,
               save_model_mode: str=None, save_model_path: str=None, log_path: str=None):

    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.run_name = run_name
    self.model = model
    self.model.to(self.device)
    self.n_epochs = n_epochs
    self.batch_size = batch_size
    self.optimizer = optimizer
    self.valid_dataset = valid_dataset
    self.shuffle = shuffle
    self.save_model_mode = save_model_mode
    self.save_model_path = os.path.join(save_model_path, self.run_name)
    if save_model_path != None and not os.path.isdir(self.save_model_path):
      os.makedirs(self.save_model_path)
    self.best_loss = float('inf')
    self.train_dataset = train_dataset
    self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                   shuffle=self.shuffle, drop_last=drop_last)
    if valid_dataset != None:
      self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, 
                                     shuffle=False, drop_last=drop_last)
    else:
      self.valid_loader = None
    
    self.log_path = os.path.join(log_path, self.run_name)
    if log_path != None and not os.path.isdir(self.log_path):
      os.makedirs(self.log_path)
    self.tensorboard_writer = SummaryWriter(self.log_path) if log_path != None else None
    self.global_step = 0
    
  def evaluate(self):
    with torch.no_grad():
      valid_total_loss = 0
      for valid_batch in self.valid_loader:
        valid_input_ids = valid_batch['input_ids'].to(self.device)
        valid_attention_mask = valid_batch['attention_mask'].to(self.device)
        valid_labels = valid_batch['label'].to(self.device)
        output = self.model(input_ids=valid_input_ids, 
                            attention_mask=valid_attention_mask, 
                            labels=valid_labels)
        valid_loss = output.loss
        valid_total_loss += valid_loss.item()
      return valid_total_loss/ len(self.valid_loader)
    
  def train(self):
    for epoch in range(self.n_epochs):
      train_total_loss = 0
      valid_mean_loss = None
      loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True)
      
      for batch_id, train_batch in loop:
        self.optimizer.zero_grad()
        self.global_step += 1
        train_input_ids = train_batch['input_ids'].to(self.device)
        train_attention_mask = train_batch['attention_mask'].to(self.device)
        train_labels = train_batch['label'].to(self.device)
        """ forward """
        output = self.model(input_ids=train_input_ids, 
                            attention_mask=train_attention_mask, 
                            labels =train_labels)
        train_loss = output.loss
        train_total_loss += train_loss.item()
        """ record training log """
        if self.tensorboard_writer != None:
          self.tensorboard_writer.add_scalar("train/loss", train_total_loss/ (batch_id+1), self.global_step)
        """ backward """
        train_loss.backward()
        """ update """
        self.optimizer.step()
        
        """ validation loss at end of epoch"""
        if self.valid_loader != None and batch_id == len(self.train_loader) - 1:
          valid_mean_loss = self.evaluate()
          if self.tensorboard_writer != None:
            self.tensorboard_writer.add_scalar("valid/loss", valid_mean_loss, self.global_step)
        """ print """
        train_mean_loss = train_total_loss / (batch_id+1)
        loop.set_description(f'Epoch [{epoch + 1}/{self.n_epochs}]')
        loop.set_postfix(train_loss=train_mean_loss, valid_loss=valid_mean_loss)
        
      """ end of epoch """
      if self.save_model_mode == 'all':
        self.save_model(epoch, train_mean_loss, valid_mean_loss)
      elif self.save_model_mode == 'best':
        if epoch == 0 or valid_mean_loss < self.best_loss:
          self.save_model(epoch, train_mean_loss, valid_mean_loss)
          
      self.best_loss = min(self.best_loss, valid_mean_loss)
            
  def save_model(self, epoch, train_loss, valid_loss):
    torch.save(self.model.state_dict(), 
               os.path.join(self.save_model_path, 
                            f'Epoch-{epoch}_trainloss-{train_loss:.4f}_validloss-{valid_loss:.4f}.pth'))


class RE_Predictor:
  def __init__(self, 
               model:AutoModelForSequenceClassification,
               tokenizer:AutoTokenizer, 
               dataset: Dataset,
               label_map:Dict,
               batch_size:int):
    """
    This class inputs a fine-tuned model and a dataset. Returns predicted relations. 

    Parameters
    ----------
    model : AutoModelForSequenceClassification
      A fine-tuned model.
    tokenizer : AutoTokenizer
      tokenizer.
    dataset : Dataset
      dataset to predict.
    label_map : Dict
      DESCRIPTION.
    batch_size : int
      DESCRIPTION.
    """
    
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.model = model
    self.model.to(self.device)
    self.model.eval()
    self.tokenizer = tokenizer
    self.label_map = label_map
    self.batch_size = batch_size
    self.dataset = dataset
    self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

  def predict(self) -> pd.DataFrame:
    """
    This method use the model to predict. Return a DataFrame with 
    probability, predicted relation, and dummies. 
    
    Returns
    -------
    pred_df : pd.DataFrame
      DataFrame contains XX_prob, XX_pred, pred

    """
    pred_list = []
    loop = tqdm(enumerate(self.dataloader), total=len(self.dataloader), leave=True)
    for i, batch in loop:  
      input_ids = batch['input_ids'].to(self.device)
      attention_mask = batch['attention_mask'].to(self.device)
      with torch.no_grad():
        p = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pred = p.logits.softmax(dim=-1).cpu().tolist()
        pred_list.extend(pred)
    
    pred_df = pd.DataFrame(pred_list, columns=[f'{v}_prob' for v in self.label_map.keys()])
    pred_df['pred'] = pred_df.idxmax(axis=1)
    dummy = pd.get_dummies(pred_df['pred'])
    dummy.columns = [c.replace('_prob', '_pred') for c in dummy.columns]
    pred_df = pd.concat([pred_df, dummy], axis=1)
    pred_df['pred'] = pred_df['pred'].str.replace('_prob', '')
    return pred_df
  
  
def evaluate(df:pd.DataFrame, label_map:Dict[str, int]) -> Dict[str, Dict[str,float]]:
  eva = {}
  gold = pd.get_dummies(df['relation_type'])
  for k in label_map.keys():
    eva[k] = {}
    eva[k]['Precision'] = precision_score(gold[k], df[f'{k}_pred'], zero_division=0)
    eva[k]['Recall'] = recall_score(gold[k], df[f'{k}_pred'], zero_division=0)
    eva[k]['F1'] = f1_score(gold[k], df[f'{k}_pred'], zero_division=0)
    eva[k]['AUROC'] = roc_auc_score(gold[k], df[f'{k}_prob'])
    eva[k]['Accuracy'] = accuracy_score(gold[k], df[f'{k}_pred'])
    
  return eva


class Entity_pair_maker:
  def __init__(self, entities:pd.DataFrame, max_dist:int, constrains:List[Tuple[str, str]]=None):
    """
    This class inputs a list of entities (from NER) and constrains, 
    outputs a pd.DataFrame of eneity-pairs for relation extraction prediction.
    Return dataframe has columns: 'document_id', 'entity_1', 'entity_2', 
          'entity_1_type', 'entity_1_start', 'entity_1_end',
          'entity_2_type', 'entity_2_start', 'entity_2_end'
    entity_1 is always < entity_2 (compare str of IDs). No duplicates.

    Parameters
    ----------
    entities : pd.DataFrame
      Entities from NER. Must have columns: document_id, start, end, pred (entity_type)
    max_dist : int
      Maximum char distance to consider relations. Any entity pairs with > max_dist 
      will be assumed no relation, thus NOT included in output.
    constrains : List[Tuple[str, str]], optional
      List of valid entity-type pairs to model. The default is None.
    """
    self.max_dist = max_dist
    self.constrains = constrains
    self.entities = entities
    
  def _apply_constrains(self, entity_1_type:str, entity_2_type:str) -> bool:
    """
    This method checks if 2 entity types need to be modeled based on constrains.
    """
    for t in self.constrains:
      if (entity_1_type == t[0] and entity_2_type == t[1]) or (entity_1_type == t[1] and entity_2_type == t[0]):
        return True
    return False
  
    
  def get_entity_pairs(self) -> pd.DataFrame:
    # All cominations of entities
    comb = pd.merge(self.entities, self.entities, on='document_id')
    comb.rename(columns={'entity_id_x':'entity_1', 'entity_id_y':'entity_2',
                     'pred_x':'entity_1_type', 'start_x':'entity_1_start', 'end_x':'entity_1_end',
                     'pred_y':'entity_2_type', 'start_y':'entity_2_start', 'end_y':'entity_2_end'
                     }, inplace=True)
    # Drop duplicates, set entity_1 < entity_2
    def dedup(document_id:str, entity_1:str, entity_2:str, entity_1_type:str, entity_1_start:int,
              entity_1_end:int, entity_2_type:str, entity_2_start:int, entity_2_end:int):
      if entity_1 < entity_2:
        return (document_id, entity_1, entity_2, entity_1_type, entity_1_start, entity_1_end, entity_2_type,
                entity_2_start, entity_2_end)
      return (document_id, entity_2, entity_1, entity_2_type, entity_2_start, entity_2_end, 
              entity_1_type, entity_1_start, entity_1_end)
    
    sorted_comb = comb.apply(lambda x:dedup(x.document_id, x.entity_1, x.entity_2, x.entity_1_type,
                              x.entity_1_start, x.entity_1_end, x.entity_2_type, 
                              x.entity_2_start, x.entity_2_end), axis=1)
    
    comb = pd.DataFrame(list(sorted_comb), columns=['document_id', 'entity_1', 'entity_2', 
                                                    'entity_1_type', 'entity_1_start', 'entity_1_end',
                                                    'entity_2_type', 'entity_2_start', 'entity_2_end'])
    comb.drop_duplicates(inplace=True)
    comb = comb.loc[abs(comb['entity_2_start'] - comb['entity_1_start']) <= self.max_dist]
    
    if self.constrains is not None:
      comb = comb.loc[comb.apply(lambda x:self._apply_constrains(x.entity_1_type, x.entity_2_type), axis=1)]
    
    return comb.reset_index(drop=True)
    
    