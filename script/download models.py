# -*- coding: utf-8 -*-
PATH = r'E:\David projects\ADE medication RE'

import os

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
tokenizer.save_pretrained(os.path.join(PATH, 'ClinicalBERT'))

from transformers import AutoModelForSequenceClassification
base_model = AutoModelForSequenceClassification.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', num_labels=9)
base_model.save_pretrained(os.path.join(PATH, 'ClinicalBERT'))
