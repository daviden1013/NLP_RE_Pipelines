# NLP_RE_Pipelines
This is a general relation extraction (RE) toolkit for BERT model training, evaluation, and prediction. The **Entity_pairs_pipeline** processes annotated corpora (with labels and relations) and creates a pandas DataFrame with entity pairs for training and evaluation. A certain constrains of entity types can be set in the config file. A split of development set and evaluation set should be considered. The **Training_pipeline** inputs the documents from development set, and train a BERT-familiy model. Checkpoints are saved to disk. The **Evaluation_pipeline** use a specified checkpoint to predict on the evaluation set, and perform relation classification evaluation with precision, recall, F1, AUROC, and accuracy. Once a final/ production NLP model is made and saved to disk, the **Prediction_pipeline** use it to predict.

Framework: **PyTorch**, **Transformers**

Compatible annotation tools: **Label-studio**, **BRAT**

![alt text](https://github.com/daviden1013/NLP_RE_Pipelines/blob/main/Diagrams.png)
