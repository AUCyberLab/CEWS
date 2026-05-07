from transformers.activations import gelu
import torch # If there's a GPU available...
import numpy as np
import pandas as pd
from argparse import ArgumentParser

from torch import nn
from torch import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from torch.utils.data import RandomSampler,SequentialSampler
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, CosineEmbeddingLoss

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from transformers import AutoModelForMaskedLM
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.models.roberta.modeling_roberta import RobertaLMHead
from transformers import AutoConfig
from torch.optim import AdamW

from transformers import get_linear_schedule_with_warmup
from transformers import DataCollatorForLanguageModeling

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint
#from pytorch_lightning.trainer.supporters import CombinedLoader

class BaseModelDistillBert(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        self._frozen = False
        
        config = AutoConfig.from_pretrained(self.hparams.pretrained,                                            
                                            output_attentions=False,
                                            output_hidden_states=False)
        self.config=config
        
        A = AutoModel 
        self.base_model = A.from_pretrained(self.hparams.pretrained, config=config)                
                
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)
        
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        
        self.CELoss = CrossEntropyLoss()

        print('Base: ', type(self.base_model))
    

    def forward(self, batch):
        
        outputs = self.base_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )        
        
        hidden_states = outputs[0]  # (bs, seq_length, dim)
        
        masked_lm_loss=None
        if 'labels' in batch:        
            labels=batch['labels']      
            
            prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
            prediction_logits = gelu(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)

            masked_lm_loss = self.CELoss(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))
            
            del labels, prediction_logits
        
        
        pooled_output = hidden_states[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled = self.dropout(pooled_output)  # (bs, dim)

        del batch
        del outputs
        
        if masked_lm_loss is not None: 
            masked_lm_loss = masked_lm_loss.view(1)
        
        return (masked_lm_loss, pooled)
        


class BaseModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        self._frozen = False
        
        config = AutoConfig.from_pretrained(self.hparams.pretrained,                                            
                                            output_attentions=False,
                                            output_hidden_states=False)
        self.config=config
        
        A = AutoModel #AutoModelForMaskedLM, AutoModelForSequenceClassification
        self.base_model = A.from_pretrained(self.hparams.pretrained, config=config)                
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        if self.hparams.pretrained in ['bert-base-uncased','bert-large-uncased']: 
            self.lm_cls = BertOnlyMLMHead(config)
                    
        elif self.hparams.pretrained in ['roberta-base','roberta-large']:
            self.lm_cls = RobertaLMHead(config)
                    
        print("LM: ",type(self.lm_cls))
        
        self.CELoss = CrossEntropyLoss()

        print('Base: ', type(self.base_model))
        
        print("Freezing Layers:-")
        if self.hparams.freeze=='True':                
            if self.hparams.pretrained in ['bert-base-uncased','roberta-base']:
                self.freeze('layer.9')
            elif self.hparams.pretrained in ['bert-large-uncased', 'roberta-large']:
                self.freeze('layer.21')            
            else:
                print("Nothing Frozen")
        elif self.hparams.freeze=='False':
            print("Nothing Frozen")
        
        else:
            self.freeze(self.hparams.freeze)
                        

    def forward(self, batch):
        
        outputs = self.base_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )        
        
        masked_lm_loss=None
        if 'labels' in batch:        
            labels=batch['labels']      
            prediction_scores = self.lm_cls(outputs.last_hidden_state)            
            masked_lm_loss = self.CELoss(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            del labels, prediction_scores
        
        pooled=outputs.pooler_output        
        pooled=self.dropout(pooled)
        
        del batch
        del outputs
        
        if masked_lm_loss is not None: 
            masked_lm_loss = masked_lm_loss.view(1)
        
        return (masked_lm_loss, pooled)
    
    def freeze(self,layername) -> None:        
        for name, param in self.base_model.named_parameters():            
#             print(name)            
            if layername in name:
                print("Froze upto: ", name)
                break
            else:
                param.requires_grad = False
                
        self._frozen = True
        




class LinkPredictionModel(pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        #self.lc_1 = nn.Linear(config.hidden_size*2,2)
        #self.lc_1 = nn.Linear(config.hidden_size,2)
        
        self.lc_1 = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.lc_2 = nn.Linear(config.hidden_size,2)
        
        self.tanh=nn.Tanh()
        self.softmax=nn.Softmax(dim=1)
        self.CELoss = CrossEntropyLoss()
    
    def forward(self, CVE_vectors, CWE_vectors, true_links=None):
        logits = self.lc_1(torch.cat((torch.abs(CVE_vectors-CWE_vectors),CVE_vectors*CWE_vectors), 1))
        #logits = self.lc_1(CVE_vectors*CWE_vectors)
        logits = self.lc_2(self.tanh(logits))
        
        loss=None
        if true_links is not None:
            loss=self.CELoss(logits,true_links)     
            
        if loss is not None: 
            loss = loss.view(1)
            
        return (loss, logits)


class Model(pl.LightningModule):
    def __init__(self,*args, **kwargs):
        super().__init__()
        self.save_hyperparameters() 
        
        if self.hparams.pretrained == "distilbert-base-uncased":
            self.base_model=BaseModelDistillBert(*args, **kwargs)
        else:        
            self.base_model=BaseModel(*args, **kwargs)
        self.link_model=LinkPredictionModel(self.base_model.config, *args, **kwargs)
        

    def forward(self, batch, CWE_pooled):                
        lm_loss, CVE_pooled = self.base_model(batch)
        
        CVE_vectors=CVE_pooled[batch['CVE_index']]
        CWE_vectors=CWE_pooled[batch['CWE_index']]
        true_links=batch['true_labels']
    
        (loss, logits)=self.link_model(CVE_vectors,CWE_vectors, true_links)        

        del CVE_vectors, CWE_vectors, batch
        
        loss=loss.mean()
        
        if lm_loss is not None:
            loss+= ((self.hparams.lm_lambda)*lm_loss.mean())

        return (loss, logits, true_links)

        
    def configure_optimizers(self):
        
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in self.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            0.01
        }, {
            'params': [
                p for n, p in self.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            0.0
        }]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate,
                          eps=1e-8 # args.adam_epsilon  - default is 1e-8.
                          )

        
        # We also use a scheduler that is supplied by transformers.
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0, # Default value in run_glue.py
            num_training_steps=self.hparams.num_training_steps)

        return optimizer, scheduler
