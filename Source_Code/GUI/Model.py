import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel

bert_layer = BertModel.from_pretrained("bert-base-uncased")

class BERT_CNN(nn.Module):
    def __init__(self, num_classes):
        super(BERT_CNN, self).__init__()
        self.bert = bert_layer
        self.dropout = nn.Dropout(0.5)
        self.conv1 = nn.Conv1d(768, 128, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              return_dict=True)
        last_hidden_state = output.last_hidden_state
        x = self.dropout(last_hidden_state)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x, _ = torch.max(x, dim=-1)
        x = self.fc(x)
        return x

 

class BERT_BiLSTM(nn.Module):
    def __init__(self, num_labels=6, dropout=0.1):
        super(BERT_BiLSTM, self).__init__()
        
        self.num_labels = num_labels
        self.bert = bert_layer
        self.bilstm = nn.LSTM(input_size=768, hidden_size=256, num_layers=2, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=2 * 256, out_features=self.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)
        last_hidden_state = output.last_hidden_state
        bilstm_out, _ = self.bilstm(last_hidden_state.permute(1,0,2))
        bilstm_out = bilstm_out.permute(1,0,2)
        bilstm_out = self.dropout(bilstm_out[:,-1,:])
        logits = self.fc(bilstm_out)
        
        return logits

 

class BERTClass(nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert_model = bert_layer
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 6)
    
    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids, 
            attention_mask=attn_mask, 
            token_type_ids=token_type_ids,
            return_dict=True
        )

        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output

 
 
class XLNetClassification(nn.Module):
    def __init__(self, num_classes):
        super(XLNetClassification, self).__init__()
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        xlnet_output = self.xlnet(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            return_dict=True
        )

        last_hidden_state = xlnet_output.last_hidden_state
        avg_pool = torch.mean(last_hidden_state, 1)
        avg_pool = self.dropout(avg_pool)
        logits = self.fc(avg_pool)
        return logits
