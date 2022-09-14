from torch import nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self,config):
        super(BertClassifier,self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        #self.bert = BertModel.from_pretrained(config.pretrained_model_dir)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size,256)
        self.relu = nn.ReLU()
        self.output = nn.Linear(256,6)

    def forward(self,input_id,mask):
        _, pooled_output = self.bert(input_ids = input_id,attention_mask = mask, return_dict=False)
        linear_output = self.linear(self.dropout(pooled_output))
        final_layer = self.relu(linear_output)
        output_layer = self.output(final_layer)
        return output_layer