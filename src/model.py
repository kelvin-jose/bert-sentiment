import config
import transformers
import torch.nn as nn


class BertBaseUncased(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids,  masks, token_type_ids):
        _, output = self.bert(ids, attention_mask=masks, token_type_ids=token_type_ids)
        after_drop_out = self.bert_drop(output)
        out = self.out(after_drop_out)
        return out


