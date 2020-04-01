import config
import torch


class BertDataset:
    def __init__(self, text, target):
        self.text = text
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        encoded_dict = self.tokenizer.encode_plus(
            self.text[item],
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_attention_mask=True
        )

        return {
            'input_ids': torch.tensor(encoded_dict['input_ids'], dtype=torch.long),
            'attention_masks': torch.tensor(encoded_dict['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(encoded_dict['token_type_ids'], dtype=torch.long),
            'target': torch.tensor(torch.tensor(self.target[item], dtype=torch.float))
        }

