import config
import torch
from model import BertBaseUncased


def prediction(text, _model, _tokenizer, _max_len, device):
    encoded = _tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=_max_len,
            pad_to_max_length=True,
            return_attention_mask=True
        )
    ids = torch.tensor(encoded['input_ids'], dtype=torch.long).unsqueeze(0)
    masks = torch.tensor(encoded['attention_mask'], dtype=torch.long).unsqueeze(0)
    t_id = torch.tensor(encoded['token_type_ids'], dtype=torch.long).unsqueeze(0)

    ids = ids.to(device, dtype=torch.long)
    masks = masks.to(device, dtype=torch.long)
    t_id = t_id.to(device, dtype=torch.long)

    output = _model(
        ids=ids,
        masks=masks,
        token_type_ids=t_id
    )

    outputs = torch.sigmoid(output).cpu().detach().numpy()
    return outputs[0][0]


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    _model = BertBaseUncased()
    _model = torch.nn.DataParallel(_model)
    _model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))

    _model.to(device)
    _model.eval()
    _tokenizer = config.TOKENIZER
    _max_len = config.MAX_LEN

    pos = prediction("that sound's really good", _model, _tokenizer, _max_len, device)
    neg = 1 - pos

    print('Pos: {} Neg: {}'.format(pos, neg))