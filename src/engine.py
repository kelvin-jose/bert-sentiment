import torch.nn as nn
import torch


def loss(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))


def train(data_loader, model, optimizer, scheduler, device):
    model.train()

    for ix, batchx in enumerate(data_loader):

        print('------- Batch {} -------'.format(ix))
        ids = batchx['input_ids']
        masks = batchx['attention_masks']
        t_ids = batchx['token_type_ids']
        target = batchx['target']

        ids = ids.to(device, dtype=torch.long)
        masks = masks.to(device, dtype=torch.long)
        t_ids = t_ids.to(device, dtype=torch.long)
        target = target.to(device, dtype=torch.float)

        optimizer.zero_grad()
        prediction = model(ids=ids, masks=masks, token_type_ids=t_ids)

        _loss = loss(prediction, target)
        _loss.backward()
        optimizer.step()
        scheduler.step()


def eval(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for step, d in enumerate(data_loader):
            ids = d['input_ids']
            masks = d['attention_masks']
            t_ids = d['token_type_ids']
            target = d['target']

            ids = ids.to(device)
            token_type_ids = t_ids.to(device)
            mask = masks.to(device)
            targets = target.to(device)

            outputs = model(
                ids=ids,
                masks=mask,
                token_type_ids=token_type_ids
            )
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets





