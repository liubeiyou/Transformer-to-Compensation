import sys
from tqdm import tqdm
from model.vit_model import *

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean').to(device)
    accu_loss = torch.zeros(1).to(device)
    bce_loss = torch.zeros(1).to(device)
    bce_loss10 = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        labels = labels.to(device)
        pred = model(images.to(device))
        loss = loss_function(pred, labels)
        bem_loss = bem_au(pred,labels)
        bce10_loss = bem_au_10(pred,labels)
        loss.backward()
        accu_loss += loss.detach()
        bce_loss += bem_loss
        bce_loss10 += bce10_loss
        data_loader.desc = "[train epoch {}] loss: {:.3f} bem:{:.3f} bem_10:{:.3f}".format(epoch,accu_loss.item() / (step + 1), bce_loss.item() / (step + 1),bce_loss10.item() / (step + 1))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), bce_loss.item() / (step + 1)


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()
    loss_function = torch.nn.L1Loss().to(device)
    accu_loss = torch.zeros(1).to(device)
    bce_loss = torch.zeros(1).to(device)
    bce_loss10 = torch.zeros(1).to(device)

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        labels = labels.to(device)
        pred = model(images.to(device))
        loss = loss_function(pred, labels)
        bem_loss = bem_au(pred, labels)
        bce10_loss = bem_au_10(pred, labels)
        accu_loss += loss.detach()
        bce_loss += bem_loss
        bce_loss10 += bce10_loss
        data_loader.desc = "[valid epoch {}] loss: {:.3f} bem:{:.3f} bem_10:{:.3f}".format(epoch, accu_loss.item() / (
                    step + 1), bce_loss.item() / (step + 1), bce_loss10.item() / (step + 1))
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

    return accu_loss.item() / (step + 1), bce_loss.item() / (step + 1),bce_loss10.item() / (step + 1)


