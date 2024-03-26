import torch
from torch import nn, optim
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def train_and_evaluate(model, train_loader, valid_loader, epochs=20, lr=1e-3):
    
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    loss_train_epoch = []
    loss_valid_epoch = []
    for epoch in tqdm(range(epochs), ncols=100):
        model.train()
        loss_tmp = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            data, mask = batch['light_curve']
            pred = model.forward(data, mask)
            loss = criterion(pred, batch['label'])
            loss.backward()
            optimizer.step()
            loss_tmp += loss.detach()
        loss_train_epoch.append(loss_tmp/len(train_loader.dataset))
        
        model.eval()
        loss_tmp = 0.0
        for batch in valid_loader:
            data, mask = batch['light_curve']
            pred = model.forward(data, mask)
            loss = criterion(pred, batch['label'])
            loss_tmp += loss.detach()
        loss_valid_epoch.append(loss_tmp/len(valid_loader.dataset))    
    
    fig, ax = plt.subplots()
    ax.plot(loss_train_epoch, label='train')
    ax.plot(loss_valid_epoch, label='validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    y_pred_cm = []
    y_true_cm = []
    with torch.no_grad():
        for batch in valid_loader:
            data, mask = batch['light_curve']
            y_pred_cm.append(model.forward(data, mask).argmax(dim=1))
            y_true_cm.append(batch['label'])
    y_pred_cm = torch.cat(y_pred_cm)
    y_true_cm = torch.cat(y_true_cm)    
    ConfusionMatrixDisplay.from_predictions(y_pred=y_pred_cm, y_true=y_true_cm, colorbar=False);