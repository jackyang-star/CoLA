import numpy as np
import torch
from torch import optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from model.model import MVCG, MyDataset, EarlyStopping


# 参数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
patience = 25
delta = 0
ifverbose = False
emb_dim = 64
num_epochs = 1000
batch_size = 64
train_validation_ratio = 0.8
dropout_rate = 0.1
head_num = 8
iftrain = True


def execute_train():
    # 读入数据集
    dataset = MyDataset('../dataset/processed/dataset/train_dataset.npy')  # 创建自定义数据集对象
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型，定义损失函数和优化器
    model = MVCG(emb_dim, dropout_rate, head_num, iftrain).to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, factor=0.5, patience=15)
    early_stopping = EarlyStopping(patience, delta, ifverbose, path='../model/checkpoint.pt')

    train_epochs_loss = []
    print(f'Train begin. Running on {device}')
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_epoch_loss = []
        for query, cand, label in data_loader:
            optimizer.zero_grad()
            # 数据类型转换&在cuda运行
            query = query.to(torch.float32).to(device)
            cand = cand.to(torch.float32).to(device)
            label = label.to(torch.float32).to(device)
            # 计算相似度
            complementary = model(query, cand)
            # 计算loss
            loss = criterion(complementary, label)
            # 优化参数
            loss.backward()
            optimizer.step()
            # 记录loss
            train_epoch_loss.append(loss.item())
        train_epochs_loss.append(np.average(train_epoch_loss))
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_epochs_loss[epoch]}')

        # 调整学习率
        scheduler.step(train_epoch_loss[-1])

        # 早停
        early_stopping(train_epochs_loss[-1], model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


if __name__ == '__main__':
    execute_train()








