


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

def save_final_metrics(file_path, model_name, epochs, lr, train_loss, train_acc, val_loss, val_acc):

    import os
    import csv

    write_header = not os.path.exists(file_path)
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model", "epochs", "learning_rate", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
        writer.writerow([model_name, epochs, lr, f"{train_loss:.6f}", f"{train_acc:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}"])


import pandas as pd

def print_metrics_table(file_path):
    pd.set_option('display.max_columns', None) 
    pd.set_option('display.width', 1000) 
    df = pd.read_csv(file_path)
    
    print(df)



def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for boards, castlings, labels in pbar:
        boards, castlings, labels = boards.to(device), castlings.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(boards, castlings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        pbar.set_postfix(loss=running_loss/total, accuracy=correct/total)

    return running_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for boards, castlings, labels in pbar:
            boards, castlings, labels = boards.to(device), castlings.to(device), labels.to(device)
            outputs = model(boards, castlings)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            pbar.set_postfix(loss=running_loss/total, accuracy=correct/total)

    return running_loss / total, correct / total

def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-3, save_path="best_model.pth", model_name="ChessNet"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0

    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with val acc: {best_val_acc:.4f}")

        print("-" * 30)

    
    save_final_metrics("train_analize.csv", model_name=model_name, epochs=epochs, lr=lr, 
                    train_loss=train_losses[-1], train_acc=train_accs[-1], val_loss=val_losses[-1], val_acc=val_accs[-1])


    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Val Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()



import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for boards, castlings, labels in pbar:
        boards, castlings, labels = boards.to(device), castlings.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(boards, castlings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        pbar.set_postfix(loss=running_loss/total, accuracy=correct/total)
    return running_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for boards, castlings, labels in pbar:
            boards, castlings, labels = boards.to(device), castlings.to(device), labels.to(device)
            outputs = model(boards, castlings)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            pbar.set_postfix(loss=running_loss/total, accuracy=correct/total)
    return running_loss / total, correct / total

def train_kfold(model_class, dataset, device, k=5, epochs=10, batch_size=32, lr=1e-3, weight_decay=1e-4, save_path="best_model.pth", model_name="ChessNet"):
    # kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    kfold = KFold(n_splits=k, shuffle=True)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset), 1):
        print(f"Fold {fold}/{k}")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size)
        model = model_class().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        best_val_acc = 0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        for epoch in range(1, epochs+1):
            print(f"Epoch {epoch}/{epochs}")
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
            print(f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model fold {fold} with val acc: {best_val_acc:.4f}")
            print("-" * 30)


        save_final_metrics("train_analize.csv", model_name=model_name, epochs=epochs, lr=lr, 
                    train_loss=train_losses[-1], train_acc=train_accs[-1], val_loss=val_losses[-1], val_acc=val_accs[-1])



        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.title(f"Fold {fold} Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label="Train Accuracy")
        plt.plot(val_accs, label="Val Accuracy")
        plt.title(f"Fold {fold} Accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()



import os
import csv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_one_epoch_3(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data.x, data.edge_index, data.batch)
        loss = criterion(outputs, data.y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.num_graphs
        preds = outputs.argmax(dim=1)
        correct += preds.eq(data.y).sum().item()
        total += data.num_graphs

        pbar.set_postfix(loss=running_loss/total, accuracy=correct/total)
    return running_loss / total, correct / total

def validate_3(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for data in pbar:
            data = data.to(device)
            outputs = model(data.x, data.edge_index, data.batch)
            loss = criterion(outputs, data.y)

            running_loss += loss.item() * data.num_graphs
            preds = outputs.argmax(dim=1)
            correct += preds.eq(data.y).sum().item()
            total += data.num_graphs
            
            pbar.set_postfix(loss=running_loss/total, accuracy=correct/total)

    return running_loss / total, correct / total

def train_model_gnn(model, train_loader, val_loader, device, epochs=10, lr=1e-3, save_path="best_model.pth", model_name="ChessNet"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0

    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss, train_acc = train_one_epoch_3(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_3(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with val acc: {best_val_acc:.4f}")

        print("-" * 30)

    save_final_metrics("train_analize.csv", model_name=model_name, epochs=epochs, lr=lr,
                       train_loss=train_losses[-1], train_acc=train_accs[-1], val_loss=val_losses[-1], val_acc=val_accs[-1])

    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Val Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()



