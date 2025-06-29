import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from dataset_manager import all_cats, all_dogs
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import matplotlib.pyplot as plt
import os

# Глобальная история accuracy
accuracy_history = {"train_acc": [], "val_acc": []}

# Dataset
class MyDataset(Dataset):
    def __init__(self, all_cats, all_dogs):
        self.images = np.concatenate([all_cats, all_dogs])  # Корректное объединение
        self.outputs = np.array([[0, 1] if i < len(all_cats) else [1, 0] for i in range(len(self.images))])
        self.transform = transforms.Compose([
            transforms.RandomCrop(100),
        ])
        print("Dataset size: ", self.__len__())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])
        if self.transform:
            image = self.transform(image)
        return image, self.outputs[idx]

# Detector
class Detector(nn.Module):
    def __init__(self, bn: bool = True):
        super(Detector, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 8, 5, 2, 2),
            nn.BatchNorm2d(8) if bn else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.BatchNorm2d(8) if bn else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.BatchNorm2d(16) if bn else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16) if bn else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32) if bn else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32) if bn else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64) if bn else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64) if bn else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128) if bn else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128) if bn else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )

        self.last_part = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*4*128, 2),
            nn.Softmax(dim=1)  # Указал dim=1 для корректной классификации
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.last_part(x)
        return x

# Данные
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_cats, val_cats, train_dogs, val_dogs = train_test_split(
    all_cats, all_dogs, test_size=0.15, random_state=52, shuffle=True
)

train_dataset = MyDataset(train_cats, train_dogs)
val_dataset = MyDataset(val_cats, val_dogs)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True)
detector = Detector(True).to(device)
optimizer = optim.Adam(detector.parameters(), lr=0.0005, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()

# Методы для эпох
def epoch(model, optimizer, criterion, dataloader, is_train=True):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in dataloader:
            images, labels = images.to(device), torch.tensor(labels, dtype=torch.float32).to(device)
            optimizer.zero_grad() if is_train else None

            outputs = model(images)
            loss = criterion(outputs, torch.argmax(labels, dim=1))  # CrossEntropyLoss ожидает индексы классов

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, true_labels = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == true_labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    if is_train:
        accuracy_history["train_acc"].append(accuracy)
    else:
        accuracy_history["val_acc"].append(accuracy)
    return avg_loss, accuracy

def go_epochs(model, optimizer, criterion, train_dataloader, val_dataloader, epochs=10):
    for i in range(epochs):
        train_loss, train_acc = epoch(model, optimizer, criterion, train_dataloader, is_train=True)
        val_loss, val_acc = epoch(model, optimizer, criterion, val_dataloader, is_train=False)
        print(f"Epoch {i+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

def test():
    from PIL import Image
    import dataset_manager as dm
    test_path = os.path.join("test", os.listdir("test")[0])
    image = np.transpose(np.expand_dims(((np.array(Image.open(test_path).convert("RGB").resize(dm.image_size), dtype=np.float32) / 127.5) - 1), 0), (0, 3, 1, 2))
    print(image.shape)
    print(image.min())
    print(image.max())
    image = torch.tensor(image)
    prediction = detector(image)
    return prediction.detach().numpy()[0]
# Сервер
app = FastAPI()

@app.get("/train/{epochs_count}")
async def train_model(epochs_count: int = 10):
    go_epochs(detector, optimizer, criterion, train_dataloader, val_dataloader, epochs_count)
    return JSONResponse(content={"message": "Training completed", "epochs": epochs_count})

@app.get("/test")
async def train_model():
    dog_prop, cat_prop = test().tolist()
    return JSONResponse(content={"dog": dog_prop, "cat": cat_prop})

@app.get("/plot-accuracy/")
async def plot_accuracy():
    if not accuracy_history["train_acc"] or not accuracy_history["val_acc"]:
        return JSONResponse(content={"message": "No accuracy data available. Train the model first."})

    epochs = range(1, len(accuracy_history["train_acc"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracy_history["train_acc"], label="Train Accuracy", marker="o")
    plt.plot(epochs, accuracy_history["val_acc"], label="Validation Accuracy", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    # Сохраняем график
    plot_path = "accuracy_plot.png"
    plt.savefig(plot_path)
    plt.close()

    return JSONResponse(content={"message": "Accuracy plot saved as accuracy_plot.png", "plot_path": plot_path})

# Запуск сервера
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)