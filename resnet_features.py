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
from torchvision.models import resnet18, resnet34
# Глобальная история accuracy
accuracy_history = {"train_acc": [], "val_acc": []}

resnet = resnet34(pretrained=True)
modules = list(resnet.children())[:-2]  # Убираем avgpool и fc
resnet18_fm = nn.Sequential(*modules)
for param in resnet18_fm.parameters():
    param.requires_grad = False

# Dataset
class MyDataset(Dataset):
    def __init__(self, all_cats, all_dogs):
        self.images = np.concatenate([all_cats, all_dogs])  # Форма (N, 100, 100, 3)
        self.outputs = np.array([[0, 1] if i < len(all_cats) else [1, 0] for i in range(len(self.images))])
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # Преобразуем numpy в PIL
            transforms.Resize((100, 100)),
            transforms.ToTensor(),  # В [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Применяем трансформации с сохранением формы (C, H, W)
        for i in range(len(self.images)):
            # Обновляем исходные данные
            # self.images[i] = self.images[i].transpose((1, 2, 0))  # (3, 100, 100) -> (100, 100, 3)
            self.images[i] = (self.images[i] + 1) / 2  # Из [-1, 1] в [0, 1]
            transformed = self.transform(self.images[i].transpose((1, 2, 0)))  # (3, 100, 100) как тензор
            self.images[i] = transformed.numpy()  # Сохраняем как (3, 100, 100)
            print(f"Sample {i}: Min = {self.images[i].min():.4f}, Max = {self.images[i].max():.4f}")
        
        if os.path.exists(f'resnet_feature_maps{len(self.images)}.npy'):
            self.fms = np.load(f'resnet_feature_maps{len(self.images)}.npy')
            self.fms = torch.from_numpy(np.array(self.fms)).float()
        else:
            self.fms = []
            for i, image in enumerate(self.images):
                self.fms.append(resnet18_fm(torch.tensor([image]))[-1])
                if i % 100 == 0:
                    print(f"Item {i}/{len(self.images)}")
            
            self.fms = torch.from_numpy(np.array(self.fms)).float()
            np.save(f'resnet_feature_maps{len(self.images)}.npy', self.fms.numpy())
        self.images = None
        print("Dataset size: ", self.__len__())
        print("Dataset shape: ", self.fms.shape)

    def __len__(self):
        return len(self.fms)

    def __getitem__(self, idx):
        image = self.fms[idx]
        return image, self.outputs[idx]

# Detector
class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()

        self.compression_layer = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )
        self.flatten = nn.Flatten()
        self.linear2 = nn.Linear(4*4*512, 2)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # 4x4x512
        # x = self.compression_layer(x)
        x = self.flatten(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Данные
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_cats, val_cats, train_dogs, val_dogs = train_test_split(
    all_cats, all_dogs, test_size=0.15, random_state=52, shuffle=True
)

train_dataset = MyDataset(train_cats, train_dogs)
val_dataset = MyDataset(val_cats, val_dogs)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True)
detector = Detector().to(device)
detector.apply(init_weights)
optimizer = optim.Adam(detector.parameters(), lr=0.0005, weight_decay=0.0001)
optimizer2 = optim.SGD(detector.parameters(), lr=0.0005, momentum=0.9)
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
    image = np.transpose(np.expand_dims((np.array(Image.open(test_path).convert("RGB").resize(dm.image_size), dtype=np.float32) / 255), 0), (0, 3, 1, 2))
    print(image.shape)
    image = torch.tensor(image)
    transformed = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
    image = transformed
    print(image.min())
    print(image.max())
    prediction = detector(resnet18_fm(image))
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

@app.get("/change_lr/{coefficient}")
async def change_learning_rate(coefficient: float):
    if coefficient <= 0:
        return JSONResponse(content={"error": "Coefficient must be positive"}, status_code=400)

    current_lr_adam = optimizer.param_groups[0]['lr']
    current_lr_sgd = optimizer2.param_groups[0]['lr']

    new_lr_adam = current_lr_adam * coefficient
    new_lr_sgd = current_lr_sgd * coefficient

    optimizer.param_groups[0]['lr'] = new_lr_adam
    optimizer2.param_groups[0]['lr'] = new_lr_sgd

    return JSONResponse(content={
        "message": "Learning rate updated",
        "old_lr_adam": current_lr_adam,
        "new_lr_adam": new_lr_adam,
        "old_lr_sgd": current_lr_sgd,
        "new_lr_sgd": new_lr_sgd
    })

# Запуск сервера
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)