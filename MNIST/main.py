import os
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
from constants import *
from model import Network
from functions import TripletLoss, MNISTDataset, plot_metric, init_weights

os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = MNIST(root=dataset_path, train=True, download=True, transform=transforms.ToTensor())
# test_dataset = MNIST(root=dataset_path, train=False, download=True, transform=transforms.ToTensor())
train_imgs = torch.vstack(list(train_dataset[i][0] for i in range(0, len(train_dataset), 60)))  # 1000, 28, 28
train_labels = torch.tensor(list(train_dataset[i][1] for i in range(0, len(train_dataset), 60)))  # 1000
val_imgs = torch.vstack(list(train_dataset[i][0] for i in range(1, len(train_dataset), 60)))  # 1000, 28, 28
val_labels = torch.tensor(list(train_dataset[i][1] for i in range(1, len(train_dataset), 60)))  # 1000

train_dataset = MNISTDataset(train_imgs, train_labels, is_train=True, transform=None)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataset = MNISTDataset(val_imgs, val_labels, is_train=True, transform=None)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

model = Network(embedding_dims)
model.apply(init_weights)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = TripletLoss()

losses = {'train': [], 'val': []}
for epoch in tqdm(range(epochs), desc="Train/Val Epochs"):
    train_loss = 0.
    model.train()
    for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(train_loader):
        optimizer.zero_grad()
        anchor_out = model(anchor_img.to(device))
        positive_out = model(positive_img.to(device))
        negative_out = model(negative_img.to(device))
        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    losses['train'].append(train_loss / batch_size / len(train_loader))

    val_loss = 0.
    model.eval()
    with torch.no_grad():
        for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(val_loader):
            anchor_out = model(anchor_img.to(device))
            positive_out = model(positive_img.to(device))
            negative_out = model(negative_img.to(device))
            loss = criterion(anchor_out, positive_out, negative_out)
            val_loss += loss.item()
    losses['val'].append(val_loss / batch_size / len(val_loader))

    print("\nEpoch {} Train_loss {:.4f} Val_loss {:.4f}".format(epoch,
                                                              losses['train'][-1], losses['val'][-1]))

plot_metric(losses, title='Loss')

train_embeddings = []
labels = []
model.eval()
with torch.no_grad():
    for img, _, _, label in tqdm(train_loader, desc='Create class embeddings'):
        train_embeddings.append(model(img.to(device)).cpu().numpy())
        model.calc_class_embeddings(img.to(device), label.to(device))
        labels.append(label)
    class_embeddings = model.class_embeddings.cpu().numpy() / np.expand_dims(
                                                                model.class_embeddings_num.cpu().numpy(), 1)
train_results = np.concatenate(train_embeddings)
labels = np.concatenate(labels)

accuracy = 0.
model.eval()
with torch.no_grad():
    for img, _, _, label in tqdm(val_loader, desc='Predict'):
        accuracy += (model.predict(img.to(device)) == label).sum().item()
print('Val_accuracy:', round(accuracy / batch_size / len(val_loader) * 100., 1), '%')

plt.figure(figsize=(15, 10), facecolor="azure")
for label in np.unique(labels):
    tmp = train_results[labels == label]
    plt.scatter(tmp[:, 0], tmp[:, 1], label=label)
plt.scatter(class_embeddings[:, 0], class_embeddings[:, 1], c='black', s=200, marker='>')
plt.legend()
plt.show()

model.eval()
with torch.no_grad():
    random_imgs = []
    labels = np.unique(val_labels)
    for label in labels:
        img = random.choice(val_imgs[val_labels == label])
        random_imgs.append(img.reshape(1, 1, 28, 28))
    random_imgs = torch.vstack(random_imgs)
    random_embeddings = model(random_imgs.to(device)).cpu().numpy()

plt.figure(figsize=(15, 10))
colors = cm.rainbow(np.linspace(0, 1, len(labels)))
for label in labels:
    plt.scatter(random_embeddings[label, 0], random_embeddings[label, 1], label=label, color=colors[label])
plt.scatter(class_embeddings[:, 0], class_embeddings[:, 1], s=300, marker='>', color=colors)
plt.legend()
plt.show()

