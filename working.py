import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. Dataset
train = datasets.MNIST(root="data", download=True, train=True, transform=transforms.ToTensor())
dataset = DataLoader(train, batch_size=32)

# 2. Define model
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # 28→26
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3), # 26→24
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), # 24→22
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 22 * 22, 10)
        )

    def forward(self, x):
        return self.model(x)

# 3. Device
device = torch.device("cpu")

clf = ImageClassifier().to(device)
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# 4. Load pretrained model (if available)
try:
    clf.load_state_dict(torch.load("model_state.pt", map_location=device))
    print("Model loaded successfully.")
except:
    print("No pretrained model found, starting fresh.")

# 5. Load custom test image
img = Image.open("number1.webp").convert("L")  # grayscale
img = img.resize((28, 28))
img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)  # shape [1,1,28,28]

# 6. Prediction
clf.eval()
with torch.no_grad():
    pred = torch.argmax(clf(img_tensor))
    print("Predicted digit:", pred.item())

# 7. (Optional) Training loop
# for epoch in range(10):
#     for X, y in dataset:
#         X, y = X.to(device), y.to(device)
#         yhat = clf(X)
#         loss = loss_fn(yhat, y)
#
#         opt.zero_grad()
#         loss.backward()
#         opt.step()
#     print(f"Epoch {epoch} - Loss: {loss.item()}")
#
# with open("model_state.pt", "wb") as f:
#     save(clf.state_dict(), f)
