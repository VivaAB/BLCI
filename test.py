import torch
import pynvml
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import explained_variance_score
import torch.optim as optim


stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val = load_it_data(path_to_data)

# ========== GPU CONFIGURATION ==========
pynvml.nvmlInit()
gpu_index = torch.cuda.current_device()
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
gpu_name = pynvml.nvmlDeviceGetName(handle)
print(f"Using GPU {gpu_index}: {gpu_name}")
device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")

# ========== DYNAMIC BATCH SIZE FUNCTION ==========
def get_max_batch_size(image_size=(3, 224, 224), output_size=128, bytes_per_float=4, safety_margin=0.8, gpu_index=0):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    free_mem = mem_info.free * safety_margin

    sample_mem = (
        torch.tensor(image_size).prod().item() * bytes_per_float +
        output_size * bytes_per_float +
        1024 * bytes_per_float
    )
    max_batch = int(free_mem / sample_mem)
    return min(max_batch, 64)

# ========== DATA AUGMENTATION & TRANSFORM ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ========== LOAD AND CONVERT DATA ==========
X_train_tensor = torch.tensor(stimulus_train, dtype=torch.float32)
y_train_tensor = torch.tensor(spikes_train, dtype=torch.float32)
X_val_tensor = torch.tensor(stimulus_val, dtype=torch.float32)
y_val_tensor = torch.tensor(spikes_val, dtype=torch.float32)

batch_size = get_max_batch_size(output_size=y_train_tensor.shape[1], gpu_index=gpu_index)
print(f"Using batch size: {batch_size}")

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, pin_memory=True)

# ========== LOAD AND MODIFY RESNET ==========
resnet = models.resnet50(pretrained=True)

for param in resnet.parameters():
    param.requires_grad = False  # Freeze all

# Modify the final fully connected layer
n_neurons = y_train_tensor.shape[1]
resnet.fc = nn.Sequential(
    nn.Linear(resnet.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, n_neurons)
)

# Unfreeze deeper layers
for param in list(resnet.layer2.parameters()) + list(resnet.layer3.parameters()) + list(resnet.layer4.parameters()):
    param.requires_grad = True

resnet = resnet.to(device)

# ========== TRAINING SETUP ==========
criterion = nn.MSELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ========== TRAINING LOOP ==========
train_losses = []
val_losses = []
epochs = 80
early_stopping_patience = 8
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    torch.cuda.empty_cache()  # Prevent GPU memory leaks

    resnet.train()
    running_loss = 0
    for inputs, targets in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = resnet(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    resnet.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = resnet(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.4f} - Val Loss: {val_loss:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(resnet.state_dict(), "best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


resnet.eval()
y_pred = []
with torch.no_grad():
    for inputs, _ in val_loader:
        inputs = inputs.to(device)
        outputs = resnet(inputs)
        y_pred.append(outputs.cpu().numpy())

y_pred = np.vstack(y_pred)

ev_values = explained_variance_score(y_val_tensor, y_pred, multioutput='raw_values')
corr_values = np.array([np.corrcoef(y_val_tensor[:, i], y_pred[:, i])[0, 1] for i in range(spikes_val.shape[1])])
print(f"✅ Final Mean EV (Deep ConvNeXt head): {np.mean(ev_values):.4f}")
print(f"✅ Final Mean Correlation (Deep ConvNeXt head): {np.mean(corr_values):.4f}")

