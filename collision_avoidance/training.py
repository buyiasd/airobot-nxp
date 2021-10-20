import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from math import ceil


dataset = datasets.ImageFolder(
    'dataset',
    transforms.Compose([
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
)

TEST_SIZE = 40
TRAIN_SIZE = len(dataset) - TEST_SIZE
BATCH_SIZE = 4
NUM_EPOCHS = 30
BEST_MODEL_PATH = 'mini_model.pth'
batch_num = ceil(TRAIN_SIZE / BATCH_SIZE)


train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - TEST_SIZE, TEST_SIZE])

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

model = models.alexnet(pretrained=True)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)


best_accuracy = 0.0

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(NUM_EPOCHS):
    with tqdm(
        total=batch_num,
        desc=f'Epoch {epoch+1}/{NUM_EPOCHS}',
        unit='it',
        bar_format='{l_bar}{bar:30}{r_bar}'
    ) as pbar:
        for images, labels in iter(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'batch_loss': loss.item()})
            pbar.update(1)

    test_error_count = 0.0
    for images, labels in iter(test_loader):
        outputs = model(images)
        test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))

    test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))
    print(f'Test accuracy: {test_accuracy}')
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        best_accuracy = test_accuracy
