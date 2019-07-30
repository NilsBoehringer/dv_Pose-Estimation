import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from nets.Unet import unet
from Hands_Dataset import Hands, Hands_test
import torchvision.transforms as transforms


batch_size = 32
num_epochs = 50
print_freq = 100
learning_rate = 0.001


train_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    #transforms.Normalize((0.0341, 0.0335, 0.0327), (0.0189, 0.0183, 0.0200))
])

train_dataset = Hands(csv_file="dataset.csv",root_dir="training/color/",transform=train_transformation)
val_dataset = Hands_test(csv_file="val_dataset.csv",root_dir="evaluation/color/",transform=train_transformation)

train_loader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle=True, num_workers=3)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=3)

model = unet(pretrained=False, n_channels=3, n_classes=2)
model.cuda()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer= torch.optim.Adam(model.parameters(), lr= learning_rate, amsgrad=True)

# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        model.train()
        # Run the forward pass
        outputs = model(images.cuda())

        #Loss
        loss = criterion(F.log_softmax(outputs), labels.cuda())
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % print_freq == 0:

            # Test the model
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                iou_score_list =[]
                for n, (images, labels) in enumerate(val_loader):
                    out = model(images.cuda())
                    pred = out
                    mask = pred[n][1]
                    mask = np.array((mask).cpu().detach())
                    mask = (mask>0.011).astype(np.uint8)

                    #IoU
                    intersection = mask-abs(np.array(labels[n])-mask)
                    intersection =(intersection > 0).astype(np.uint8)

                    union = abs(np.array(labels[n])+mask)
                    union = (union > 0).astype(np.uint8)

                    iou_score_list.append(sum(sum(intersection))/ sum(sum(union)))

                iou_score = np.mean(np.array(iou_score_list))
                acc_list.append(iou_score)

            torch.save(model.state_dict(),"HandSegNet_checkpoint.pth.tar")
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Validation_Acc: {:.4f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),iou_score*100))



#Loss&Accuracy plot
plt.figure(figsize=(20,10))
plt.subplot(2, 1, 1)
plt.plot(np.convolve(len(loss_list), loss_list))
plt.legend(['loss'], loc='upper left')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.figure(figsize=(20,10))
plt.subplot(2, 1, 2)
plt.plot(np.convolve(len(acc_list), acc_list))
plt.legend(['mean IoU score'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('mean Iou sore')

plt.show()
