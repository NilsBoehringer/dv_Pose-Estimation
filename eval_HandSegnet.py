import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nets.Unet import unet
from Hands_Dataset import Hands, Hands_test
import torchvision.transforms as transforms


batch_size = 1


train_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    #transforms.Normalize((0.0341, 0.0335, 0.0327), (0.0189, 0.0183, 0.0200))
])

test_dataset = Hands_test(csv_file="test_dataset.csv",root_dir="evaluation/color/",transform=train_transformation)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=3)

model = unet(pretrained=True, checkpoint_path="checkpoint_HandSegNet.pth.tar" ,n_channels=3,n_classes=2)
model.cuda()
model.eval()
classIndex=1


for i, (images, labels) in enumerate(test_loader, 0):
    out = model(images.cuda())
    for n, image in enumerate(images, 0):
        image = np.array(image)
        pred = out
        mask = pred[n][1]
        mask = np.array((mask).cpu().detach())
        mask = (mask>0.011).astype(np.uint8)
        fig = plt.figure(figsize=(20,20))
        ax1 = plt.subplot(221)
        ax1.imshow(image[0], interpolation='nearest', cmap="gray")
        ax1.imshow(mask, alpha=0.5, cmap="gray")

        intersection = mask-abs(np.array(labels[n])-mask)
        intersection =(intersection > 0).astype(np.uint8)

        union = abs(np.array(labels[n])+mask)
        union = (union > 0).astype(np.uint8)
        iou_score = sum(sum(intersection))/ sum(sum(union))
        print("Accuracy: " + str(iou_score*100) + "%")

        ax2 = plt.subplot(222)
        ax2.imshow(labels[n], cmap="gray")

        plt.savefig("test_perf", dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
        plt.show()
