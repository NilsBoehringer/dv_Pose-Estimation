import time
import torch
import torchvision
from PIL import Image
import numpy as np
import os, os.path
import pickle
from torch.utils.data import DataLoader, Dataset
from nets.PoseNet import poseNet2D
from tqdm import tqdm
from torch.autograd import Variable
import matplotlib.pyplot as plt
from nets.Unet import unet


batch_size = 16
n_epochs = 60
print_freq = 100
learning_rate = 0.001


# Custom Data Set to load the images with corresponding annotations
class CustomDataSet(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.transformResize = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128,128)),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        path = os.path.join(self.root_dir, "color")
        return int(len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])/2)

    def __getitem__(self, idx):
        if self.transform is not None:
            image = transform(Image.open(os.path.join(self.root_dir, 'color', '%.5d.png' % idx)))
        else:
            image = Image.open(os.path.join(self.root_dir, 'color', '%.5d.png' % idx))
        with open(os.path.join(self.root_dir, 'color', '%.5d.annotation.pickle' % idx), 'rb') as fi:
            annotations = pickle.load(fi)
        return image, annotations

seed = 42
np.random.seed(seed)
torch.manual_seed

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128,128)),
    torchvision.transforms.ToTensor()
])

train_dataset = CustomDataSet("../mnt/data/training___", transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=4, drop_last=True)
test_dataset = CustomDataSet("../mnt/data/evaluation___", transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=4, drop_last=True)


model = unet(pretrained=False, n_channels=3, n_classes=21)
model.cuda()
model.train()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
loss = torch.nn.MSELoss()
#loss = torch.nn.L1Loss()
eval_loss = torch.nn.L1Loss()

print("Starting Training")
for epoch in range(n_epochs):
    running_loss = 0.0
    start_time = time.time()
    total_train_loss = 0

    for i, (image, annotation) in enumerate(train_loader, 0):
        #Wrap them in a Variable object
        inputs = Variable(image)

        #Set the parameter gradients to zero
        optimizer.zero_grad()

        #Forward pass, backward pass, optimize
        outputs = model(inputs.cuda())
        loss_size = loss(outputs, annotation.cuda())
        loss_size.backward()
        optimizer.step()


        if (i + 1) % print_freq == 0:
            torch.save(model.state_dict(),"PoseNet_checkpoint.pth.tar")
            print('Epoch [{}/{}], Loss: {:.4f}'
                .format(epoch + 1, n_epochs,  loss_size.item()))

    #Print statistics to file for plotting
    accuracy = 0
    iou_score = []
    for i, (image, annotation) in enumerate(test_loader, 0):
        inputs = Variable(image)
        outputs = model(inputs.cuda())
        for j in range(batch_size):
            for k in range(21):
                mask = outputs[j][k]
                mask = np.array((mask).cpu().detach())
                mask = (mask>0.8).astype(np.uint8)

                intersection = mask-abs(np.array(annotation[j][k])-mask)
                intersection =(intersection > 0).astype(np.uint8)
                union = abs(np.array(annotation[j][k])+mask)
                union = (union > 0).astype(np.uint8)
                if(sum(sum(union)) == 0):
                    iou_score.append(0.0)
                else:
                    iou_score.append(sum(sum(intersection))/ sum(sum(union)))
    accuracy = np.mean(iou_score)
    print(accuracy)
    f = open(os.path.join("log_PoseNet.txt"), 'a')
    f.write("{}\t{:.5f}\t{:.4f}".format(epoch + 1, accuracy, loss_size.item())+"\n")
    f.close()
