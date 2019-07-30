import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from nets.Unet import unet

model = unet(pretrained=True,n_channels=3,n_classes=2)

model.cuda()
model.eval()
classIndex=1

cam_transformation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128,128))])

cv2.namedWindow("preview",cv2.WINDOW_NORMAL)
cv2.resizeWindow('preview', 600,600)
cv2.namedWindow("mask",cv2.WINDOW_NORMAL)
cv2.resizeWindow('mask', 600,600)
cv2.namedWindow("hand",cv2.WINDOW_NORMAL)
cv2.resizeWindow('hand', 600,600)

vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    original = cam_transformation(frame)
    original = np.array(original)
    original = np.reshape(original,(128,128,3))

    frame = train_transformation(frame)
    frame = frame.view(1,3,128,128)

    out = model(frame.cuda())
    if (out.shape != None):

        pred = out
        mask = pred[0][1]
        mask = np.array((mask).cpu().detach())
        mask = (mask>0.8).astype(np.uint8)

        out = torch.max(out.data, 1)
        out = np.array(out[1].cpu().detach())


        mask_n = mask*255

        rect = cv2.boundingRect(mask_n)

        # function that computes the rectangle of interest
        #ax1.plot((rect[0]+rect[2]),(rect[1]+rect[3]),'ro')
        #ax1.plot((rect[0]),(rect[1]+rect[3]),'ro')
        #ax1.plot((rect[0]+rect[2]),(rect[1]),'ro')
        #ax1.plot((rect[3]),(rect[1]),'ro')

        cropped_img = original[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]  # crop the image to the desired rectangle
        if (cropped_img.shape[0] > 0):
            if (cropped_img.shape[1] > 0):
                cv2.imshow("hand",np.array(cropped_img))


        cv2.imshow("preview", np.array(original))
        cv2.imshow("mask",mask*255)

    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("preview")
cv2.destroyWindow("mask")
cv2.destroyWindow("hand")
