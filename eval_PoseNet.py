from PIL import Image, ImageDraw
from nets.PoseNet import poseNet2D
import numpy as np
from torchvision import transforms
from matplotlib import pyplot as plt

net = unet(pretrained=True, checkpoint_path="checkpoint_PoseNet.pth.tar" ,n_channels=3,n_classes=21)

transform = transforms.ToTensor()
img = Image.open("/home/gcp/mnt/data/training__/color/00000.png")
img_draw = ImageDraw.Draw(img)
img_tensor = transform(img)
img_tensor.unsqueeze_(0)
print(np.shape(img_tensor))

outputs = net(img_tensor)
print(np.shape(outputs))
outputs.squeeze_(0)
#img = outputs.detach().numpy()
xs = []
ys = []
for i in range(21):
    elem = outputs[i,:,:].detach().numpy()
    print(elem)
    plt.imshow(Image.fromarray(elem))
    plt.savefig("posenet_heatmap{}.png".format(i))
    pos = np.where(elem == np.amax(elem))
    xs.append(pos[0])
    ys.append(pos[1])

plt.imshow(img)
plt.scatter(xs, ys, color="r")
plt.savefig("posenet_eval_overlay.png")
