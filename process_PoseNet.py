import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
from tqdm import tqdm
from torch.autograd import Variable
from scipy.ndimage.filters import gaussian_filter
import torchvision.transforms as transforms


crop_padding = 15 #padding on each side in pixels
data_folder = "../mnt/data/evaluation___"

resize_to_tensor = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])
resize = transforms.Resize((128,128))
resizeCompose = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

sets = ['training', 'evaluation']
for set in sets:
    print("Started preparing", set, "dataset")
    data_path = os.path.join('data', set + '_PoseNet')
    # Load annotations of training set
    with open(os.path.join('data', set, 'anno_evaluation.pickle'), 'rb') as fi:
        annotations_in = pickle.load(fi)

    if not os.path.exists(data_path):
            os.makedirs(data_path)
    if not os.path.exists(data_path + '/color'):
            os.makedirs(data_path + '/color')

    sample_id_out = 0
    for sample_id in tqdm(range(len(annotations_in))):
        image_in = np.array(Image.open(os.path.join('data', set, 'color', '%.5d.png' % sample_id)))

        # get info from annotation dictionary
        anno = annotations_in[sample_id]
        kp_coord_uv = anno['uv_vis'][:, :2]  # u, v coordinates of 42 hand keypoints, pixel
        kp_visible = (anno['uv_vis'][:, 2] == 1)  # visibility of the keypoints, boolean

        # keypoints of hand1
        hand1 = kp_coord_uv[:21]
        hand1v = kp_visible[:21]
        hand2 = kp_coord_uv[21:]
        hand2v = kp_visible[21:]

        # Process Hand 1
        try:
            x1,y1 = int(min(hand1[hand1v,0])-crop_padding), int(min(hand1[hand1v,1])-crop_padding)
            x2,y2 = int(max(hand1[hand1v,0])+crop_padding), int(max(hand1[hand1v,1])+crop_padding)
            if(x1 < 0 or y1 <0):
                delta_padding = min(x1, y1)
                x1,y1 = int(min(hand1[hand1v,0])-crop_padding+delta_padding), int(min(hand1[hand1v,1])-crop_padding+delta_padding)
                x2,y2 = int(max(hand1[hand1v,0])+crop_padding-delta_padding), int(max(hand1[hand1v,1])+crop_padding-delta_padding)
            img_out = resize(Image.fromarray(image_in[y1:y2,x1:x2]))
            img_out.save(os.path.join(data_path, 'color', '%.5d.png' % sample_id_out))
            heatmap_list = np.array([np.zeros((x2-x1, y2-y1), dtype=np.float32) for i in range(21)])
            heatmap_out = np.array([np.zeros((128, 128), dtype=np.float32) for i in range(21)])
            for i in range(len(hand1)):
                if hand1v[i]:
                    heatmap_list[i][int(hand1[i,0] - x1), int(hand1[i,1] - y1)] = 1
                    heatmap_out[i] = resizeCompose(gaussian_filter(heatmap_list[i], 5))
            with open(os.path.join(data_path, 'color', '%.5d.annotation.pickle' % sample_id_out), 'wb') as pickle_file:
                pickle.dump(heatmap_out, pickle_file)
            sample_id_out += 1
        except ValueError:
           pass

        # Process Hand 2
        try:
            x1,y1 = int(min(hand2[hand2v,0])-crop_padding), int(min(hand2[hand2v,1])-crop_padding)
            x2,y2 = int(max(hand2[hand2v,0])+crop_padding), int(max(hand2[hand2v,1])+crop_padding)
            if(x1 < 0 or y1 <0):
                delta_padding = min(x1, y1)
                x1,y1 = int(min(hand2[hand2v,0])-crop_padding+delta_padding), int(min(hand2[hand2v,1])-crop_padding+delta_padding)
                x2,y2 = int(max(hand2[hand2v,0])+crop_padding-delta_padding), int(max(hand2[hand2v,1])+crop_padding-delta_padding)
            img_out = resize(Image.fromarray(image_in[y1:y2,x1:x2]))
            img_out.save(os.path.join(data_path, 'color', '%.5d.png' % sample_id_out))
            heatmap_list = np.array([np.zeros((x2-x1, y2-y1), dtype=np.float32) for i in range(21)])
            heatmap_out = np.array([np.zeros((128, 128), dtype=np.float32) for i in range(21)])
            for i in range(len(hand1)):
                if hand2v[i]:
                    heatmap_list[i][int(hand2[i,0] - x1), int(hand2[i,1] - y1)] = 1
                    heatmap_out[i] = resizeCompose(gaussian_filter(heatmap_list[i], 5))
            with open(os.path.join(data_path, 'color', '%.5d.annotation.pickle' % sample_id_out), 'wb') as pickle_file:
                pickle.dump(heatmap_out, pickle_file)
            sample_id_out += 1
        except ValueError:
            pass
