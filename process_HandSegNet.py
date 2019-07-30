from __future__ import print_function, unicode_literals
import PIL
import pickle
import os
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from tqdm import tqdm


sets = ['training', 'evaluation']
# load annotations of this set
for set in sets:
    print("Started preparing", set, "dataset")
    with open(os.path.join(set, 'anno_%s.pickle' % set), 'rb') as fi:
        anno_all = pickle.load(fi)
    images_cropped = []

    #Example for 10 images
    for sample_id in tqdm(range(len(anno_all))):
        one_hand = False
        anno = anno_all[sample_id]
        image = scipy.misc.imread(os.path.join(set, 'color', '%.5d.png' % sample_id))

        # get info from annotation dictionary
        kp_coord_uv = anno['uv_vis'][:, :2]  # u, v coordinates of 42 hand keypoints, pixel
        kp_visible = (anno['uv_vis'][:, 2] == 1)  # visibility of the keypoints, boolean

        #keypoints of hand1
        hand1 = kp_coord_uv[:21]
        hand1v = kp_visible[:21]
        hand2 = kp_coord_uv[21:]
        hand2v = kp_visible[21:]

        if(len(hand1[hand1v,0]) <= (len(hand2[hand2v,0]))):
            one_hand = True
            hand1 = hand2
            hand1v = hand2v

        #bounding box edges
        x1,y1 = max(hand1[hand1v,0]), max(hand1[hand1v,1])
        x2,y2 = min(hand1[hand1v,0]), min(hand1[hand1v,1])

        #unpadded width and height
        w = abs(x1-x2)
        h = abs(y1-y2)

        #padding 10% hand 1 (Keypoints lie IN the hand -> Bounding box needs padding to cover whole hand)
        pad_x1 =x1+ 0.1 * w
        pad_y1 =y1+ 0.1 * h
        pad_x2 =max(0,(x2 - 0.1 * w))
        pad_y2 =max(0,(y2 - 0.1 * h))


        #bbox width and height
        w = abs(pad_x1-pad_x2)
        h = abs(pad_y1-pad_y2)

        if h < w:
            h = w
        else:
            w = h
        if(w>0):
            if one_hand:
                cropped_image = image[int(round(pad_y2)):int(round(pad_y2))+int(round(h)) , int(round(pad_x2)):int(round(pad_x2))+int(round(w)), :]
            else:
                cropped_image = image[int(round(pad_y2)):int(round(pad_y2))+int(round(h)) , int(round(pad_x2)):int(round(pad_x2))+int(round(w)), :]

            cropped_image = PIL.Image.fromarray(cropped_image)
            images_cropped.append([image])

    print("Writing csv for", set, "dataset")
    dataF = pd.DataFrame((images_cropped))
    dataF.to_csv(path_or_buf="dataset.csv", index=False)
