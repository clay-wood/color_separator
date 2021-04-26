import numpy as np
import colorsys
from PIL import Image, ImageDraw
from skimage import data, color, io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from time import time


class SepColors:
    
    def __init__(self, impath, n_colors):
        self.impath = impath
        self.n_colors = n_colors
        
        
    def load_image(self):
        # Load the thin section image
        self.impath = '/Users/clay/Desktop/img_proc/thinsection.jpg'
        im = Image.open(self.impath)

        # Convert to floats instead of the default 8 bits integer coding. Dividing by
        # 255 is important so that plt.imshow behaves works well on float data (need to
        # be in the range [0-1])
        self.im = np.array(im, dtype=np.float64) / 255

        return self.im
    
    def unique_colors(self, a):
        a = np.ascontiguousarray(a)
        unique_a = np.unique(a.view([('', a.dtype)]*a.shape[2]))
        unique_b = unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[2]))
        return unique_b.size
    
    def cluster_img(self, subsample):
        # Load Image and transform to a 2D numpy array.
        self.w, self.h, d = original_shape = tuple(self.im.shape)
        assert d == 3
        image_array = np.reshape(self.im, (self.w * self.h, d))

        print("Fitting model on a small sub-sample of the data")
        t0 = time()
        image_array_sample = shuffle(image_array, random_state=0)[:subsample]
        self.kmeans = KMeans(n_clusters=self.n_colors, random_state=0).fit(image_array_sample)
        # kmeans = KMeans(random_state=0).fit(image_array_sample)
        print("done in %0.3fs." % (time() - t0))

        # Get labels for all points
        print("Predicting color indices on the full image (k-means)")
        t0 = time()
        self.labels = self.kmeans.predict(image_array)
        print("done in %0.3fs." % (time() - t0))
        
        return self.kmeans, self.labels, self.w, self.h
    
    def recreate_image(self, codebook, labels, w, h):
        
        """Recreate the (compressed) image from the code book & labels"""
        self.d = codebook.shape[1]
        self.image = np.zeros((self.w, self.h, self.d))
        label_idx = 0
        for i in range(w):
            for j in range(h):
                self.image[i][j] = codebook[labels[label_idx]]
                label_idx += 1
        return self.image


    def colorbar(self, kmeanLabels, newim):

        labelnum=np.unique(kmeanLabels).size

        results = np.empty([labelnum])

        for index in range(labelnum):
            results[index]=kmeanLabels[kmeanLabels==np.unique(kmeanLabels)[index]].size

        colorBar2 = np.empty((200,3))
        colorBar2_weight = np.empty(200)

        weight = results/kmeanLabels.size
        cumweight = np.cumsum(np.round(results/kmeanLabels.size * 200))
        colors = np.unique(newim.reshape((newim.shape[0] * newim.shape[1], 3)),axis=0)

        # NEED TO FIX THIS LATER
#         for aa in range(weight.size):
#         #     colorBar2[cumweight[aa-1]:cumweight[aa].astype(int)-1,:] = colors[aa]
#             if aa-1 < 0:
#                 colorBar2[0:cumweight[aa-1].astype(int),:] = colors[aa]
#                 colorBar2_weight[0:cumweight[aa-1].astype(int)] = weight[aa]
#             elif aa == weight.size-1:
#                 colorBar2[cumweight[aa-1].astype(int)-1:,:] = colors[aa]
#                 colorBar2_weight[cumweight[aa-1].astype(int)-1:] = weight[aa]
#             else:
#                 colorBar2[cumweight[aa-1].astype(int):cumweight[aa].astype(int),:] = colors[aa]
#                 colorBar2_weight[cumweight[aa-1].astype(int):cumweight[aa].astype(int)] = weight[aa]

#         colorBarhsv = np.empty((200,3))

#         # RGB values to HSV
#         for bb in range(200):
#             colorBarhsv[bb,:] = colorsys.rgb_to_hsv(colorBar2[bb,0],colorBar2[bb,1],colorBar2[bb,2])

#         # Sort colors by H
#         colorBar2 = colorBar2[colorBarhsv[:,0].argsort()]

#         # Sort weight values for colors
#         _, idx = np.unique(colorBar2_weight, return_index=True)
#         self.colorBar2_weight = colorBar2_weight[np.sort(idx)]

#         self.colorBar = np.repeat(colorBar2.reshape((1,200,3)),50, axis=0)

#         return self.colorBar, self.colorBar2_weight
        return colors, weight