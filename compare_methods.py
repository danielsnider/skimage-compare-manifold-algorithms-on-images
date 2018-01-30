
import sys
import glob
import scipy
import plot
import skimage.io
import numpy as np
import matplotlib.pyplot as plt

from time import time
from sklearn import manifold, datasets
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, datasets

# Next line to silence pyflakes. This import is needed.
Axes3D


# Input parameters
n_neighbors = 10
n_components = 2
image_dir = './Beautiful_Cityscapes_low_res/'
dpi=500
res=70
perplexity=int(sys.argv[1]) if len(sys.argv)>1 else 30

# Load images files
filenames=list(glob.glob(image_dir+'*.jpg'))
images = []
total_res = res**2*3 # 2 for x and y, 3 for rgb
X = np.zeros((len(filenames),total_res)) # Dimension of the image: 70*70=4900; X will store images in 2d array
count = 0
for imageName in filenames:
  image = skimage.io.imread(imageName) # load image
  images.append(image)
  image1d = scipy.misc.imresize(image, (res,res)) #reshape size to 70,70 for every image
  image1d = image1d.flatten() #image1d stores a 1d array for each image
  X[count,:] = image1d # add a row of values
  count+=1


# Begin plotting
fig = plt.figure(figsize=(15, 8))
plt.suptitle("Manifold Learning with %i images, %i neighbors"
             % (count, n_neighbors), fontsize=14)


methods = ['standard', 'ltsa', 'hessian', 'modified']
labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

for i, method in enumerate(methods):
    t0 = time()
    Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                        eigen_solver='auto',
                                        method=method).fit_transform(X)
    t1 = time()
    print("%s: %.2g sec" % (methods[i], t1 - t0))

    ax = fig.add_subplot(252 + i)
    plt.title("%s (%.2g sec)" % (labels[i], t1 - t0))
    canvas = plot.image_scatter(Y[:, 0], Y[:, 1], images, min_canvas_size=3000, bg_color=255, lw=10)
    plt.imshow(canvas)
    plt.xticks([])
    plt.yticks([])



t0 = time()
Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
t1 = time()
print("Isomap: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(257)
plt.title("Isomap (%.2g sec)" % (t1 - t0))
canvas = plot.image_scatter(Y[:, 0], Y[:, 1], images, min_canvas_size=3000, bg_color=255, lw=10)
plt.imshow(canvas)
plt.xticks([])
plt.yticks([])

t0 = time()
mds = manifold.MDS(n_components, max_iter=100, n_init=1)
Y = mds.fit_transform(X)
t1 = time()
print("MDS: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(258)
plt.title("MDS (%.2g sec)" % (t1 - t0))
canvas = plot.image_scatter(Y[:, 0], Y[:, 1], images, min_canvas_size=3000, bg_color=255, lw=10)
plt.imshow(canvas)
plt.xticks([])
plt.yticks([])


t0 = time()
se = manifold.SpectralEmbedding(n_components=n_components,
                                n_neighbors=n_neighbors)
Y = se.fit_transform(X)
t1 = time()
print("SpectralEmbedding: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(259)
plt.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
canvas = plot.image_scatter(Y[:, 0], Y[:, 1], images, min_canvas_size=3000, bg_color=255, lw=10)
plt.imshow(canvas)
plt.xticks([])
plt.yticks([])

t0 = time()
tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0, perplexity=perplexity)
Y = tsne.fit_transform(X)
t1 = time()
print("t-SNE: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(2, 5, 10)
plt.title("t-SNE (%.2g sec)" % (t1 - t0))
canvas = plot.image_scatter(Y[:, 0], Y[:, 1], images, min_canvas_size=3000, bg_color=255, lw=10)
plt.imshow(canvas)
plt.xticks([])
plt.yticks([])

# Save output to disk
save_location = 'output_plot.jpg'
plt.savefig(save_location,dpi=dpi,pad_inches=1,bbox_inches='tight')
print('Saved image scatter to %s' % save_location)

# plt.show()