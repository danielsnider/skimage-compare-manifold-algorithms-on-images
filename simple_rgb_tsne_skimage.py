from time import time
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
import numpy as np
import skimage.io
import glob
import scipy
import plot
import sys



image_dir = './Beautiful_Cityscapes_low_res/'
dpi=500
perplexity=int(sys.argv[1]) if len(sys.argv)>1 else 4


filenames=list(glob.glob(image_dir+'*.jpg'))


images = []
res=100
total_res = res**2*3 # 2 for x and y, 3 for rgb
x_value = np.zeros((len(filenames),total_res)) # Dimension of the image: 70*70=4900; x_value will store images in 2d array
print filenames
count = 0
for imageName in filenames:
  image = skimage.io.imread(imageName) # load image
  images.append(image)
  image1d = scipy.misc.imresize(image, (res,res)) #reshape size to 70,70 for every image
  image1d = image1d.flatten() #image1d stores a 1d array for each image
  x_value[count,:] = image1d # add a row of values
  count+=1



n_components = 2
tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0, perplexity=perplexity)
Y = tsne.fit_transform(x_value)

canvas = plot.image_scatter(Y[:, 0], Y[:, 1], images, min_canvas_size=3000, bg_color=255, lw=10)
plt.imshow(canvas)
plt.xticks([])
plt.yticks([])
save_location = 'output_plot_%sdpi_perplexity%s.jpg' % (dpi,perplexity)
plt.savefig(save_location,dpi=dpi,pad_inches=1,bbox_inches='tight')
print('Saved image scatter to %s' % save_location)



