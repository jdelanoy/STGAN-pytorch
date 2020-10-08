import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from skimage import transform
import numpy as np
from torchvision import transforms


def _imscatter(x, y, image, color=None, ax=None, zoom=1.):
    """ Auxiliary function to plot an image in the location [x, y]
        image should be an np.array in the form H*W*3 for RGB
    """
    if ax is None:
        ax = plt.gca()
    try:
        image=image.numpy().transpose((1,2,0))*0.5+0.5
        #image = plt.imread(image)
        size = min(image.shape[0], image.shape[1])
        image = transform.resize(image[:size, :size], (256, 256))
    except TypeError:
        # Likely already an array...
        pass
    #print(x)
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        edgecolor = dict(boxstyle='round,pad=0.05',
                         edgecolor=color, lw=1) \
            if color is not None else None
        ab = AnnotationBbox(im, (x0, y0),
                            xycoords='data',
                            frameon=True,
                            bboxprops=edgecolor,
                            )
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists
