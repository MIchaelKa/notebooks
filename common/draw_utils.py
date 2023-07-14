import numpy as np
import matplotlib.pyplot as plt

def prepare_image(image):
    if image.ndim == 3:
        cmap = None
        image = image[..., ::-1].astype(np.uint8)
    else:
        cmap = 'gray'
        image = image.astype(np.uint8)
    return image, cmap

#
# show_image / show_images
#

def show_image_hist(image, size=7):
    plt.figure(figsize=(size, int(size / 1.5)))
    plt.hist(image.reshape(-1), bins=100)
    plt.show()

def show_image(image, label=None, size=7):
    '''
    Convenient method, see show_images for documentation
    '''
    show_images([image], [label], size=7)

def show_images(images, labels=None, size=7):
    '''
    Similar on draw_image but:
        - less settings
        - support for different sizes
        - support for multiple images for side by side comparison
    '''
    images_len = len(images)
    plt.figure(figsize=((size/2)*images_len,size))
    
    for i, image in enumerate(images):
        image, cmap = prepare_image(image)
        plt.subplot(1, images_len, i+1)
        if labels is not None:
            plt.title(labels[i])
        plt.imshow(image, cmap=cmap)