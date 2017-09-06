"""Utility to apply random affine transformations and gaussian noise
to an input image. Useful for data augmentation"""

# Please note - Some portions of the code were obtained from the following resource
# https://florianmuellerklein.github.io/cnn_streetview/
from skimage import transform, filters, exposure
import numpy as np

PIXELS = 32

# much faster than the standard skimage.transform.warp method
def fast_warp(img, tf, output_shape, mode='nearest'):
    return transform._warps_cy._warp_fast(img, tf.params,
                                          output_shape=output_shape, mode=mode)

def noisy(image):
    row,col,ch= image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy

def transform_image(data):
    # set empty copy to hold augmented images so that we don't overwrite
    X_out = np.empty(shape = (1, PIXELS, PIXELS, 3), dtype = 'float32')

    # random rotations betweein -10 and 10 degrees
    dorotate = np.random.randint(-10,10)

    # random translations
    trans_1 = np.random.randint(-6,6)
    trans_2 = np.random.randint(-6,6)

    # random zooms
    zoom = np.random.uniform(0.9, 1.1)

    # shearing
    shear_deg = np.random.uniform(-4, 4)

    #add noise
    no_add_noise = np.random.randint(0,3)
    if not no_add_noise:
        data = noisy(data)

    # set the transform parameters for skimage.transform.warp
    # have to shift to center and then shift back after transformation otherwise
    # rotations will make image go out of frame
    center_shift   = np.array((PIXELS, PIXELS)) / 2. - 0.5
    tform_center   = transform.SimilarityTransform(translation=-center_shift)
    tform_uncenter = transform.SimilarityTransform(translation=center_shift)

    tform_aug = transform.AffineTransform(rotation = np.deg2rad(dorotate),
                                          scale =(1/zoom, 1/zoom),
                                          shear = np.deg2rad(shear_deg),
                                          translation = (trans_1, trans_2))

    tform = tform_center + tform_aug + tform_uncenter

    for ch in range(3):
        X_out[:,:,:,ch] = fast_warp(data[:,:,ch], tform, output_shape = (PIXELS, PIXELS), mode='symmetric')

    return X_out