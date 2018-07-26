from pycoop import potentials
from skimage import io
import numpy as np
from sklearn.mixture import GaussianMixture

im = io.imread('data/bee.jpg').astype(np.float64)
mark_im = io.imread('data/marks.png')[:,:,:-1]

X = im.reshape((-1, 3))

def test_learn_gmm():
    gmm = potentials.learn_gmm(X)
    assert(isinstance(gmm, GaussianMixture))


def test_gmm_unaries():
    fg_pix, bg_pix = potentials.extract_pix_from_marks(im, mark_im)

    fg_gmm = potentials.learn_gmm(fg_pix)
    bg_gmm = potentials.learn_gmm(bg_pix)

    fg_un, bg_un = potentials.make_gmm_unaries(im.reshape((-1, 3)), fg_gmm, bg_gmm)

    assert(len(fg_un)==len(bg_un))