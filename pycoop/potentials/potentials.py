import numpy as np
from sklearn import cluster, mixture


def extract_pix_from_marks(im, marks):
  """
  Extract pixels at marked locations.

  Take
    im: image in skimage ndarray format

  Give
    fg_pix, bg_pix: tuple of N x 3 ndarrays,
                    for foreground and background pixels
  """
  fg_color = (255, 0, 0)
  bg_color = (0, 0, 255)
  fg_pix = im[np.all(marks == fg_color, axis=-1)]
  bg_pix = im[np.all(marks == bg_color, axis=-1)]
  return fg_pix, bg_pix


def learn_gmm(X, n_comp=5, covar='full'):
  """
  Learn a Gaussian mixture model on features X,
  with 5 components and full covariance by default.

  Take
    X: data in N x K ndarray where N is no. samples and K is no. features
    n_comp: number of components in mixture model
    covar: covariance type ('full' or 'diag' most useful)

  Give
    gmm: a fitted sklearn GMM:
    http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GMM.html
  """
  gmm = mixture.GaussianMixture(n_components=5, covariance_type='full')
  gmm.fit(X)
  return gmm


def make_gmm_unaries(X, fg_gmm, bg_gmm):
  """
  Make unaries by log probability under GMM.

  Take
    X: data in N x K ndarray where N is no. samples and K is no. features
    fg_gmm, bg_gmm: fitted sklearn.mixture.gmm

  Give
    fg_unary, bg_unary: log probabilities under the fg and bg gmms resp.
  """
  fg_un = fg_gmm.score_samples(X)
  bg_un = bg_gmm.score_samples(X)
  return (fg_un, bg_un)


def cluster_edges(edges, k=None, centroids=None):
  """
  Make edge classes by k-means clustering. The kth class is for zero edges.

  Take
    edges: gradients in 2*m x f ndarray for m undirected edges and f features.
           n.b. edges must be in coop order (see image_graph.hpp)
    k: number of clusters
    ...OR...
    centroids: custom centroids for cluster assigment in k x f ndarray

  Give
    clusters: cluster assignments {0, ..., k}
    centroids: cluster centroids
  """
  # args check: need either k or centroids
  if not k and not centroids:
    raise ValueError('Need number of clusters k or centroids for clustering.')

  clusters = np.empty(edges.shape[0], dtype=np.int32)
  clusters.fill(k)  # zero edges are assigned to last class

  # only cluster actual gradients and ignore zero edges
  diff_ix = np.where((edges ** 2).sum(axis=1) > 0)[0]

  if k:
    # make cluster centroids
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(edges[diff_ix])
  else:
    # custom centroids
    kmeans = cluster.KMeans(init=centroids)
    kmeans.fit(centroids)  # need to make sklearn shut up about fitting

  diff_clusters = kmeans.predict(edges[diff_ix])
  clusters[diff_ix] = diff_clusters
  return clusters, kmeans.cluster_centers_
