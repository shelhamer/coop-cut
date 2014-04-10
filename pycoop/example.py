import pycoop, numpy as np, skimage.io

example_dir = '../experiments/example-input/'

im = skimage.io.imread(example_dir + '005_bee.jpg').astype(np.float64)
ig = pycoop.InputGraph(im)
source = np.load(example_dir + 'source_unaries.npy')
sink = np.load(example_dir + 'sink_unaries.npy')
classes = np.load(example_dir + 'classes.npy').astype(np.int32)
ig.setClasses(classes)
ig.setUnaries(source, sink)
label_im, cost, cut = pycoop.segment(ig, 2.5, 0.01, 12)
segment_im = np.ones_like(im)
segment_im[label_im] = im[label_im]*255

print 'cut cost {:.2f} with {} edges'.format(cost, len(cut))
skimage.io.imsave('segment.png', segment_im)
