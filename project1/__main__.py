from test_case import run_test_case
import tensorflow as tf

layers = [1, 3, 5]
kernel_sizes = [3, 5]
padding = [0, 1, 2]  # TODO
stride = [1, 2, 3, 5]  # TODO
pooling = [None, 'max_pooling', 'average_pooling']
normalization = [None, 'batch', 'layer']  # TODO
optimizeers = [None, 'adam', 'SGD', 'rmsprop', 'adagrad']

# PNG warning: iCCP: known incorrect sRGB profile
tf.autograph.set_verbosity(
    level=0, alsologtostdout=False
)


run_test_case(5, 3, 0, 1, 'average_pooling', 'none', 'rmsprop')
