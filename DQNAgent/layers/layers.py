import numpy as np
from time import time

try:
    from cython_im2col import col2im, im2col, col2im, im2col
except ImportError:
    print 'run the following from the cs231n directory and try again:'
    print 'python setup.py build_ext --inplace'
    print 'You may also need to restart your iPython kernel'


def affine_forward(x, w, b):

    x_flat = x.reshape(x.shape[0], -1)
    out = np.dot(x_flat, w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):

    x, w, b = cache
    dx_flat = np.dot(dout, w.T)
    dx = dx_flat.reshape(x.shape)
    x_flat = x.reshape(x.shape[0], -1)
    dw = np.dot(x_flat.T, dout)
    db = np.sum(dout, axis=0)

    return dx, dw, db


def relu_forward(x):
    """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):

	x = cache
	dx = np.where(x > 0, dout, 0)
	return dx


def batchnorm_forward(x, gamma, beta, bn_param):

  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    mean = np.mean(x,axis = 0)
    var = np.sum(((x - mean) ** 2), axis = 0) / N
    x_hat = (x - mean) / np.sqrt(var + eps)
    running_mean = momentum * running_mean + (1 - momentum) * mean
    running_var = momentum * running_var + (1 - momentum) * var
    out = x_hat * gamma + beta
    cache = (x,x_hat,mean,var,gamma,eps)
  elif mode == 'test':
    mean = running_mean
    var = running_var
    x_hat = (x - mean) / np.sqrt(var + eps)
    out = x_hat * gamma + beta
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):

  x,x_hat,mean,var,gamma,eps = cache

  N = x.shape[0]
  dgamma = np.sum(dout * x_hat, axis = 0)
  dbeta = np.sum(dout, axis = 0)

  dx_hat = dout * gamma
  partial_x = dx_hat / np.sqrt(var + eps)
  partial_var = -0.5 * (1 / np.sqrt(var + eps) ** 3) * np.sum((x - mean) * dx_hat, axis = 0)
  partial_mean = np.sum(-dx_hat / np.sqrt(var + eps), axis = 0) + partial_var / N * np.sum(-2 * (x - mean), axis = 0)

  dx = partial_x + partial_var * 2 * (x - mean) / N + partial_mean / N

  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        mask = np.random.binomial(1, 1 - p, x.shape)
        out = x * mask

    elif mode == 'test':
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
    dropout_param, mask = cache
    mode = dropout_param['mode']
    if mode == 'train':
        dx = dout * mask
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward(x, w, b, conv_param):
    """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    # Check dimensions
    assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
    assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

    # Create output
    out_height = (H + 2 * pad - filter_height) / stride + 1
    out_width = (W + 2 * pad - filter_width) / stride + 1

    x_cols = im2col(x, w.shape[2], w.shape[3], pad, stride)  # (10000 * 1024,27)

    out = x_cols.dot(w.reshape((w.shape[0], -1)).T) + b  # (10000 * 1024,4)
    out = out.reshape(N, out_height, out_width, -1)
    out = out.transpose(0, 3, 1, 2)

    cache = (x, w, b, conv_param, x_cols)
    return out, cache


def conv_backward(dout, cache):
    """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
    x, w, b, conv_param, x_cols = cache
    N, C, H, W = x.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    db = np.sum(dout, axis=(0, 2, 3))

    num_filters, c, filter_height, filter_width = w.shape
    dout_reshape = dout.reshape(N, num_filters, -1).transpose(0, 2, 1)  # (10000,1024,4)
    dout_reshape = dout_reshape.reshape(-1, num_filters)  # (10000 * 1024,4)
    dw = (dout_reshape.T).dot(x_cols).reshape(w.shape)

    dx_cols = dout_reshape.dot(w.reshape(num_filters, -1))
    dx = col2im(dx_cols, x.shape[0], x.shape[1], x.shape[2], x.shape[3],
                filter_height, filter_width, pad, stride)
    return dx, dw, db


def max_pool_forward(x, pool_param):
    """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']

    same_size = pool_height == pool_width == stride
    tiles = H % pool_height == 0 and W % pool_width == 0
    if same_size and tiles:
        out, reshape_cache = max_pool_forward_reshape(x, pool_param)
        cache = ('reshape', reshape_cache)
    else:
        out, im22col_cache = max_pool_forward_im2col(x, pool_param)
        cache = ('im2col', im22col_cache)
    return out, cache


def max_pool_forward_reshape(x, pool_param):
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    assert pool_height == pool_width == stride, 'Invalid pool params'
    assert H % pool_height == 0
    assert W % pool_height == 0
    x_reshaped = x.reshape(N, C, H / pool_height, pool_height,
                           W / pool_width, pool_width)
    out = x_reshaped.max(axis=5).max(axis=3)

    cache = (x, x_reshaped, out)
    return out, cache


def max_pool_forward_im2col(x, pool_param):
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']

    assert (H - pool_height) % stride == 0, 'Invalid height'
    assert (W - pool_width) % stride == 0, 'Invalid width'

    out_height = (H - pool_height) / stride + 1
    out_width = (W - pool_width) / stride + 1

    x_split = x.reshape(N * C, 1, H, W)
    x_cols = im2col(x_split, pool_height, pool_width, 0, stride)  # (10000 * 1024 * 3, 9)
    x_cols_argmax = np.argmax(x_cols, axis=1)
    x_cols_max = x_cols[np.arange(x_cols.shape[0]), x_cols_argmax]  # (10000 * 3 * 1024)
    out = x_cols_max.reshape(N, C, out_height, out_width)

    cache = (x, x_cols, x_cols_argmax, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
    method, real_cache = cache
    if method == 'reshape':
        return max_pool_backward_reshape(dout, real_cache)
    elif method == 'im2col':
        return max_pool_backward_im2col(dout, real_cache)
    else:
        raise ValueError('Unrecognized method "%s"' % method)


def max_pool_backward_reshape(dout, cache):
    x, x_reshaped, out = cache
    x_reshaped = x_reshaped.transpose(0, 1, 2, 4, 3, 5)  # (100,3,16,16,2,2)
    out_newaxis = out[:, :, :, :, np.newaxis, np.newaxis]  # (100,3,16,16,1,1)
    mask = (x_reshaped == out_newaxis)  # (100,3,16,16,2,2)
    dx_reshaped = mask * dout[:, :, :, :, np.newaxis, np.newaxis]
    dx = dx_reshaped.transpose(0, 1, 2, 4, 3, 5).reshape(x.shape)
    return dx


def max_pool_backward_im2col(dout, cache):
    x, x_cols, x_cols_argmax, pool_param = cache
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']

    dx_cols = np.zeros(x_cols.shape)
    dx_cols[np.arange(x_cols.shape[0]), x_cols_argmax] = dout.flatten()
    dx = col2im(dx_cols, N, C, H, W, pool_height, pool_width, 0, stride)
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
  Computes the forward pass for spatial batch normalization.

  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
    out, cache = None, None
    N, C, H, W = x.shape
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    running_mean = bn_param.get('running_mean', np.zeros((H, W), dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros((H, W), dtype=x.dtype))

    if mode == 'train':
        mean = np.mean(x, axis=0)
        var = np.mean((x - mean) ** 2, axis=0)
        x_hat = (x - mean) / np.sqrt(var + eps)
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var
        out = x_hat * gamma.reshape(C, 1, 1) + beta.reshape(C, 1, 1)
        cache = (x, x_hat, mean, var, gamma, eps)
        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var
    elif mode == 'test':
        mean = running_mean
        var = running_var
        x_hat = (x - mean) / np.sqrt(var + eps)
        out = x_hat * gamma.reshape(C, 1, 1) + beta.reshape(C, 1, 1)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    return out, cache


def spatial_batchnorm_backward(dout, cache):

    x, x_hat, mean, var, gamma, eps = cache
    dgamma = np.sum(dout * x_hat, axis=(0, 2, 3))
    dbeta = np.sum(dout, axis=(0, 2, 3))
    N, C, H, W = x.shape

    dx_hat = dout * gamma.reshape(C, 1, 1)
    partial_x = dx_hat / np.sqrt(var + eps)
    partial_var = -0.5 * (1 / np.sqrt(var + eps) ** 3) * np.sum((x - mean) * dx_hat, axis=0)
    partial_mean = np.sum(-dx_hat / np.sqrt(var + eps), axis=0) + partial_var / N * np.sum(-2 * (x - mean), axis=0)

    dx = partial_x + partial_var * 2 * (x - mean) / N + partial_mean / N
    return dx, dgamma, dbeta


def conv_forward_im2col(x, w, b, conv_param):
    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    # Check dimensions
    assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
    assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

    # Create output
    out_height = (H + 2 * pad - filter_height) / stride + 1
    out_width = (W + 2 * pad - filter_width) / stride + 1

    x_cols = im2col(x, w.shape[2], w.shape[3], pad, stride)
    out = x_cols.dot(w.reshape((w.shape[0],)).T) + b
    out = out.transpose(0, 2, 1).rehape(N, num_filters, out_height, out_width)

    cache = (x, w, b, conv_param, x_cols)
    return out, cache


def conv_backward_im2col(dout, cache):
    x, w, b, conv_param, x_cols = cache
    stride, pad = conv_param['stride'], conv_param['pad']

    db = np.sum(dout, axis=(0, 2, 3))
    N = x.shape[0]

    num_filters, c, filter_height, filter_width = w.shape
    dout_reshape = dout.reshape(N, num_filters, -1).transpose(0, 2, 1)  # (10000,1024,4)
    x_col_reshape = x_cols.reshape(-1, c * filter_width * filter_height)  # (10000*1024,27
    dw = (dout_reshape.reshape(-1, num_filters).T).dot(x_col_reshape).reshape(w.shape)

    dx_cols = dout_reshape.dot(w.reshape(num_filters, -1))
    dx = col2im(dx_cols, x.shape[0], x.shape[1], x.shape[2], x.shape[3],
                filter_height, filter_width, pad, stride)

    return dx, dw, db





def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward(da, conv_cache)
  return dx, dw, db



def svm_loss(x, y):

    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):

    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

def mean_square(x, y, a):
	N = x.shape[0]
	q_action = x * a
	loss = np.sum((q_action - y) ** 2) / N
	dx = 2.0 / N * (q_action - y)
	return loss, dx



