import numpy as np
import torch
import sys
import time
import torch.nn.functional as F

def soft_max(x, dim=1):
    '''
    soft_max
    :param x: shape = (n, c)
    :param dim:
    :return:
    '''
    return np.exp(x) / np.sum(np.exp(x), axis=dim, keepdims=True)

# if __name__ == '__main__':
#     a = np.random.random((3, 5))
#     print(soft_max(a))
#     print(torch.softmax(torch.from_numpy(a), dim=1))

def sigmoid(x):
    '''
    sigmoid
    :param x: shape = (n, c)
    :return:
    '''
    return 1 / (1 + np.exp(-x))

# if __name__ == '__main__':
#     a = np.random.random((3, 5))
#     print(sigmoid(a))
#     print(torch.sigmoid(torch.from_numpy(a)))

def _bn_1d(x, weight, bias, eps=1e-05):
    '''
    :param x: shape = (n, c)
    :param weight: shape = (c, )
    :param bias: shape = (c, )
    :param eps:
    :return:
    '''
    m_ = np.mean(x, axis=0, keepdims=True)
    v_ = np.mean((x - m_)**2, axis=1, keepdims=True)
    x_ = (x - m_) / np.sqrt(v_ + eps)

    return weight * x + bias

def _bn_2d(x, weight, bias, eps=1e-05):
    '''
    :param x: shape = (n, c, h, w)
    :param weight: shape = (c, )
    :param bias: shape = (c, )
    :param eps:
    :return: (n, c, h, w)
    '''
    m_ = np.mean(x, axis=(0, 2, 3), keepdims=True)
    v_ = np.mean((x - m_)**2, axis=(0, 2, 3), keepdims=True)
    x_ = (x - m_) / np.sqrt(v_ + eps)
    weight = weight[np.newaxis, :, np.newaxis, np.newaxis]
    bias = bias[np.newaxis, :, np.newaxis, np.newaxis]

    return x_ * weight + bias

def linear(x, weight, bias):
    '''
    :param x: shape = (n, c_in)
    :param weight: shape = (c_out, c_in)
    :param bias: shape = (c_out, )
    :return: (n, c_out)
    '''
    return np.add(np.matmul(x, weight.T), bias)

def one_hot(y, n_class):
    '''
    :param y: shape = (n, ), n = batch_size
    :param n_class: total catergories
    :return: (n, n_class)
    '''

    n = y.shape[0]
    res = np.zeros((n, n_class))
    res[range(n), y] = 1
    return res
#
# if __name__ == '__main__':
#     a = np.random.randint(0, 10, (5))
#     print(one_hot(a, 10))
#     print(F.one_hot(torch.from_numpy(a).to(torch.int64), 10))

def cross_entropy(true_label, pred_label, weight=None):
    '''
    :param true_label: shape = (n, n_class)
    :param pre_label: shape = (n, n_class)
    :param weight: weight for unbalanced training set, shape = (n_class, )
    :return:
    '''
    res = None
    if true_label.ndim == 1:
        true_label = one_hot(true_label, 10)
    if np.sum(pred_label) != pred_label.shape[0]:
        # if unnormalized scores, the normalize
        pred_label = soft_max(pred_label)
    y_pred = np.log(pred_label)

    if weight is not None:
        res = -np.sum(true_label * y_pred * weight)
    else:
        res = -np.sum(true_label * y_pred)

    return res / pred_label.shape[0]

# if __name__ == '__main__':
#     a = np.random.random((4, 10))
#     b = np.random.randint(0, 10, 4)
#     b_ = one_hot(b, 10)
#     print(cross_entropy(b_, a))
#     print(F.cross_entropy(torch.from_numpy(a), torch.from_numpy(b).to(torch.int64)))

def mse_loss(input, target):
    '''
    :param input: shape = (*)
    :param target: shape = (*)
    :return:
    '''
    return np.sum((input - target)**2) / input.shape[0]

def conv2d(x, weight, bias, stride, padding, dilation, groups=1):
    '''
    :param x: shape = (n, c_in, h_in, w_in)
    :param weight: filter, shape = (c_out, c_in // groups, kh, kw)
    :param bias: shape = (c_out, ), default = None
    :param stride: (sh, sw)
    :param padding: (ph, pw)
    :param dilation: default = 1
    :param groups: default = 1
        at groups = 1, all inputs are convolved to all outputs
        at groups = 2, equal to having two conv layers side by side,
        each seeing half the input channels and producing half output channels,
        and both subsequently concated
        at groups = c_in, each input channel is convolved with its own set of filters(
        of size c_out/c_in), in other words, each input channel is convolved
        c_out/c_int times, e.g. c_out = 10, c_in = 5, then 0, 5 <----> 0, 1, 6 <----> 1
    :return: (N, c_out, h_out, w_out)

    '''
    h_in, w_in = x.shpae[2:]
    n, c_in = x.shape[:2]
    c_out = weight.shape[0]
    kh, kw = weight.shape[2:]
    sh, sw = stride
    ph, pw = padding
    pass


# a = np.random.randint(0, 3, (4, 2, 2))
# b = np.random.randint(0, 3, (2, 2, 2))
# print(a)
# print(b)
# a = a.reshape((4, -1))
# b = b.reshape((2, -1))
#
# print(np.sum(np.matmul(a, b.T), axis=0))

def pad(input, pad, mode='constant', value=None):
    '''
    pad the last dimension and move forward
    len(pad)//2 dimensions will be padded
    pad order (left, right, top, bottom)
    only support for 1 or 2 dimensions padding
    :param input: (n, c_in, h_in, h_out)
    :param pad: tuple (2, ) (4, )
    :param mode:
    :param value:
    :return: (n, c, h_out, w_out)
    '''
    n, c, h_in, w_in = input.shape
    value = value if value is not None else 0.
    p = len(pad)
    res = None
    assert p in (2, 4), 'p should be in (2, 4)'
    if p == 2:
        l, r = pad
        res = np.zeros((n, c, h_in, w_in + l + r)) + value
        res[:, :, :, l:l + w_in] = input

    if p == 4:
        l, r, t, b = pad
        res = np.zeros((n, c, h_in + t + b, w_in + l + r)) + value
        res[:, :, t:h_in + t, l:w_in + l] = input

    return res

# inner-product use np.dot or np.matmul of @
# element-wise product use np.multiply or *



def conv2d(input, weight, bias, stride=1, padding=0, dialition=1, groups=1):
    '''
    :param x: shape = (n, c_in, h_in, w_in)
    :param weight: filter, shape = (c_out, c_in // groups, kh, kw)
    :param bias: shape = (c_out, ), default = None
    :param stride: (sh, sw) or int
    :param padding: int or (ph, pw)
    :param dilation: default = 1
    :param groups: default = 1, c_in and c_out must both be divisible by groups
            at groups = 1, all inputs are convolved to all outputs
            at groups = 2, equal to having two conv layers side by side,
            each seeing half the input channels and producing half output channels,
            and both subsequently concatenated
            at groups = c_in, each input channel is convolved with its own set of filters(
            of size c_out/c_in), in other words, each input channel is convolved
            c_out/c_int times, e.g. c_out = 10, c_in = 5, then 0, 5 <----> 0, 1, 6 <----> 1
            for example, if group = 2, input with (n, 32, 100, 100), filter with (48, 3, 3),
            weight has shape (48, 16, 3, 3), c_in = 32 should be divided into 2 groups with
            channel = 16, c_out = 48 also should be divided into 2 groups with channel = 24,
            the first 24 output channels, each output channel comes from the first input
            channel group and conduct a full convolution.
    :return: (N, c_out, h_out, w_out)
    '''
    a = input
    b = weight
    p_h, p_w = padding
    a = pad(a, (p_w, p_w, p_h, p_h))
    c_in_out, h_k, w_k = b.shape[1:]
    c_in = a.shape[1]
    n = a.shape[0]
    c_out = b.shape[0]
    h_in, w_in = a.shape[2:]
    assert c_in_out * groups == c_in and c_out % groups == 0, \
        'in_channels and out_channels must be divisible by groups'

    h_out, w_out = ((h_in - h_k) // stride + 1, (w_in - w_k) // stride + 1)

    out = np.zeros((n, c_out, h_out, w_out))

    # for i in range(n):
    #     for j in range(c_out):
    #         for k in range(h_out):
    #             for l in range(w_out):
    #                 out[i, j, k, l] = np.sum(
    #                     a[i, :, k * stride: k * stride + h_k, l * stride: l * stride + w_k] * b[j,]
    #                 )
    g_idx = 0
    start, end = None, None
    for j in range(c_out):
        if j % (c_out / groups) == 0:
            start = g_idx * c_in_out
            end = (g_idx + 1) * c_in_out
            g_idx += 1
        for k in range(h_out):
            for l in range(w_out):
                out[:, j, k, l] = np.sum(
                    a[:, start:end, k * stride: k * stride + h_k, l * stride: l * stride + w_k] * b[j,],
                    axis=(1, 2, 3)
                )
    if bias is not None:
        bias = bias[np.newaxis, :, np.newaxis, np.newaxis]
        out = np.add(out, bias)
    return out

# if __name__ == '__main__':
#     '''
#     numpy consumes more time the torch implementing
#     '''
#     c_out = 50
#     a = np.random.randint(0, 3, (4, 10, 12, 12))
#     b = np.random.randint(0, 2, (c_out, 2, 3, 3))
#     stride = 2
#     padding = (1, 2)
#     c = np.random.randint(0, 100, (c_out))
#     s = time.time()
#     out = conv2d(a, b, bias=c, stride=stride, padding=padding, groups=5)
#     print('time consume ', time.time() - s)
#     a = torch.from_numpy(a)
#     b = torch.from_numpy(b)
#     c = torch.from_numpy(c)
#
#     # padding = (padH, padW)
#     s = time.time()
#     f = torch.conv2d(a, b, bias=c, stride=stride, padding=padding, groups=5)
#     print('torch time consume ', time.time() - s)
#     print(np.array(f) == out)

