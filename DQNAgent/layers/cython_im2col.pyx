import numpy as np
cimport numpy as np
cimport cython

# DTYPE = np.float64
# ctypedef np.float64_t DTYPE_t

ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t

def im2col(np.ndarray[DTYPE_t, ndim=4] x, int field_height,
                  int field_width, int padding, int stride):
    cdef int N = x.shape[0]
    cdef int C = x.shape[1]
    cdef int H = x.shape[2]
    cdef int W = x.shape[3]

    cdef int HH = (H + 2 * padding - field_height) / stride + 1
    cdef int WW = (W + 2 * padding - field_width) / stride + 1

    cdef int p = padding
    cdef np.ndarray[DTYPE_t, ndim=4] x_padded = np.pad(x,
            ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    cdef np.ndarray[DTYPE_t, ndim=2] cols = np.zeros(
            (N * HH * WW,C * field_width * field_height),dtype=x.dtype)

    im2col_inner(cols,x_padded,N,C,HH,WW,field_height,field_width,padding,stride)
    return cols

@cython.boundscheck(False)
cdef int im2col_inner(np.ndarray[DTYPE_t, ndim=2] cols,
                              np.ndarray[DTYPE_t, ndim=4] x_padded,
							  int N, int C, int HH, int WW,
                              int field_height, int field_width, int padding, int stride) except? -1:
    cdef int i, c, hh, ww, fh, fw, r, k
    for i in range(N):
        for hh in range(HH):
            for ww in range(WW):
                r = i * HH * WW + hh * WW + ww
                for c in range(C):
                    for fh in range(field_height):
                        for fw in range(field_width):
                            k = c * field_height * field_height + fh * field_width + fw
                            cols[r,k] = x_padded[i,c,hh * stride + fh,ww * stride + fw ]


def col2im(np.ndarray[DTYPE_t, ndim=2] cols, int N, int C, int H, int W,
                  int field_height, int field_width, int padding, int stride):
    cdef np.ndarray x = np.empty((N, C, H, W), dtype=cols.dtype)
    cdef int HH = (H + 2 * padding - field_height) / stride + 1
    cdef int WW = (W + 2 * padding - field_width) / stride + 1
    cdef np.ndarray[DTYPE_t, ndim=4] x_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding),
                                        dtype=cols.dtype)

    # Moving the inner loop to a C-function with no bounds checking improves
    # performance quite a bit for col2im.
    col2im_inner(cols, x_padded, N, C, H, W, HH, WW,
                        field_height, field_width, padding, stride)
    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded

@cython.boundscheck(False)
cdef int col2im_inner(np.ndarray[DTYPE_t, ndim=2] cols,
                             np.ndarray[DTYPE_t, ndim=4] x_padded,
                             int N, int C, int H, int W, int HH, int WW,
                             int field_height, int field_width, int padding, int stride) except? -1:
    cdef int i, c, hh, ww, fh, fw, k

    for i in range(N):
        for hh in range(HH):
            for ww in range(WW):
                k = i * HH * WW + hh * WW + ww
                for c in range(C):
                    for fh in range(field_height):
                        for fw in range(field_width):
                            x_padded[i, c, stride * hh + fh, stride * ww + fw] += cols[k, c * field_width * field_height + fh * field_width + fw]

