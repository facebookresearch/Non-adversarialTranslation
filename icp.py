# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import faiss
from fbpca import pca
import collections
from numba.decorators import jit


@jit(nopython=True, parallel=True)
def outer_numba(a, b):
    bsz = a.shape[0]
    m = a.shape[1]
    n = b.shape[1]

    result = np.zeros((m, n), dtype=np.float32)
    for k in range(bsz):
        for i in range(m):
            for j in range(n):
                result[i, j] += a[k, i]*b[k, j]
    result = result / bsz

    return result


def mse_fn(x, y, M):
    dim, n = x.shape
    r = y - M.dot(x)
    er = 0.5 * r ** 2
    er = er.sum() / n

    if r.T.shape[1] > 25:
        grad = outer_numba(-r.T, x.T)
    else:
        grad = -r.T[:, :, None] * x.T[:, None, :]
        grad = grad.mean(0)

    return er, grad


def cyc_fn(x, MX, MY):
    dim, n = x.shape
    y_est = MX.dot(x)
    x_cyc = MY.dot(y_est)
    r = x_cyc - x
    er = 0.5 * r ** 2
    er = er.sum() / n

    # grad_MX = np.zeros((dim, dim))
    # grad_MY = np.zeros((dim, dim))
    MYMX = MY.dot(MX)
    eye = np.eye(MX.shape[1], dtype=np.float32)
    if r.T.shape[1] > 25:
        xxt = outer_numba(x.T, x.T)
    else:
        xxt = x.T[:, :, None] * x.T[:, None, :]
        xxt = xxt.mean(0)

    grad_MX = MY.T.dot(MYMX - eye).dot(xxt)
    grad_MY = (MYMX - eye).dot(xxt).dot(MX.T)
    return er, grad_MX, grad_MY


class SGD():
    def __init__(self, lr, alpha=0.0, wd=0.0):
        self.alpha = alpha
        self.init = False
        self.alpha = alpha
        self.lr = lr
        self.wd = wd
        self.v = None

    def step(self, g):
        if not self.init:
            self.v = np.zeros(g.shape)
            self.init = True
        g += self.wd
        self.v = self.v * self.alpha + self.lr * g
        return self.v

class RMSProp():
    def __init__(self, lr, alpha=0.99, wd=0):
        self.alpha = alpha
        self.init = False
        self.ma = None
        self.eps = 1e-8
        self.lr = lr
        self.wd = wd

    def step(self, g):
        if not self.init:
            self.g2_ma = np.zeros(g.shape)
            self.init = True
        g += self.wd
        self.g2_ma = self.g2_ma * self.alpha + g ** 2 * (1 - self.alpha)
        dg = self.lr *  g / (np.sqrt(self.g2_ma) + self.eps)
        return dg


class ADAM():
    def __init__(self, lr, beta1=0.5, beta2=0.999, wd=0):
        self.init = False
        self.ma_g = None
        self.ma_g2 = None
        self.eps = 1e-8
        self.lr = lr
        self.wd = wd
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0

    def step(self, g):
        self.t += 1
        if not self.init:
            self.g2_ma = np.zeros(g.shape, dtype=np.float32)
            self.g_ma = np.zeros(g.shape, dtype=np.float32)
            self.init = True
        g += self.wd
        self.g_ma = self.g_ma * self.beta1 + g * (1 - self.beta1)
        self.g2_ma = self.g2_ma * self.beta2 + g ** 2 * (1 - self.beta2)
        m = self.g_ma / (1 - self.beta1 ** self.t)
        v = self.g2_ma / (1 - self.beta2 ** self.t)
        dg = self.lr *  m / (np.sqrt(v) + self.eps)
        return dg


OptParams = collections.namedtuple('OptParams', 'lr batch_size epochs ' +
                                                'decay_epochs decay_rate ' +
                                                'cyc_k')
OptParams.__new__.__defaults__ = (None, None, None,
                                  None, None, None)



class _ICP():
    def __init__(self, dim, verbose=False):
        self.verbose = verbose
        self.TX = np.eye(dim, dtype=np.float32)
        self.TY = np.eye(dim, dtype=np.float32)
        self.dim = dim
        self.rec_hist = []
        self.bb_hist = []
        self.rpa = None
        self.rpb = None
        self.fixed_rp = False

    def train(self, x_all, y_all, opt_params, is_init, indices_x,
              indices_y, do_reciprocal):
        self.opt_params = opt_params
        for epoch in range(opt_params.epochs):
            if self.verbose:
                print("_" * 98)
            indices_x, indices_y, rec, bb = self.train_epoch(x_all, y_all,
                                                             is_init, indices_x,
                                                             indices_y, epoch,
                                                             do_reciprocal)
            self.rec_hist.append(rec)
            self.bb_hist.append(bb)
        return indices_x, indices_y, rec, bb

    def train_epoch(self, x_all, y_all, is_init, indices_x,
                    indices_y, epoch, do_reciprocal=False):
        # Compute batch size
        batch_size = self.opt_params.batch_size
        d, n = x_all.shape
        rpa = np.random.permutation(n)
        rpb = np.random.permutation(n)

        x_est = self.TY.dot(y_all).astype(np.float32)
        y_est = self.TX.dot(x_all).astype(np.float32)
        if not is_init or epoch > 0:
            nbrs_x = faiss.IndexFlatL2(d)
            nbrs_x.add(np.ascontiguousarray(x_est.T))
            _, indices_x = nbrs_x.search(x_all.T, k=1)
            indices_x = indices_x.squeeze()

            nbrs_y = faiss.IndexFlatL2(d)
            nbrs_y.add(np.ascontiguousarray(y_est.T))
            _, indices_y = nbrs_y.search(y_all.T, k=1)
            indices_y = indices_y.squeeze()

        # TODO: score???
        rec_x = ((x_all - x_est[:, indices_x]) ** 2).sum(0).mean()
        rec_y = ((y_all - y_est[:, indices_y]) ** 2).sum(0).mean()

        bb_idx_y = np.where(indices_x[indices_y] == np.arange(n))[0]
        bb_idx_x = np.where(indices_y[indices_x] == np.arange(n))[0]
        if self.verbose:
            print(len(bb_idx_x), len(bb_idx_y))

        if do_reciprocal:
            rpa = np.random.permutation(len(bb_idx_x))
            rpb = np.random.permutation(len(bb_idx_y))
            batch_n = min(len(bb_idx_x), len(bb_idx_y)) // batch_size
        else:
            rpa = np.random.permutation(n)
            rpb = np.random.permutation(n)
            batch_n = n // batch_size

        x_t_all = x_all[:, indices_y]
        y_t_all = y_all[:, indices_x]

        # Compute learning rate
        decay_steps = epoch // self.opt_params.decay_epochs
        lr = self.opt_params.lr * self.opt_params.decay_rate ** decay_steps
        cyc_k = self.opt_params.cyc_k
        weight_decay = 1e-5

       # optX = SGD(lr, 0.0, weight_decay)
        # optY = SGD(lr, 0.0, weight_decay)
        # optX = RMSProp(lr, 0.99, weight_decay)
        # optY = RMSProp(lr, 0.99, weight_decay)
        optX = ADAM(lr, 0.5, 0.999, weight_decay)
        optY = ADAM(lr, 0.5, 0.999, weight_decay)

        # Start optimizing
        tot_er_x = 0
        tot_er_y = 0
        tot_er_cycx = 0
        tot_er_cycy = 0

        for i in range(batch_n):

            # Put numpy data into tensors
            idx_np = i * batch_size + np.arange(batch_size)

            idx = rpa[idx_np]
            if do_reciprocal:
                idx = bb_idx_x[idx]

            x = x_all[:, idx]
            y_t = y_t_all[:, idx]
            er_x, grad_x_TX = mse_fn(x, y_t, self.TX)
            er_cycx, grad_cycx_TX, grad_cycx_TY = cyc_fn(x, self.TX, self.TY)

            idx = rpb[idx_np]
            if do_reciprocal:
                idx = bb_idx_y[idx]

            y = y_all[:, idx]
            x_t = x_t_all[:, idx]
            er_y, grad_y_TY = mse_fn(y, x_t, self.TY)
            er_cycy, grad_cycy_TY, grad_cycy_TX = cyc_fn(y, self.TY, self.TX)

            tot_er_x += er_x
            tot_er_cycx += er_cycx
            tot_er_y += er_y
            tot_er_cycy += er_cycy

            grad_TX = grad_x_TX + cyc_k * (grad_cycx_TX + grad_cycy_TX)
            grad_TY = grad_y_TY + cyc_k * (grad_cycx_TY + grad_cycy_TY)

            self.TX -= optX.step(grad_TX)
            self.TY -= optY.step(grad_TY)

        if self.verbose:
            print("Epoch: %d" % epoch)
            print("NN: %f/%f" % (rec_x, rec_y))
            print("TC: T: %f/%f Cyc: %f/%f " % (tot_er_x/batch_n,
                                                tot_er_y/batch_n,
                                                tot_er_cycx/batch_n,
                                                tot_er_cycy/batch_n))

        return indices_x, indices_y, min(rec_x, rec_y), min(len(bb_idx_x), len(bb_idx_y))


class ICPTrainer():
    def __init__(self, x_all, y_all, is_pca, n_pca):
        if is_pca:
            x_all = self.pca_reduction(x_all.T, n_pca).T
            y_all = self.pca_reduction(y_all.T, n_pca).T
        self.x_all = x_all.astype(np.float32)
        self.y_all = y_all.astype(np.float32)
        self.icp = _ICP(n_pca)

    def pca_reduction(self, f, n_pca):
        f -= f.mean(1)[:, None]
        f /= f.std(1)[:, None]
        (U, s, V) = pca(f, n_pca, True, n_iter=20)
        f = f.dot(V.T)
        f /= f.std()
        return f

    def sd_norm(self, f):
        f -= f.mean(1)[:, None]
        f /= f.std(1)[:, None]
        return f

    def train_icp(self, n_epochs, is_init=False,
                  indices_x=None, indices_y=None, do_reciprocal=False):
        opt_params = OptParams(lr=1e-2, batch_size=128, epochs=n_epochs,
                               decay_epochs=100, decay_rate=0.5, cyc_k=1e-2)
        indices_x, indices_y, rec, bb = self.icp.train(self.x_all, self.y_all,
                                                       opt_params, is_init,
                                                       indices_x, indices_y,
                                                       do_reciprocal)
        return indices_x, indices_y, rec, bb
