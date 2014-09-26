#!/usr/bin/env python

import sys
import numpy
import theano
import argparse
import numpy.random as nr
import theano.tensor as TT
import theano.printing as TP
import logging
logger = logging.getLogger(__name__)

def watch(x):
    def func(_, x):
        import ipdb; ipdb.set_trace()
    return TP.Print(global_fn=func)(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Codename of this run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    # Parameters from an actual machine tranlation run
    batch_size = 80
    seq_len = 50
    n_words = 80 * 50
    dim = 1000

    # Weight matrices
    U = theano.shared(nr.normal(size=(dim, dim), scale=0.0001).astype("float32"))
    U.name = 'U'
    V = theano.shared(U.get_value())
    V.name = 'V'
    W = theano.shared(U.get_value())
    W.name = 'W'

    # Variables and their values
    x = TT.tensor3('x')
    x_value = nr.normal(size=(seq_len, batch_size, dim), scale=0.0001).astype("float32")

    ri = TT.tensor3('ri')
    ri_value = x_value

    zi = TT.tensor3('zi')
    zi_value = x_value

    init = TT.alloc(numpy.float32(0), batch_size, dim)

    # Backward pass
    def grad_step(
            # sequences
            h, r, z, new_h,
            # outputs_info
            e_h_next):

        # Duplicate forward propagation
        # pre_r = ri + h.dot(U)
        # pre_z = zi + h.dot(V)
        # r = TT.nnet.sigmoid(pre_r) !
        # z = TT.nnet.sigmoid(pre_z) !

        # after_r = r * h
        # pre_h = x + after_r.dot(W)
        # new_h = TT.tanh(pre_h) !

        # h_next = z * new_h + (1 - z) * h

        # Push the gradient through the update gates
        e_h = (1 - z) * e_h_next
        e_new_h = z * e_h_next
        e_z = (new_h - h) * e_h_next

        # Push the gradient through tanh
        e_pre_h = e_new_h * (1 - new_h ** 2)

        # Push the gradint through the reset gates
        e_after_r = e_pre_h.dot(W.T)
        e_h += r * e_after_r
        e_r = h * e_after_r

        # Push the gate gradients
        e_pre_r = r * (1 - r) * e_r
        e_pre_z = z * (1 - z) * e_z
        e_h += e_pre_r.dot(U.T)
        e_h += e_pre_z.dot(V.T)

        return e_h, e_pre_r, e_pre_z, e_pre_h

    # Forward pass
    def rnn_step1(
            # sequences
            x, ri, zi,
            # outputs_info
            h):
        pre_r = ri + h.dot(U)
        pre_z = zi + h.dot(V)
        r = TT.nnet.sigmoid(pre_r)
        z = TT.nnet.sigmoid(pre_z)

        after_r = r * h
        pre_h = x + after_r.dot(W)
        new_h = TT.tanh(pre_h)

        res_h = z * new_h + (1 - z) * h
        return res_h

    # Forward pass
    def rnn_step3(
            # sequences
            x, ri, zi,
            # outputs_info
            h):
        pre_r = ri + h.dot(U)
        pre_z = zi + h.dot(V)
        r = TT.nnet.sigmoid(pre_r)
        z = TT.nnet.sigmoid(pre_z)

        after_r = r * h
        pre_h = x + after_r.dot(W)
        new_h = TT.tanh(pre_h)

        res_h = z * new_h + (1 - z) * h
        return res_h, r, z, new_h

    # Gradient computation - method 1
    h, _ = theano.scan(rnn_step1,
            sequences=[x, ri, zi],
            n_steps=seq_len,
            outputs_info=init,
            name='fpass1')
    cost = h[-1].sum()
    grad1 = TT.grad(cost, [U, V, W])

    # Gradient computation - method2
    res, _ = theano.scan(rnn_step3,
            sequences=[x, ri, zi],
            n_steps=seq_len,
            outputs_info=[init, None, None, None],
            name='fpass2')
    def shift_right(x):
        return TT.concatenate([
            TT.shape_padleft(TT.zeros_like(x[0])),
            x[:-1]])
    h, r, z, new_h = res
    h = shift_right(h)
    (e_h, e_pre_r, e_pre_z, e_pre_h), _ = theano.scan(grad_step,
            sequences=[h, r, z, new_h],
            n_steps=seq_len,
            go_backwards=True,
            outputs_info=[TT.ones_like(h[0]), None, None, None],
            name='bpass2')
    def reshape(x):
        return x.dimshuffle(2, 0, 1).reshape((dim, n_words))
    (h, r, e_pre_r, e_pre_z, e_pre_h) = map(reshape,
            [h, r, e_pre_r[::-1], e_pre_z[::-1], e_pre_h[::-1]])
    eU = h.dot(e_pre_r.T)
    eV = h.dot(e_pre_z.T)
    eW = (h * r).dot(e_pre_h.T)
    grad2 = [eU, eV, eW]

    # Gradient computation - method3
    res, _ = theano.scan(rnn_step3,
            sequences=[x, ri, zi],
            n_steps=seq_len,
            outputs_info=[init, None, None, None],
            name='fpass3')
    h = res[0]
    cost = h[-1].sum()
    grad3 = TT.grad(cost, [U, V, W])

    logger.info("Compile a function")
    func1 = theano.function(inputs=[x, ri, zi], outputs=grad1, name="grad1")
    func2 = theano.function(inputs=[x, ri, zi], outputs=grad2, name="grad2")
    func3 = theano.function(inputs=[x, ri, zi], outputs=grad3, name="grad3")

    logger.info("Run the function")
    for i in range(1):
        g1 = func1(x_value, ri_value, zi_value)
        g2 = func2(x_value, ri_value, zi_value)
        g3 = func3(x_value, ri_value, zi_value)
        for g in [g1, g2, g3]:
            print map(lambda x : x.sum(), g)
        for v1, v2 in zip(g1, g2):
            print numpy.sum(numpy.abs(v1 - v2))

    TP.pydotprint(func1, outfile=args.name + "1", scan_graphs=True)
    TP.pydotprint(func2, outfile=args.name + "2", scan_graphs=True)
    TP.pydotprint(func3, outfile=args.name + "3", scan_graphs=True)
    logger.info("Finished")

if __name__ == "__main__":
    main()

