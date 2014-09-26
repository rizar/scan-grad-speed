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
    n_words = batch_size * seq_len
    dim = 1000

    # Weight matrices
    W = theano.shared(nr.normal(size=(dim, dim), scale=0.0001).astype("float32"))
    WT = theano.shared(W.get_value().T)

    # Variables and their values
    x = TT.tensor3('x')
    x_value = nr.normal(size=(seq_len, batch_size, dim), scale=0.0001).astype("float32")

    # Backward pass
    def grad_step(
            # sequences
            x,
            h, pre_h,
            # outputs_info
            e_h_next):
        e_pre_h = e_h_next / TT.cosh(pre_h) ** 2
        e_h = e_pre_h.dot(WT)
        return e_h

    # Forward pass
    def rnn_step(
            # sequences
            x,
            # outputs_info
            h):
        pre_h = x + h.dot(W)
        new_h = TT.tanh(pre_h)
        return new_h, pre_h

    (h, pre_h), _ = theano.scan(rnn_step,
            sequences=[x],
            n_steps=seq_len,
            outputs_info=[TT.zeros_like(x[0]), None],
            name='fpass')
    cost = h[-1].sum()
    grad1 = TT.grad(cost, [W])

    h = TT.concatenate([
        TT.shape_padleft(TT.zeros_like(h[0])),
        h[:-1]])
    pre_h = TT.concatenate([
        TT.shape_padleft(TT.zeros_like(pre_h[0])),
        pre_h[:-1]])
    eh, _ = theano.scan(grad_step,
            sequences=[x, h, pre_h],
            n_steps=seq_len,
            outputs_info=[TT.ones_like(x[0])],
            go_backwards=True,
            name='bpass')
    eh = TT.concatenate([
        eh[1:],
        TT.shape_padleft(TT.ones_like(eh[0]))])
    h = h.dimshuffle(2, 0, 1).reshape((dim, n_words))
    eh = eh.dimshuffle(2, 0, 1).reshape((dim, n_words)).T
    eW = h.dot(eh)
    grad2 = [eW]

    logger.info("Compile a function")
    func1 = theano.function(inputs=[x], outputs=grad1, name="grad1")
    func2 = theano.function(inputs=[x], outputs=grad2, name="grad2")
    # TP.pydotprint(func, outfile=args.name, scan_graphs=True)

    logger.info("Run the function")
    on_gpu = theano.config.device == 'gpu'
    times = 1
    if on_gpu:
        times = 50
    for i in range(times):
        g1 = func1(x_value)[0]
        g2 = func2(x_value)[0]
        if not on_gpu:
            print g1.sum(), g2.sum()

    logger.info("Finished")

if __name__ == "__main__":
    main()

