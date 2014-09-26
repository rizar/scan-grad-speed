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
    dim = 1000

    # Weight matrices
    W = theano.shared(nr.normal(size=(dim, dim), scale=0.0001).astype("float32"))

    # Variables and their values
    x = TT.tensor3('x')
    x_value = nr.normal(size=(seq_len, batch_size, dim), scale=0.0001).astype("float32")

    # Backward pass
    def grad_step(
            # sequences
            x,
            h,
            # outputs_info
            e_h_next, e_W):
        pre_h = x + h.dot(W)
        e_pre_h = e_h_next / TT.cosh(pre_h) ** 2
        e_h = e_pre_h.dot(W.T)
        e_W += h.T.dot(e_pre_h)
        return e_h, e_W

    # Forward pass
    def rnn_step(
            # sequences
            x,
            # outputs_info
            h):
        pre_h = x + h.dot(W)
        new_h = TT.tanh(pre_h)
        return new_h

    h, _ = theano.scan(rnn_step,
            sequences=[x],
            n_steps=seq_len,
            outputs_info=TT.zeros_like(x[0]),
            name='fpass')
    cost = h[-1].sum()
    grad1 = TT.grad(cost, [W])

    h = TT.concatenate([
        TT.shape_padleft(TT.zeros_like(h[0])),
        h[:-1]])
    (eh, eW), _ = theano.scan(grad_step,
            sequences=[x[::-1], h[::-1]],
            n_steps=seq_len,
            outputs_info=[TT.ones_like(x[0]), TT.zeros_like(W)],
            name='bpass')
    grad2 = [eW[-1]]

    logger.info("Compile a function")
    func1 = theano.function(inputs=[x], outputs=grad1, name="grad1")
    func2 = theano.function(inputs=[x], outputs=grad2, name="grad2")
    # TP.pydotprint(func, outfile=args.name, scan_graphs=True)

    logger.info("Run the function")
    on_gpu = theano.config.device == 'gpu'
    times=1
    if on_gpu:
        times=50
    for i in range(times):
        g1 = func1(x_value)[0]
        g2 = func2(x_value)[0]
        if not on_gpu:
            print g1.sum(), g2.sum()

    logger.info("Finished")

if __name__ == "__main__":
    main()

