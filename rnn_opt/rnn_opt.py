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
    parser.add_argument('name')
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
    W.name = 'W'
    WT = theano.shared(W.get_value().T)
    WT.name = 'WT'

    # Variables and their values
    x = TT.tensor3('x')
    x_value = nr.normal(size=(seq_len, batch_size, dim), scale=0.0001).astype("float32")

    # Backward pass
    def grad_step(
            # sequences
            h, mult,
            # outputs_info
            e_h_next):
        h.name = 'h'
        mult.name = 'mul'
        e_h_next.name = 'e_h_next'

        e_pre_h = e_h_next * mult
        e_pre_h.name = 'e_pre_h'
        e_h = e_pre_h.dot(WT)
        e_h.name = 'e_h'
        return e_h

    # Forward pass
    def rnn_step(
            # sequences
            x,
            # outputs_info
            h):
        x.name = 'x'
        h.name = 'h'

        pre_h = x + h.dot(W)
        pre_h.name = 'pre_h'
        new_h = TT.tanh(pre_h)
        new_h.name = 'new_h'
        return new_h, pre_h

    (h, pre_h), _ = theano.scan(rnn_step,
            sequences=[x],
            n_steps=seq_len,
            outputs_info=[TT.zeros_like(x[0]), None],
            name='fpass')
    cost = h[-1].sum()
    grad1 = TT.grad(cost, [W])

    mult = 1 - h ** 2
    mult.name = 'mult'
    h = TT.concatenate([
        TT.shape_padleft(TT.zeros_like(h[0])),
        h[:-1]])
    h.name = 'h*'
    eh, _ = theano.scan(grad_step,
            sequences=[h, mult],
            n_steps=seq_len,
            outputs_info=[TT.ones_like(x[0])],
            go_backwards=True,
            name='bpass')
    eh.name = 'eh'
    eh = TT.concatenate([
        eh[1:],
        TT.shape_padleft(TT.ones_like(eh[0]))])
    eh.name = 'eh*'
    h = h.dimshuffle(2, 0, 1).reshape((dim, n_words))
    h.name = 'h_shu'
    eh = eh.dimshuffle(2, 0, 1).reshape((dim, n_words)).T
    eh.name = 'eh_shu'
    eW = h.dot(eh)
    eW.name = 'eW'
    grad2 = [eW]

    logger.info("Compile a function")
    func1 = theano.function(inputs=[x], outputs=grad1, name="grad1")
    TP.pydotprint(func1, outfile=args.name, scan_graphs=True)
    func2 = theano.function(inputs=[x], outputs=grad2, name="grad2")

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

