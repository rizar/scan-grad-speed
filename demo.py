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
    U = theano.shared(nr.normal(size=(dim, dim)).astype("float32"))
    V = theano.shared(U.get_value())
    W = theano.shared(U.get_value())

    # Variables and their values
    x = TT.tensor3('x')
    x_value = nr.normal(size=(seq_len, batch_size, dim)).astype("float32")

    ri = TT.tensor3('ri')
    ri_value = x_value

    zi = TT.tensor3('zi')
    zi_value = x_value

    init = TT.alloc(numpy.float32(0), batch_size, dim)

    # Build computations graph
    def rnn_step(
            # sequences
            x, ri, zi,
            # outputs_info
            h):
        pre_r = x + h.dot(U)
        pre_z = x + h.dot(V)
        r = TT.nnet.sigmoid(pre_r)
        z = TT.nnet.sigmoid(pre_z)

        after_r = r * h
        pre_h = x + after_r.dot(W)
        new_h = TT.tanh(pre_h)

        res_h = z * new_h + (1 - z) * h
        assert res_h.ndim == h.ndim
        return res_h
    hs, _ = theano.scan(rnn_step,
            sequences=[x, ri, zi],
            n_steps=seq_len,
            outputs_info=init)
    cost = hs[-1].sum()
    grad = TT.grad(cost, [U, V, W])

    logger.info("Compile a function")
    func = theano.function(inputs=[x, ri, zi], outputs=grad)
    TP.pydotprint(func, outfile=args.name, scan_graphs=True)

    logger.info("Run the function")
    for i in range(5):
        func(x_value, ri_value, zi_value)

    logger.info("Finished")

if __name__ == "__main__":
    main()

