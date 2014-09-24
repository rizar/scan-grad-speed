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

from groundhog.layers import RecurrentLayer

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

    # Variables and their values
    x = TT.matrix('x')
    x_value = nr.normal(size=(n_words, dim)).astype("float32")

    ri = TT.matrix('ri')
    ri_value = x_value

    zi = TT.matrix('zi')
    zi_value = x_value

    mask = TT.matrix('mask')
    mask_value = numpy.ones((seq_len, batch_size)).astype("float32")

    # Build computations graphs
    rec_layer = RecurrentLayer(
        rng=nr.RandomState(1),
        n_hids=dim,
        gating=True,
        reseting=True,
        init_fn="sample_weights_classic",
        name="rec")
    hs = rec_layer(
        state_below=x,
        mask=mask,
        gater_below=zi,
        reseter_below=ri,
        nsteps=seq_len,
        batch_size=batch_size).out
    cost = hs[-1].sum()
    grad = TT.grad(cost, rec_layer.params)

    logger.info("Compile a function")
    func = theano.function(inputs=[x, ri, zi, mask], outputs=grad)
    TP.pydotprint(func, outfile=args.name, scan_graphs=True)

    logger.info("Run the function")
    for i in range(5):
        func(x_value, ri_value, zi_value, mask_value)

    logger.info("Finished")

if __name__ == "__main__":
    main()

