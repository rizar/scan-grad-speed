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
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    n = 1000
    m = 80

    x = TT.matrix('x')
    y = TT.matrix('y')

    a = nr.normal(size=(n, m), scale=0.01).astype("float32")
    b = nr.normal(size=(n, n), scale=0.01).astype("float32")

    logger.info("Compiling")
    func1 = theano.function(inputs=[x, y], outputs=[x.dot(y)], name='case1cc')
    func2 = theano.function(inputs=[x, y], outputs=[x.T.dot(y)], name='case1fc')
    func3 = theano.function(inputs=[x, y], outputs=[x.dot(y.T)], name='case1cf')
    func4 = theano.function(inputs=[x, y], outputs=[x.T.dot(y.T)], name='case1ff')
    func5 = theano.function(inputs=[x, y], outputs=[x.dot(y)], name='case2cc')
    func6 = theano.function(inputs=[x, y], outputs=[x.T.dot(y)], name='case2fc')
    func7 = theano.function(inputs=[x, y], outputs=[x.dot(y.T)], name='case2cf')
    func8 = theano.function(inputs=[x, y], outputs=[x.T.dot(y.T)], name='case2ff')

    logger.info("Run")
    times = 500
    for i in range(times):
        func1(a, a.T)
        func2(a.T, a.T)
        func3(a, a)
        func4(a.T, a)
        func5(a.T, b)
        func6(a, b)
        func7(a.T, b)
        func8(a, b)
    logger.info("Finished")

if __name__ == "__main__":
    main()

