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
    z = TT.matrix('z')

    a = nr.normal(size=(n, m), scale=0.01).astype("float32")
    b = nr.normal(size=(n, n), scale=0.01).astype("float32")

    logger.info("Compiling DOTs")
    func1 = theano.function(inputs=[x, y], outputs=[x.dot(y)], name='dot1cc')
    func2 = theano.function(inputs=[x, y], outputs=[x.T.dot(y)], name='dot1fc')
    func3 = theano.function(inputs=[x, y], outputs=[x.dot(y.T)], name='dot1cf')
    func4 = theano.function(inputs=[x, y], outputs=[x.T.dot(y.T)], name='dot1ff')
    func5 = theano.function(inputs=[x, y], outputs=[x.dot(y)], name='dot2cc')
    func6 = theano.function(inputs=[x, y], outputs=[x.T.dot(y)], name='dot2fc')
    func7 = theano.function(inputs=[x, y], outputs=[x.dot(y.T)], name='dot2cf')
    func8 = theano.function(inputs=[x, y], outputs=[x.T.dot(y.T)], name='dot2ff')
    logger.info("Compiling GEMMs")
    func9 = theano.function(inputs=[x, y, z], outputs=[x.dot(y) + z], name="gemm1cc")
    func10 = theano.function(inputs=[x, y, z], outputs=[x.T.dot(y) + z], name="gemm1fc")
    func11 = theano.function(inputs=[x, y, z], outputs=[x.dot(y.T) + z], name="gemm1cf")
    func12 = theano.function(inputs=[x, y, z], outputs=[x.T.dot(y.T) + z], name="gemm1ff")

    logger.info("Run")
    times = 50 if theano.config.device == 'cpu' else 500
    for i in range(times):
        func1(a, a.T)
        func2(a.T, a.T)
        func3(a, a)
        func4(a.T, a)
        func5(a.T, b)
        func6(a, b)
        func7(a.T, b)
        func8(a, b)
        func9(a, a.T, b)
        func10(a.T, a.T, b)
        func11(a, a, b)
        func12(a.T, a, b)
    logger.info("Finished")

if __name__ == "__main__":
    main()

