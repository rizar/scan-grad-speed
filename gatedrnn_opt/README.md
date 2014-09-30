In this miniproject I investigate speed of Theano implementation of 
backpropagation through time for a gated RNN [1]. 
I compare three methods: 

* using TT.grad as provided by Theano
* using my own backward pass
* taking TT.grad, but forcing additional outputs for the forward pass

Results
-------

#####50 calls on Quadro

TT.grad: 15s

* forward pass calls: 2.4s
* forward pass overhead: 0.17s
* backward pass calls: 9.82s
* backward pass overhead: 1.88s

My grad: 8.82s

* forward pass calls: 2.8s
* forward pass overhead: 0.64s
* backward pass calls: 2.9s
* backward pass overhead: 0.62s
* dot product: 0.56s
* reshape: 0.38s

TT.grad + extra outputs: 13.6s

* forward pass calls: 2.68s
* forward pass overhead: 0.46s
* backward pass calls: 7.92s
* backward pass overhead: 1.66s

#####50 calls on GTX 480

TT.grad: 21.8s

* forward pass calls: 3.9s
* forward pass overhead: 0.22s
* backward pass calls: 14.4s
* backward pass overhead: 2.5s

My grad: 13.3s

* forward pass calls: 4.07s
* forward pass overhead: 0.84s
* backward pass calls: 4.41s
* backward pass overhead: 0.81s
* dot product: 1.5s
* reshape: 0.74s

TT.grad + extra outputs: 19.s

* forward pass calls: 4.02s
* forward pass overhead: 0.59s
* backward pass calls: 11.3s
* backward pass overhead: 2.22s

Notes
-----
    
* For the third option one would expect to see 6 matrix multiplication in the gradient scan
(3 for propagating the gradient back and 3 for accumulation of gradient w.r. weight matrices). 
There are however 7, indicating the fact that the hidden states before the reset gate
are recomputed, despite being available as a forward pass output. It should be fixed.

