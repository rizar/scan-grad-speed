Results
-------

In this miniproject I investigate speed of Theano implementation of 
backpropagation through time for a gated RNN [1]. 
I compare three methods: 

* using TT.grad as provided by Theano
* using my own backward pass
* taking TT.grad, but forcing additional outputs for forward pass

50 calls on GTX 480

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
    


