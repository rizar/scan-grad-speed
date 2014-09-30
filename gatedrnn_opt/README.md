In this miniproject I investigate speed of Theano implementation of 
backpropagation through time for a gated RNN [1]. 
I compare three methods: 

* using TT.grad as provided by Theano
* using my own backward pass that moves some dot products from internal
scan graph and merges them
* taking TT.grad, but forcing additional outputs for the forward pass

The code is in `gatedrnn_opt.py`, the profiles are in `profile-gatedopt-580.txt` and
`profmem-gatedopt-quadro.txt` for different gpus respectively. Pydotprint outputs
are also available.

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

Discussion
----------
    
* For the third option one would expect to see 6 matrix multiplications in `grad_of_fpass3` scan
(3 for propagating the gradient back and 3 for accumulation of gradient w.r. weight matrices). 
There are however 7, indicating the fact that the hidden states before the reset gate
are recomputed, despite being available as a forward pass output
(one can see that at `gatedrnn-cpu-laptop3_grad_of_fpass3_62.png`. It should be fixed.

* It should be possible to write a scan optimization that does what I did here. Already
available `PushOutDot1` is a simplified version of what we need.

* A potential reduction of overhead would help but not critically (takes less than 15% in total).

References
----------

Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk and Yoshua Bengio. Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. EMNLP 2014. http://arxiv.org/abs/1406.1078
