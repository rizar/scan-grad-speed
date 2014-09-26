Faster backpropagation for a simple RNN
===============================

I tried to build computation graph faster that the one produced 
by TT.grad. The source code is in `rnn_opt.py`. In the case of 
a simple RNN considered here the only issue I had to fix 
was inefficient computation of the gradient with respect to the parameter matrix W.

Results
-------

For 50 calls on TITAN BLACK:

* TT.grad: 2.95s
* my grad 1.91s

Where does this difference come from? Let's look at profiling in `profile-rnnopt-gpu-bart10.txt`:

* TT.grad:
    * forward pass calls: 0.67s
    * forward pass overhead: 0.12s
    * backward pass calls: 1.55s
    * backward pass overhead: 0.48s

* My grad:
    * forward pass calls: 0.69s
    * forward pass overhead: 0.14s
    * backward pass calls: 0.66s
    * backward pass overhead: 0.14s
    * big dot product to compute gradient for W: 0.16s

