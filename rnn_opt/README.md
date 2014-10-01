Faster backpropagation for a simple RNN
===============================

I tried to build computation graph faster that the one produced 
by TT.grad. The source code is in `rnn_opt.py`. In the case of 
a simple RNN considered here the only issue I had to fix 
was inefficient computation of the gradient with respect to the parameter matrix W.

Results
-------

For 50 calls on TITAN BLACK:

* TT.grad: 4.73s
* my grad: 2.79s

Where does this difference come from? Let's look at profiling in `profile-rnnopt-quadro.txt`:

* TT.grad:
    * forward pass calls: 0.86s
    * forward pass overhead: 0.14s
    * backward pass calls: 3.37s
    * backward pass overhead: 0.85s

* My grad:
    * forward pass calls: 0.85
    * forward pass overhead: 0.18s
    * backward pass calls: 1.04s
    * backward pass overhead: 0.25s
    * big dot product to compute gradient for W: 0.21s
    * reshapes: 0.16s

Questions
---------

* Why so much overhead? Is it normal?
* Is it possible somehow to make the trick I used an optimization in theano?

