I tried to investigate why and under which circumstances 
computing the gradient of scan takes so much longer than
doing scan itself. Three experiments were conducted:

* optimizing gradient computation for a simple RNN, see 'rnn_opt'
* optimizing gradient computation for a gaten RNN, see 'gatedrnn_opt'
* benchmarking matrix multiplications on GPU, see 'dotspeed'

The following version of theano was used:

* Theano: 239b6d8001e290f6c65b6f516a75e6bb1594fb02

Conclusions
=======

* We need a new optimization for scan. 
It should detect the following pattern. Let W be an
output of the scan, which _is not used in the scan as input_.
 Let A[i] and B[i] be some intermediate 
matrices of the internal scan graph 
which are used to compute W as follows:

        for i in range(0, n):
            W += A[i].dot(B[i])

where `for` means iterations of the scan. We should replace it with

        C = concatenate(A[0], A[1], ..., A[n - 1], axis=1)
        D = concatenate(B[0], B[1], ..., B[n - 1])
        W = C.dot(D)

that is we _new outputs_ C and D, which are then used to compute W.
I  could speed up things up to 2 times with this trick.

* Somebody should find out why under certaun circumstances an explicit output
of forward pass is recomputed during the backward pass (see function `grad3` 
in `gatedrnn_opt/gatedrnn_opt.py`)

* Reducing scan overhead could give us some more 15% speedup, but not more.

* As a general rule instead of:

        x2 = x1.dot(U)
        x3 = x1.dot(V) 

it's better to write:

        x = x1.dot([U; V])
        x2 = x[...]
        x3 = x[...]

though I did not measure how much it would speed up the gated RNN
(but I tried in in `dotspeed` and got some speedup).
    


