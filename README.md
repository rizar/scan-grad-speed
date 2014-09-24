I try to investigate why and under which circumstances 
computing the gradient of scan takes so much longer than
doing scan itself. To do that I created a script demo.py,
that imitates decoding part of a machine translation architecture.
The script currently depends of groundhog. The profiling results
and also pydotprint outputs are also stored in this repository.

The following versions of theano and groundhog were used:

* Theano: 239b6d8001e290f6c65b6f516a75e6bb1594fb02
* GroundHog: 4d124e7696d4fb98b3f26d32916165b13bd610e3

Results
=======

So far I tried three different setups:

* at my laptop (cpu-rizar-laptop), forward pass  26.5%, backward pass 73.1%
* at bart10, cpu (cpu-bart10), forward pass 34.1%, backward pass 65.7%
* at bart10, gpu (gpu-bart10), forward pass 15.5%, backward pass 79.6%
