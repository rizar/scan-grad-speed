I try to investigate why and under which circumstances 
computing the gradient of scan takes so much longer than
doing scan itself. To do that I created a script demo.py,
that imitates decoding part of a machine translation architecture.
The script currently depends of groundhog. The profiling results
and also pydotprint outputs are also stored in this repository.

So far I tried three different setups:

* at my laptop (cpu-rizar-laptop), scan 
