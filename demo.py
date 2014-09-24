import numpy
import theano
import theano.tensor as TT

def init_bias(size, scale, rng):
    return numpy.ones((size,), dtype=theano.config.floatX)*scale

def sample_weights_classic(sizeX, sizeY, sparsity, scale, rng):
    sizeX = int(sizeX)
    sizeY = int(sizeY)
    if sparsity < 0:
        sparsity = sizeY
    else:
        sparsity = numpy.minimum(sizeY, sparsity)
    sparsity = numpy.minimum(sizeY, sparsity)
    values = numpy.zeros((sizeX, sizeY), dtype=theano.config.floatX)
    for dx in xrange(sizeX):
        perm = rng.permutation(sizeY)
        new_vals = rng.normal(loc=0, scale=scale, size=(sparsity,))
        values[dx, perm[:sparsity]] = new_vals
    return values.astype(theano.config.floatX)

class RecurrentLayer(Layer):
    """
        Standard recurrent layer with gates.
        See arXiv verion of our paper.
    """
    def __init__(self, rng,
                 n_hids=500,
                 scale=.01,
                 sparsity = -1,
                 activation = TT.tanh,
                 activ_noise=0.,
                 weight_noise=False,
                 bias_fn='init_bias',
                 bias_scale = 0.,
                 dropout = 1.,
                 init_fn='sample_weights',
                 kind_reg = None,
                 grad_scale = 1.,
                 profile = 0,
                 gating = False,
                 reseting = False,
                 gater_activation = TT.nnet.sigmoid,
                 reseter_activation = TT.nnet.sigmoid,
                 name=None):
        """
        :type rng: numpy random generator
        :param rng: numpy random generator

        :type n_in: int
        :param n_in: number of inputs units

        :type n_hids: int
        :param n_hids: Number of hidden units on each layer of the MLP

        :type activation: string/function or list of
        :param activation: Activation function for the embedding layers. If
            a list it needs to have a value for each layer. If not, the same
            activation will be applied to all layers

        :type scale: float or list of
        :param scale: depending on the initialization function, it can be
            the standard deviation of the Gaussian from which the weights
            are sampled or the largest singular value. If a single value it
            will be used for each layer, otherwise it has to have one value
            for each layer

        :type sparsity: int or list of
        :param sparsity: if a single value, it will be used for each layer,
            otherwise it has to be a list with as many values as layers. If
            negative, it means the weight matrix is dense. Otherwise it
            means this many randomly selected input units are connected to
            an output unit


        :type weight_noise: bool
        :param weight_noise: If true, the model is used with weight noise
            (and the right shared variable are constructed, to keep track of the
            noise)

        :type dropout: float
        :param dropout: the probability with which hidden units are dropped
            from the hidden layer. If set to 1, dropout is not used

        :type init_fn: string or function
        :param init_fn: function used to initialize the weights of the
            layer. We recommend using either `sample_weights_classic` or
            `sample_weights` defined in the utils

        :type bias_fn: string or function
        :param bias_fn: function used to initialize the biases. We recommend
            using `init_bias` defined in the utils

        :type bias_scale: float
        :param bias_scale: argument passed to `bias_fn`, depicting the scale
            of the initial bias

        :type grad_scale: float or theano scalar
        :param grad_scale: factor with which the gradients with respect to
            the parameters of this layer are scaled. It is used for
            differentiating between the different parameters of a model.

        :type gating: bool
        :param gating: If true, an update gate is used

        :type reseting: bool
        :param reseting: If true, a reset gate is used

        :type gater_activation: string or function
        :param name: The activation function of the update gate

        :type reseter_activation: string or function
        :param name: The activation function of the reset gate

        :type name: string
        :param name: name of the layer (used to name parameters). NB: in
            this library names are very important because certain parts of the
            code relies on name to disambiguate between variables, therefore
            each layer should have a unique name.

        """
        self.grad_scale = grad_scale

        if type(init_fn) is str or type(init_fn) is unicode:
            init_fn = eval(init_fn)
        if type(bias_fn) is str or type(bias_fn) is unicode:
            bias_fn = eval(bias_fn)
        if type(activation) is str or type(activation) is unicode:
            activation = eval(activation)
        if type(gater_activation) is str or type(gater_activation) is unicode:
            gater_activation = eval(gater_activation)
        if type(reseter_activation) is str or type(reseter_activation) is unicode:
            reseter_activation = eval(reseter_activation)

        self.scale = scale
        self.sparsity = sparsity
        self.activation = activation
        self.n_hids = n_hids
        self.bias_scale = bias_scale
        self.bias_fn = bias_fn
        self.init_fn = init_fn
        self.weight_noise = weight_noise
        self.activ_noise = activ_noise
        self.profile = profile
        self.dropout = dropout
        self.gating = gating
        self.reseting = reseting
        self.gater_activation = gater_activation
        self.reseter_activation = reseter_activation

        assert rng is not None, "random number generator should not be empty!"

        super(RecurrentLayer, self).__init__(self.n_hids,
                self.n_hids, rng, name)

        self.trng = RandomStreams(self.rng.randint(int(1e6)))
        self.params = []
        self._init_params()

    def _init_params(self):
        self.W_hh = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="W_%s"%self.name)
        self.params = [self.W_hh]
        if self.gating:
            self.G_hh = theano.shared(
                    self.init_fn(self.n_hids,
                    self.n_hids,
                    self.sparsity,
                    self.scale,
                    rng=self.rng),
                    name="G_%s"%self.name)
            self.params.append(self.G_hh)
        if self.reseting:
            self.R_hh = theano.shared(
                    self.init_fn(self.n_hids,
                    self.n_hids,
                    self.sparsity,
                    self.scale,
                    rng=self.rng),
                    name="R_%s"%self.name)
            self.params.append(self.R_hh)
        self.params_grad_scale = [self.grad_scale for x in self.params]
        self.restricted_params = [x for x in self.params]
        if self.weight_noise:
            self.nW_hh = theano.shared(self.W_hh.get_value()*0, name='noise_'+self.W_hh.name)
            self.nG_hh = theano.shared(self.G_hh.get_value()*0, name='noise_'+self.G_hh.name)
            self.noise_params = [self.nW_hh,self.nG_hh]
            self.noise_params_shape_fn = [constant_shape(x.get_value().shape)
                            for x in self.noise_params]

    def step_fprop(self,
                   state_below,
                   mask = None,
                   state_before = None,
                   gater_below = None,
                   reseter_below = None,
                   use_noise=True,
                   no_noise_bias = False):
        """
        Constructs the computational graph of this layer.

        :type state_below: theano variable
        :param state_below: the input to the layer

        :type mask: None or theano variable
        :param mask: mask describing the length of each sequence in a
            minibatch

        :type state_before: theano variable
        :param state_before: the previous value of the hidden state of the
            layer

        :type gater_below: theano variable
        :param gater_below: the input to the update gate

        :type reseter_below: theano variable
        :param reseter_below: the input to the reset gate

        :type use_noise: bool
        :param use_noise: flag saying if weight noise should be used in
            computing the output of this layer

        :type no_noise_bias: bool
        :param no_noise_bias: flag saying if weight noise should be added to
            the bias as well
        """

        rval = []
        if self.weight_noise and use_noise and self.noise_params:
            W_hh = self.W_hh + self.nW_hh
            if self.gating:
                G_hh = self.G_hh + self.nG_hh
            if self.reseting:
                R_hh = self.R_hh + self.nR_hh
        else:
            W_hh = self.W_hh
            if self.gating:
                G_hh = self.G_hh
            if self.reseting:
                R_hh = self.R_hh

        # Reset gate:
        # optionally reset the hidden state.
        if self.reseting and reseter_below:
            reseter = self.reseter_activation(TT.dot(state_before, R_hh) +
                    reseter_below)
            reseted_state_before = reseter * state_before
        else:
            reseted_state_before = state_before

        # Feed the input to obtain potential new state.
        preactiv = TT.dot(reseted_state_before, W_hh) + state_below
        h = self.activation(preactiv)

        # Update gate:
        # optionally reject the potential new state and use the new one.
        if self.gating and gater_below:
            gater = self.gater_activation(TT.dot(state_before, G_hh) +
                    gater_below)
            h = gater * h + (1-gater) * state_before

        if self.activ_noise and use_noise:
            h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
        if mask is not None:
            if h.ndim ==2 and mask.ndim==1:
                mask = mask.dimshuffle(0,'x')
            h = mask * h + (1-mask) * state_before
        return h

    def fprop(self,
              state_below,
              mask=None,
              init_state=None,
              gater_below=None,
              reseter_below=None,
              nsteps=None,
              batch_size=None,
              use_noise=True,
              truncate_gradient=-1,
              no_noise_bias = False
             ):

        if theano.config.floatX=='float32':
            floatX = numpy.float32
        else:
            floatX = numpy.float64
        if nsteps is None:
            nsteps = state_below.shape[0]
            if batch_size and batch_size != 1:
                nsteps = nsteps / batch_size
        if batch_size is None and state_below.ndim == 3:
            batch_size = state_below.shape[1]
        if state_below.ndim == 2 and \
           (not isinstance(batch_size,int) or batch_size > 1):
            state_below = state_below.reshape((nsteps, batch_size, self.n_in))
            if gater_below:
                gater_below = gater_below.reshape((nsteps, batch_size, self.n_in))
            if reseter_below:
                reseter_below = reseter_below.reshape((nsteps, batch_size, self.n_in))

        if not init_state:
            if not isinstance(batch_size, int) or batch_size != 1:
                init_state = TT.alloc(floatX(0), batch_size, self.n_hids)
            else:
                init_state = TT.alloc(floatX(0), self.n_hids)

        # FIXME: Find a way to clean this up
        if self.reseting and reseter_below:
            if self.gating and gater_below:
                if mask:
                    inps = [state_below, mask, gater_below, reseter_below]
                    fn = lambda x,y,g,r,z : self.step_fprop(x,y,z, gater_below=g, reseter_below=r, use_noise=use_noise,
                                                       no_noise_bias=no_noise_bias)
                else:
                    inps = [state_below, gater_below, reseter_below]
                    fn = lambda tx, tg,tr, ty: self.step_fprop(tx, None, ty, gater_below=tg,
                                                        reseter_below=tr,
                                                        use_noise=use_noise,
                                                        no_noise_bias=no_noise_bias)
            else:
                if mask:
                    inps = [state_below, mask, reseter_below]
                    fn = lambda x,y,r,z : self.step_fprop(x,y,z, use_noise=use_noise,
                                                        reseter_below=r,
                                                       no_noise_bias=no_noise_bias)
                else:
                    inps = [state_below, reseter_below]
                    fn = lambda tx,tr,ty: self.step_fprop(tx, None, ty,
                                                        reseter_below=tr,
                                                        use_noise=use_noise,
                                                        no_noise_bias=no_noise_bias)
        else:
            if self.gating and gater_below:
                if mask:
                    inps = [state_below, mask, gater_below]
                    fn = lambda x,y,g,z : self.step_fprop(x,y,z, gater_below=g, use_noise=use_noise,
                                                       no_noise_bias=no_noise_bias)
                else:
                    inps = [state_below, gater_below]
                    fn = lambda tx, tg, ty: self.step_fprop(tx, None, ty, gater_below=tg,
                                                        use_noise=use_noise,
                                                        no_noise_bias=no_noise_bias)
            else:
                if mask:
                    inps = [state_below, mask]
                    fn = lambda x,y,z : self.step_fprop(x,y,z, use_noise=use_noise,
                                                       no_noise_bias=no_noise_bias)
                else:
                    inps = [state_below]
                    fn = lambda tx, ty: self.step_fprop(tx, None, ty,
                                                        use_noise=use_noise,
                                                        no_noise_bias=no_noise_bias)

        rval, updates = theano.scan(fn,
                        sequences = inps,
                        outputs_info = [init_state],
                        name='layer_%s'%self.name,
                        profile=self.profile,
                        truncate_gradient = truncate_gradient,
                        n_steps = nsteps)
        new_h = rval
        self.out = rval
        self.rval = rval
        self.updates =updates

        return self.out
