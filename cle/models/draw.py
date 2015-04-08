import ipdb
import numpy as np
import theano
import theano.tensor as T

from cle.cle.data import Iterator
from cle.cle.cost import Gaussian
from cle.cle.models import Model
from cle.cle.models.draw import (
    CanvasLayer,
    ErrorLayer,
    ReadLayer,
    WriteLayer
)
from cle.cle.layers import InitCell
from cle.cle.layers.cost import GaussianLayer
from cle.cle.layers.feedforward import FullyConnectedLayer
from cle.cle.layers.layer import PriorLayer
from cle.cle.layers.recurrent import LSTM
from cle.cle.train import Training
from cle.cle.train.ext import (
    EpochCount,
    GradientClipping,
    Monitoring,
    Picklize,
    EarlyStopping
)
from cle.cle.train.opt import Adam
from cle.cle.utils import flatten
from cle.cle.utils.compat import OrderedDict
from course_works.ift6266h15.datasets.dogs_and_cats import DogsnCats


datapath = '/home/junyoung/data/dogs_and_cats/'
savepath = '/home/junyoung/repos/course_works/ift6266h15/saved/'

batch_size = 100
num_channel = 1

inpsz = 48 * 48 * num_channel
latsz = 300
n_steps = 64

input_shape = (batch_size, num_channel, 48, 48)
read_shape = (batch_size, num_channel, 4, 4)
write_shape = (batch_size, num_channel, 10, 10)

rnn_dim = 512
rnn_input_dim = read_shape[1] * read_shape[2] * read_shape[3]
w_dim = write_shape[1] * write_shape[2] * write_shape[3]

if num_channel == 3:
    use_color = 1
else:
    use_color = 0
debug = 1

model = Model()
trdata = DogsnCats(name='train',
                   path=datapath,
                   use_color=use_color,
                   unsupervised=1,
                   prep='global_normalize')
testdata = DogsnCats(name='test',
                     path=datapath,
                     use_color=use_color,
                     unsupervised=1,
                     X_mean=trdata.X_mean,
                     X_std=trdata.X_std,
                     prep='global_normalize')


init_W = InitCell('rand')
init_U = InitCell('ortho')
init_b = InitCell('zeros')
init_b_sig = InitCell('const', mean=0.6)

x = trdata.theano_vars()
if debug:
    x.tag.test_value = np.zeros((batch_size, inpsz), dtype=np.float32)

error = ErrorLayer(name='error',
                   parent=['x'],
                   recurrent=['sampling'],
                   batch_size=batch_size)

read_param = FullyConnectedLayer(name='read_param',
                                 parent=['dec_tm1'],
                                 parent_dim=[rnn_dim],
                                 nout=5,
                                 unit='linear',
                                 init_W=init_W,
                                 init_b=init_b)

read = ReadLayer(name='read',
                 parent=['x', 'error', 'read_param'],
                 glimpse_shape=read_shape,
                 input_shape=input_shape)

enc = LSTM(name='enc',
           parent=['read'],
           parent_dim=[rnn_input_dim],
           recurrent=['dec'],
           recurrent_dim=[rnn_dim],
           batch_size=batch_size,
           nout=rnn_dim,
           unit='tanh',
           init_W=init_W,
           init_U=init_U,
           init_b=init_b)

phi_mu = FullyConnectedLayer(name='phi_mu',
                             parent=['enc'],
                             parent_dim=[rnn_dim],
                             nout=latsz,
                             unit='linear',
                             init_W=init_W,
                             init_b=init_b)

phi_sig = FullyConnectedLayer(name='phi_sig',
                              parent=['enc'],
                              parent_dim=[rnn_dim],
                              nout=latsz,
                              unit='softplus',
                              init_W=init_W,
                              init_b=init_b_sig)

prior = PriorLayer(name='prior',
                   parent=['phi_mu', 'phi_sig'],
                   parent_dim=[latsz, latsz],
                   use_sample=1,
                   nout=latsz)

kl = PriorLayer(name='kl',
                parent=['phi_mu', 'phi_sig'],
                parent_dim=[latsz, latsz],
                use_sample=0,
                nout=latsz)

dec = LSTM(name='dec',
           parent=['prior'],
           parent_dim=[latsz],
           batch_size=batch_size,
           nout=rnn_dim,
           unit='tanh',
           init_W=init_W,
           init_U=init_U,
           init_b=init_b)

w = FullyConnectedLayer(name='w',
                        parent=['dec'],
                        parent_dim=[rnn_dim],
                        nout=w_dim,
                        unit='linear',
                        init_W=init_W,
                        init_b=init_b)

w_sig = FullyConnectedLayer(name='w_sig',
                            parent=['dec'],
                            parent_dim=[rnn_dim],
                            nout=w_dim,
                            unit='softplus',
                            cons=1e-4,
                            init_W=init_W,
                            init_b=init_b)

write_param = FullyConnectedLayer(name='write_param',
                                  parent=['dec'],
                                  parent_dim=[rnn_dim],
                                  nout=5,
                                  unit='linear',
                                  init_W=init_W,
                                  init_b=init_b)

write = WriteLayer(name='write',
                   parent=['w', 'write_param'],
                   glimpse_shape=write_shape,
                   input_shape=input_shape)

canvas_mu = CanvasLayer(name='canvas_mu',
                        parent=['write'],
                        nout=inpsz,
                        batch_size=batch_size)

canvas_sig = CanvasLayer(name='canvas_sig',
                         parent=['write_sig'],
                         nout=inpsz,
                         init_state_cons=1e-4,
                         batch_size=batch_size)

sampling = GaussianLayer(name='sampling',
                         parent=['canvas_mu_tm1', 'canvas_sig_tm1'],
                         use_sample=1)

nodes = [error, read_param, read, enc, phi_mu, phi_sig, prior, kl, dec, w, w_sig, write_param, write, canvas_mu, canvas_sig, sampling]
for node in nodes:
    node.initialize()
params = flatten([node.get_params().values() for node in nodes])

def inner_fn(enc_tm1, dec_tm1, canvas_mu_tm1, canvas_sig_tm1, x):

    sampling_t = sampling.fprop([canvas_mu_tm1, canvas_sig_tm1])
    error_t = error.fprop([[x], [sampling_t]])

    read_param_t = read_param.fprop([dec_tm1])
    read_t = read.fprop([x, error_t, read_param_t])

    enc_t = enc.fprop([[read_t], [enc_tm1, dec_tm1]])

    phi_mu_t = phi_mu.fprop([enc_t])
    phi_sig_t = phi_sig.fprop([enc_t])

    prior_t = prior.fprop([phi_mu_t, phi_sig_t])
    kl_t = kl.fprop([phi_mu_t, phi_sig_t])

    dec_t = dec.fprop([[prior_t], [dec_tm1]])

    w_t = w.fprop([dec_t])
    w_sig_t = w_sig.fprop([dec_t])

    write_param_t = write_param.fprop([dec_t])
    write_t = write.fprop([w_t, write_param_t])
    write_sig_t = write.fprop([w_sig_t, write_param_t])

    canvas_mu_t = canvas_mu.fprop([[write_t], [canvas_mu_tm1]])
    canvas_sig_t = canvas_sig.fprop([[write_sig_t], [canvas_sig_tm1]])

    return enc_t, dec_t, canvas_mu_t, canvas_sig_t, kl_t, phi_sig_t

((enc_t, dec_t, canvas_mu_t, canvas_sig_t, kl_t, phi_sig_t), updates) =\
    theano.scan(fn=inner_fn,
                outputs_info=[enc.get_init_state(),
                              dec.get_init_state(),
                              canvas_mu.get_init_state(),
                              canvas_sig.get_init_state(),
                              None, None],
                non_sequences=[x],
                n_steps=n_steps)
for k, v in updates.iteritems():
    k.default_update = v

canvas_mu_T = canvas_mu_t[-1]
canvas_sig_T = canvas_sig_t[-1]
recon_term = Gaussian(x, canvas_mu_T, canvas_sig_T).mean()
kl_term = kl_t.sum(axis=0).mean()
cost = recon_term + kl_term
cost.name = 'cost'
recon_term.name = 'recon_term'
kl_term.name = 'kl_term'
recon_err = ((x - canvas_mu_T)**2).mean() / x.std()
recon_err.name = 'recon_err'

# monitoring attributes
max_phi_sig = phi_sig_t.max()
mean_phi_sig = phi_sig_t.mean()
min_phi_sig = phi_sig_t.min()
max_phi_sig.name = 'max_phi_sig'
mean_phi_sig.name = 'mean_phi_sig'
min_phi_sig.name = 'min_phi_sig'

max_canvas_sig = canvas_sig_T.max()
mean_canvas_sig = canvas_sig_T.mean()
min_canvas_sig = canvas_sig_T.min()
max_canvas_sig.name = 'max_canvas_sig'
mean_canvas_sig.name = 'mean_canvas_sig'
min_canvas_sig.name = 'min_canvas_sig'

max_canvas_mu = canvas_mu_T.max()
mean_canvas_mu = canvas_mu_T.mean()
min_canvas_mu = canvas_mu_T.min()
max_canvas_mu.name = 'max_canvas_mu'
mean_canvas_mu.name = 'mean_canvas_mu'
min_canvas_mu.name = 'min_canvas_mu'

model.inputs = [x]
model._params = params
model.nodes = nodes

optimizer = Adam(
    lr=0.001
)

extension = [
    GradientClipping(batch_size=batch_size),
    EpochCount(10000),
    Monitoring(freq=10,
               ddout=[cost, recon_term, kl_term, recon_err,
                      max_phi_sig, mean_phi_sig, min_phi_sig,
                      max_canvas_sig, mean_canvas_sig, min_canvas_sig,
                      max_canvas_mu, mean_canvas_mu, min_canvas_mu],
               data=[Iterator(testdata, batch_size)]),
    Picklize(freq=2000,
             path=savepath),
    EarlyStopping(freq=500, path=savepath)
]

mainloop = Training(
    name='draw',
    data=Iterator(trdata, batch_size),
    model=model,
    optimizer=optimizer,
    cost=cost,
    outputs=[cost],
    extension=extension
)
mainloop.run()
