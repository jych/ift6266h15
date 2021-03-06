import ipdb
import numpy as np
import theano
import theano.tensor as T

from cle.cle.cost import NllBin
from cle.cle.data import Iterator
from cle.cle.graph.net import Net
from cle.cle.models import Model
from cle.cle.layers import InitCell
from cle.cle.layers.feedforward import FullyConnectedLayer
from cle.cle.layers.conv import ConvertLayer, Conv2DLayer
from cle.cle.layers.layer import MaxPool2D, DropoutLayer
from cle.cle.train import Training
from cle.cle.train.ext import (
    EpochCount,
    GradientClipping,
    Monitoring,
    Picklize,
    WeightNorm
)
from cle.cle.train.opt import Adam
from cle.cle.utils import error, flatten, predict
from course_works.ift6266h15.datasets.dogs_and_cats import DogsnCats


datapath = '/home/junyoung/data/dogs_and_cats/'
savepath = '/home/junyoung/repos/course_works/ift6266h15/saved/'

batch_size = 128
debug = 1

model = Model()
trdata = DogsnCats(name='train',
                   path=datapath,
                   use_color=1)
testdata = DogsnCats(name='test',
                     path=datapath,
                     use_color=1,
                     X_mean=trdata.X_mean,
                     X_std=trdata.X_std)

init_W = InitCell('rand', low=-0.08, high=0.08)
init_b = InitCell('zeros')

x, y = trdata.theano_vars()
if debug:
    x.tag.test_value = np.zeros((batch_size, 3*48*48), dtype=np.float32)
    y.tag.test_value = np.zeros((batch_size, 1), dtype=np.float32)

c1 = ConvertLayer(name='c1',
                  parent=['x'],
                  outshape=(batch_size, 3, 32, 32))
                  #outshape=(batch_size, 3, 48, 48))
h1 = Conv2DLayer(name='h1',
                 parent=['c1'],
                 #parshape=[(batch_size, 3, 48, 48)],
                 #outshape=(batch_size, 64, 46, 46),
                 parshape=[(batch_size, 3, 32, 32)],
                 outshape=(batch_size, 64, 30, 30),
                 unit='relu',
                 init_W=init_W,
                 init_b=init_b)
p1 = MaxPool2D(name='p1',
               parent=['h1'],
               pool_size=(2, 2),
               pool_stride=(2, 2),
               set_shape=0)
h2 = Conv2DLayer(name='h2',
                 parent=['p1'],
                 #parshape=[(batch_size, 64, 23, 23)],
                 #outshape=(batch_size, 128, 20, 20),
                 parshape=[(batch_size, 64, 15, 15)],
                 outshape=(batch_size, 128, 12, 12),
                 unit='relu',
                 init_W=init_W,
                 init_b=init_b)
p2 = MaxPool2D(name='p2',
               parent=['h2'],
               pool_size=(2, 2),
               pool_stride=(2, 2),
               set_shape=0)
h3 = Conv2DLayer(name='h3',
                 parent=['p2'],
                 #parshape=[(batch_size, 128, 10, 10)],
                 #outshape=(batch_size, 128, 8, 8),
                 parshape=[(batch_size, 128, 6, 6)],
                 outshape=(batch_size, 128, 4, 4),
                 unit='relu',
                 init_W=init_W,
                 init_b=init_b)
p3 = MaxPool2D(name='p3',
               parent=['h3'],
               pool_size=(2, 2),
               pool_stride=(2, 2),
               set_shape=0)
h4 = Conv2DLayer(name='h4',
                 parent=['p3'],
                 #parshape=[(batch_size, 128, 4, 4)],
                 #outshape=(batch_size, 128, 2, 2),
                 parshape=[(batch_size, 128, 2, 2)],
                 outshape=(batch_size, 128, 1, 1),
                 unit='relu',
                 init_W=init_W,
                 init_b=init_b)
#p4 = MaxPool2D(name='p4',
#               parent=['h4'],
#               pool_size=(2, 2),
#               pool_stride=(2, 2),
#               set_shape=0)
c2 = ConvertLayer(name='c2',
                  parent=['p4'],
                  outshape=(batch_size, 128))
h5 = FullyConnectedLayer(name='h5',
                         parent=['c2'],
                         parent_dim=[128],
                         nout=512,
                         unit='relu',
                         init_W=init_W,
                         init_b=init_b)
d1 = DropoutLayer(name='d1', parent=['h5'], nout=512)
h6 = FullyConnectedLayer(name='h6',
                         parent=['d1'],
                         parent_dim=[512],
                         nout=512,
                         unit='relu',
                         init_W=init_W,
                         init_b=init_b)
d2 = DropoutLayer(name='d2', parent=['h6'], nout=512)
h7 = FullyConnectedLayer(name='h7',
                         parent=['d2'],
                         parent_dim=[512],
                         nout=1,
                         unit='sigmoid',
                         init_W=init_W,
                         init_b=init_b)

#nodes = [c1, h1, p1, h2, p2, h3, p3, c2, h4, p4, c2, h5, d1, h6, d2, h7]
nodes = [c1, h1, p1, h2, p2, h3, p3, c2, h4, c2, h5, d1, h6, d2, h7]
for node in nodes:
    node.initialize()
params = flatten([node.get_params().values() for node in nodes])

c1_out = c1.fprop([x])
h1_out = h1.fprop([c1_out])
p1_out = p1.fprop([h1_out])
h2_out = h2.fprop([p1_out])
p2_out = p2.fprop([h2_out])
h3_out = h3.fprop([p2_out])
p3_out = p3.fprop([h3_out])
h4_out = h4.fprop([p3_out])
#p4_out = p4.fprop([h4_out])
#c2_out = c2.fprop([p4_out])
c2_out = c2.fprop([h4_out])
h5_out = h5.fprop([c2_out])
d1_out = d1.fprop([h5_out])
h6_out = h6.fprop([d1_out])
d2_out = d2.fprop([h6_out])
h7_out = h7.fprop([d2_out])
cost = NllBin(y, h7_out).mean()
err = error(T.sum(h7_out > 0.5, axis=1), T.sum(y > 0.5, axis=1))
cost.name = 'cost'
err.name = 'error_rate'
model.inputs = [x, y]
model._params = params
model.nodes = nodes

c1_out = c1.fprop([x])
h1_out = h1.fprop([c1_out])
p1_out = p1.fprop([h1_out])
h2_out = h2.fprop([p1_out])
p2_out = p2.fprop([h2_out])
h3_out = h3.fprop([p2_out])
p3_out = p3.fprop([h3_out])
h4_out = h4.fprop([p3_out])
#p4_out = p4.fprop([h4_out])
#c2_out = c2.fprop([p4_out])
c2_out = c2.fprop([h4_out])
h5_out = h5.fprop([c2_out])
h6_out = h6.fprop([h5_out])
h7_out = h7.fprop([h6_out])
mn_cost = NllBin(y, h7_out).mean()
mn_err = error(T.sum(h7_out > 0.5, axis=1), T.sum(y > 0.5, axis=1))
mn_cost.name = 'cost'
mn_err.name = 'error_rate'
monitor_fn = theano.function([x, y], [mn_cost, mn_err])

optimizer = Adam(
    lr=0.001
)

extension = [
    GradientClipping(batch_size=batch_size),
    EpochCount(100),
    Monitoring(freq=100,
               ddout=[cost, err],
               data=[Iterator(testdata, batch_size)]),
    Picklize(freq=10000,
             path=savepath),
    WeightNorm(param_name='W')
]

mainloop = Training(
    name='dogcat',
    data=Iterator(trdata, batch_size),
    model=model,
    optimizer=optimizer,
    cost=cost,
    outputs=[cost, err],
    extension=extension
)
mainloop.run()
