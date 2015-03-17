import ipdb
import numpy as np
import theano.tensor as T

from cle.cle.cost import NllBin
from cle.cle.data import Iterator
from cle.cle.graph.net import Net
from cle.cle.models import Model
from cle.cle.layers import InitCell
from cle.cle.layers.feedforward import FullyConnectedLayer
from cle.cle.layers.conv import ConvertLayer, Conv2DLayer
from cle.cle.layers.layer import MaxPool2D
from cle.cle.train import Training
from cle.cle.train.ext import (
    EpochCount,
    GradientClipping,
    Monitoring,
    Picklize
)
from cle.cle.train.opt import Adam, RMSProp, Momentum
from cle.cle.utils import error, predict
from courseworks.ift6266h15.datasets.dogs_and_cats import DogsnCats


datapath = '/home/junyoung/data/dogs_and_cats/'
savepath = '/home/junyoung/repos/courseworks/ift6266h15/saved/'

batch_size = 128
inpsz = 6912
debug = 0

model = Model()
trdata = DogsnCats(name='train',
                   path=datapath,
                   use_color=1)
valdata = DogsnCats(name='valid',
                    path=datapath,
                    use_color=1,
                    X_mean=trdata.X_mean,
                    X_std=trdata.X_std)
testdata = DogsnCats(name='test',
                     path=datapath,
                     use_color=1,
                     X_mean=trdata.X_mean,
                     X_std=trdata.X_std)

init_W = InitCell('rand')
init_b = InitCell('zeros')

model.inputs = trdata.theano_vars()
x, y = model.inputs
if debug:
    x.tag.test_value = np.zeros((batch_size, inpsz), dtype=np.float32)
    y.tag.test_value = np.zeros((batch_size, 1), dtype=np.float32)

inputs = [x, y]
inputs_dim = {'x':inpsz, 'y':1}
c1 = ConvertLayer(name='c1',
                  parent=['x'],
                  outshape=(batch_size, 3, 48, 48))
h1 = Conv2DLayer(name='h1',
                 parent=['c1'],
                 outshape=(batch_size, 32, 40, 40),
                 unit='relu',
                 init_W=init_W,
                 init_b=init_b)
p1 = MaxPool2D(name='p1',
               parent=['h1'],
               poolsize=(3, 3),
               poolstride=(2, 2))
h2 = Conv2DLayer(name='h2',
                 parent=['p1'],
                 outshape=(batch_size, 64, 16, 16),
                 unit='relu',
                 init_W=init_W,
                 init_b=init_b)
p2 = MaxPool2D(name='p2',
               parent=['h2'],
               poolsize=(3, 3),
               poolstride=(2, 2))
h3 = Conv2DLayer(name='h3',
                 parent=['p2'],
                 outshape=(batch_size, 64, 4, 4),
                 unit='relu',
                 init_W=init_W,
                 init_b=init_b)
p3 = MaxPool2D(name='p3',
               parent=['h3'],
               poolsize=(3, 3),
               poolstride=(2, 2))
c2 = ConvertLayer(name='c2',
                  parent=['p3'],
                  outshape=(batch_size, 256))
h4 = FullyConnectedLayer(name='h4',
                         parent=['c2'],
                         nout=256,
                         unit='relu',
                         init_W=init_W,
                         init_b=init_b)
h5 = FullyConnectedLayer(name='h5',
                         parent=['h4'],
                         nout=256,
                         unit='relu',
                         init_W=init_W,
                         init_b=init_b)
h6 = FullyConnectedLayer(name='h6',
                         parent=['h5'],
                         nout=1,
                         unit='sigmoid',
                         init_W=init_W,
                         init_b=init_b)
nodes = [c1, c2, h1, h2, h3, h4, h5, h6, p1, p2, p3]

cnn = Net(inputs=inputs, inputs_dim=inputs_dim, nodes=nodes)
cnn.build_graph()

cost = NllBin(y, cnn.nodes['h6'].out).mean()
err = error(T.ge(cnn.nodes['h6'].out, 0.5), y)
cost.name = 'cost'
err.name = 'error_rate'
model.graphs = [cnn]

optimizer = Adam(
    lr=0.00005,
)

extension = [
    GradientClipping(batch_size=batch_size),
    EpochCount(100),
    Monitoring(freq=10,
               ddout=[cost, err],
               data=[Iterator(valdata, batch_size),
                     Iterator(testdata, batch_size)]),
    Picklize(freq=1,
             path=savepath)
]

mainloop = Training(
    name='cnn',
    data=Iterator(trdata, batch_size),
    model=model,
    optimizer=optimizer,
    cost=cost,
    outputs=[cost, err],
    extension=extension
)
mainloop.run()
