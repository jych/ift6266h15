import ipdb
import fnmatch
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import random
import tables
import tarfile
import theano.tensor as T
import zipfile

from cle.cle.data import DesignMatrix, TemporalSeries
from cle.cle.data.prep import StaticPrepMixin
from cle.cle.utils import segment_axis, tolist, totuple

from scipy.misc import imresize


class DogsnCats(DesignMatrix, StaticPrepMixin):
    """
    Dogs and Cats dataset batch provider

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, use_color=0, prep='normalize', X_mean=None,
                 X_std=None, unsupervised=0, quadrisect=0, **kwargs):
        self.prep = prep
        self.use_color = use_color
        self.X_mean = X_mean
        self.X_std = X_std
        self.unsupervised = unsupervised
        self.quadrisect = quadrisect
        super(DogsnCats, self).__init__(**kwargs)

    def load(self, data_path):
        if self.use_color:
            return self.load_color(data_path)
        else:
            return self.load_gray(data_path)

    def load_color(self, data_path, random_seed=123522):
        dataset = 'train.zip'
        data_file = os.path.join(data_path, dataset)
        if os.path.isfile(data_file):
            dataset = data_file

        if (not os.path.isfile(data_file)):
            try:
                import urllib
                urllib.urlretrieve('http://google.com')
            except AttributeError:
                import urllib.request as urllib
            url = 'https://dl.dropboxusercontent.com/u/15378192/train.zip'
            print('Downloading data from %s' % url)
            urllib.urlretrieve(url, data_file)

        data_dir = os.path.join(data_path, 'cvd')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            zf = zipfile.ZipFile(data_file)
            zf.extractall(data_dir)

        data_file = os.path.join(data_path, 'cvd_color.hdf5')
        label_file = os.path.join(data_path, 'cvd_color_labels.npy')
        if not os.path.exists(data_file):
            print('... loading data')
            cat_matches = []
            dog_matches = []
            for root, dirname, filenames in os.walk(data_dir):
                for filename in fnmatch.filter(filenames, 'cat*'):
                    cat_matches.append(os.path.join(root, filename))
                for filename in fnmatch.filter(filenames, 'dog*'):
                    dog_matches.append(os.path.join(root, filename))

            sort_key = lambda x: int(x.split('.')[-2])
            cat_matches = sorted(cat_matches, key=sort_key)
            dog_matches = sorted(dog_matches, key=sort_key)

            def square(X):
                resize_shape = (40, 40)
                slice_size = (32, 32)
                slice_left = (resize_shape[0] - slice_size[0]) / 2
                slice_upper = (resize_shape[1] - slice_size[1]) / 2
                return imresize(X, resize_shape, interp='nearest')[
                    slice_left:slice_left + slice_size[0],
                    slice_upper:slice_upper + slice_size[1]].transpose(
                        2, 0, 1).astype('float32')

            matches = cat_matches + dog_matches
            matches = np.array(matches)
            random_state = np.random.RandomState(random_seed)
            idx = random_state.permutation(len(matches))
            c = [0] * len(cat_matches)
            d = [1] * len(dog_matches)
            y = np.array(c + d).astype('float32')
            matches = matches[idx]
            y = y[idx]

            compression_filter = tables.Filters(complevel=5, complib='blosc')
            h5_file = tables.openFile(data_file, mode='w')
            example = square(mpimg.imread(matches[0]))
            image_storage = h5_file.createEArray(h5_file.root, 'images',
                                                 tables.Float32Atom(),
                                                 shape=(0,) + example.shape,
                                                 filters=compression_filter)
            for n, f in enumerate(matches):
                print("Processing image %i of %i" % (n, len(matches)))
                X = square(mpimg.imread(f)).astype('float32')
                image_storage.append(X[None])
            h5_file.close()
            np.save(label_file, y)
        h5_file = tables.openFile(data_file, mode='r')
        X_s = h5_file.root.images
        y_s = np.load(label_file)

        ntrain = 20000
        nvalid = 2500
        ntest = 2500

        train_x = X_s[:ntrain]
        valid_x = X_s[ntrain:ntrain+nvalid]
        test_x = X_s[ntrain+nvalid:]
        train_y = y_s[:ntrain]
        valid_y = y_s[ntrain:ntrain+nvalid]
        test_y = y_s[ntrain+nvalid:]
        test_x = test_x.astype('float32').reshape(ntest, -1)
        test_y = test_y.astype('float32')[:, None]
        valid_x = valid_x.astype('float32').reshape(nvalid, -1)
        valid_y = valid_y.astype('float32')[:, None]
        train_x = train_x.astype('float32').reshape(ntrain, -1)
        train_y = train_y.astype('float32')[:, None]

        if self.name == 'train':
            if self.prep == 'normalize':
                train_x, self.X_mean, self.X_std =\
                    self.normalize(train_x, self.X_mean, self.X_std, 0)
            elif self.prep == 'global_normalize':
                train_x, self.X_mean, self.X_std =\
                    self.global_normalize(train_x, self.X_mean, self.X_std)
            if self.quadrisect:
                train_x = train_x.reshape((ntrain, 3, 32, 32))
                A = train_x[:, :, :16, :16].reshape(ntrain, -1)
                B = train_x[:, :, :16, 16:].reshape(ntrain, -1)
                C = train_x[:, :, 16:, :16].reshape(ntrain, -1)
                D = train_x[:, :, 16:, 16:].reshape(ntrain, -1)
                E = np.concatenate([A, B, C, D], axis=1)
                train_x = E.reshape((ntrain, 4, -1))
            if self.unsupervised:
                return [train_x]
            else:
                return [train_x, train_y]
        elif self.name == 'valid':
            if self.prep == 'normalize':
                valid_x, self.X_mean, self.X_std =\
                    self.normalize(valid_x, self.X_mean, self.X_std, 0)
            elif self.prep == 'global_normalize':
                valid_x, self.X_mean, self.X_std =\
                    self.global_normalize(valid_x, self.X_mean, self.X_std)
            if self.quadrisect:
                valid_x = valid_x.reshape((nvalid, 3, 32, 32))
                A = valid_x[:, :, :16, :16].reshape(nvalid, -1)
                B = valid_x[:, :, :16, 16:].reshape(nvalid, -1)
                C = valid_x[:, :, 16:, :16].reshape(nvalid, -1)
                D = valid_x[:, :, 16:, 16:].reshape(nvalid, -1)
                E = np.concatenate([A, B, C, D], axis=1)
                valid_x = E.reshape((nvalid, 4, -1))
            if self.unsupervised:
                return [valid_x]
            else:
                return [valid_x, valid_y]
        elif self.name == 'test':
            if self.prep == 'normalize':
                test_x, self.X_mean, self.X_std =\
                    self.normalize(test_x, self.X_mean, self.X_std, 0)
            elif self.prep == 'global_normalize':
                test_x, self.X_mean, self.X_std =\
                    self.global_normalize(test_x, self.X_mean, self.X_std)
            if self.quadrisect:
                test_x = test_x.reshape((ntest, 3, 32, 32))
                A = test_x[:, :, :16, :16].reshape(ntest, -1)
                B = test_x[:, :, :16, 16:].reshape(ntest, -1)
                C = test_x[:, :, 16:, :16].reshape(ntest, -1)
                D = test_x[:, :, 16:, 16:].reshape(ntest, -1)
                E = np.concatenate([A, B, C, D], axis=1)
                test_x = E.reshape((ntest, 4, -1))
            if self.unsupervised:
                return [test_x]
            else:
                return [test_x, test_y]

    def load_gray(self, data_path, random_seed=123522):
        dataset = 'train.zip'
        data_file = os.path.join(data_path, dataset)
        if os.path.isfile(data_file):
            dataset = data_file

        if (not os.path.isfile(data_file)):
            try:
                import urllib
                urllib.urlretrieve('http://google.com')
            except AttributeError:
                import urllib.request as urllib
            url = 'https://dl.dropboxusercontent.com/u/15378192/train.zip'
            print('Downloading data from %s' % url)
            urllib.urlretrieve(url, data_file)

        data_dir = os.path.join(data_path, 'cvd')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            zf = zipfile.ZipFile(data_file)
            zf.extractall(data_dir)

        data_file = os.path.join(data_path, 'cvd_gray.npy')
        label_file = os.path.join(data_path, 'cvd_gray_labels.npy')
        if not os.path.exists(data_file):
            print('... loading data')
            cat_matches = []
            dog_matches = []
            for root, dirname, filenames in os.walk(data_dir):
                for filename in fnmatch.filter(filenames, 'cat*'):
                    cat_matches.append(os.path.join(root, filename))
                for filename in fnmatch.filter(filenames, 'dog*'):
                    dog_matches.append(os.path.join(root, filename))

            sort_key = lambda x: int(x.split('.')[-2])
            cat_matches = sorted(cat_matches, key=sort_key)
            dog_matches = sorted(dog_matches, key=sort_key)

            def square_and_gray(X):
                # From Roland
                gray_consts = np.array([[0.299], [0.587], [0.114]])
                return imresize(X, (32, 32)).dot(gray_consts).squeeze()

            X_cat = np.asarray([square_and_gray(mpimg.imread(f))
                                for f in cat_matches])
            y_cat = np.zeros((len(X_cat),))
            X_dog = np.asarray([square_and_gray(mpimg.imread(f))
                                for f in dog_matches])
            y_dog = np.ones((len(X_dog),))
            X = np.concatenate((X_cat, X_dog), axis=0).astype('float32')
            y = np.concatenate((y_cat, y_dog), axis=0).astype('float32')
            np.save(data_file, X)
            np.save(label_file, y)
        else:
            X = np.load(data_file)
            y = np.load(label_file)

        random_state = np.random.RandomState(random_seed)
        idx = random_state.permutation(len(X))
        X_s = X[idx].reshape(len(X), -1)
        y_s = y[idx]

        ntrain = 20000
        nvalid = 2500
        ntest = 2500

        train_x = X_s[:ntrain]
        valid_x = X_s[ntrain:ntrain+nvalid]
        test_x = X_s[ntrain+nvalid:]
        train_y = y_s[:ntrain]
        valid_y = y_s[ntrain:ntrain+nvalid]
        test_y = y_s[ntrain+nvalid:]
        test_x = test_x.astype('float32').reshape(ntest, -1)
        test_y = test_y.astype('float32')[:, None]
        valid_x = valid_x.astype('float32').reshape(nvalid, -1)
        valid_y = valid_y.astype('float32')[:, None]
        train_x = train_x.astype('float32').reshape(ntrain, -1)
        train_y = train_y.astype('float32')[:, None]

        if self.name == 'train':
            if self.prep == 'normalize':
                train_x, self.X_mean, self.X_std =\
                    self.normalize(train_x, self.X_mean, self.X_std, 0)
            elif self.prep == 'global_normalize':
                train_x, self.X_mean, self.X_std =\
                    self.global_normalize(train_x, self.X_mean, self.X_std)
            if self.quadrisect:
                train_x = train_x.reshape((ntrain, 32, 32))
                A = train_x[:, :16, :16].reshape(ntrain, -1)
                B = train_x[:, :16, 16:].reshape(ntrain, -1)
                C = train_x[:, 16:, :16].reshape(ntrain, -1)
                D = train_x[:, 16:, 16:].reshape(ntrain, -1)
                E = np.concatenate([A, B, C, D], axis=1)
                train_x = E.reshape((ntrain, 4, -1))
            if self.unsupervised:
                return [train_x]
            else:
                return [train_x, train_y]
        elif self.name == 'valid':
            if self.prep == 'normalize':
                valid_x, self.X_mean, self.X_std =\
                    self.normalize(valid_x, self.X_mean, self.X_std, 0)
            elif self.prep == 'global_normalize':
                valid_x, self.X_mean, self.X_std =\
                    self.global_normalize(valid_x, self.X_mean, self.X_std)
            if self.quadrisect:
                valid_x = valid_x.reshape((nvalid, 32, 32))
                A = valid_x[:, :16, :16].reshape(nvalid, -1)
                B = valid_x[:, :16, 16:].reshape(nvalid, -1)
                C = valid_x[:, 16:, :16].reshape(nvalid, -1)
                D = valid_x[:, 16:, 16:].reshape(nvalid, -1)
                E = np.concatenate([A, B, C, D], axis=1)
                valid_x = E.reshape((nvalid, 4, -1))
            if self.unsupervised:
                return [valid_x]
            else:
                return [valid_x, valid_y]
        elif self.name == 'test':
            if self.prep == 'normalize':
                test_x, self.X_mean, self.X_std =\
                    self.normalize(test_x, self.X_mean, self.X_std, 0)
            elif self.prep == 'global_normalize':
                test_x, self.X_mean, self.X_std =\
                    self.global_normalize(test_x, self.X_mean, self.X_std)
            if self.quadrisect:
                test_x = test_x.reshape((ntest, 32, 32))
                A = test_x[:, :16, :16].reshape(ntest, -1)
                B = test_x[:, :16, 16:].reshape(ntest, -1)
                C = test_x[:, 16:, :16].reshape(ntest, -1)
                D = test_x[:, 16:, 16:].reshape(ntest, -1)
                E = np.concatenate([A, B, C, D], axis=1)
                test_x = E.reshape((ntest, 4, -1))
            if self.unsupervised:
                return [test_x]
            else:
                return [test_x, test_y]

    def theano_vars(self):
        if self.unsupervised:
            if self.quadrisect:
                return T.ftensor3('x')
            else:
                return T.fmatrix('x')
        else:
            if self.quadrisect:
                return [T.ftensor3('x'), T.fmatrix('y')]
            else:
                return [T.fmatrix('x'), T.fmatrix('y')]


if __name__ == "__main__":
    datapath = '/home/chungjun/data/dogs_and_cats/'
    use_color = 1
    data = DogsnCats(name='train',
                     path=datapath,
                     use_color=use_color,
                     prep='global_normalize')
    ipdb.set_trace()
