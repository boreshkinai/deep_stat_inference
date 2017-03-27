import numpy as np
import numpy.random as nprnd
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from keras.layers import Input, merge
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization

N = 100
M = 20000
PERCENTILE_RANGE = np.arange(0, 101)
NN_WIDTH = 2*N

def myGenerator():
    sigma = 10.0 * nprnd.rand()
    print("\nTraining sigma=" + str(sigma))

    Z1 = nprnd.normal(0, sigma, (N, M))
    Z2 = nprnd.normal(0, sigma, (N, M))

    # H0 model parameter estimation under H0:
    muH0 = np.mean(np.concatenate([Z1, Z2], axis=0), axis=0)
    sigmaH0 = np.sqrt(np.var(np.concatenate([Z1, Z2], axis=0), axis=0))
    # H1 model parameter estimation under H0:
    muZ1H1 = np.mean(Z1, axis=0)
    muZ2H1 = np.mean(Z2, axis=0)
    sigmaZ1H1 = np.sqrt(np.var(Z1, axis=0))
    sigmaZ2H1 = np.sqrt(np.var(Z2, axis=0))

    TZ = np.sum(
        stats.norm.logpdf(Z1, loc=muZ1H1, scale=sigmaZ1H1) + stats.norm.logpdf(Z2, loc=muZ2H1, scale=sigmaZ2H1)
        - stats.norm.logpdf(Z1, loc=muH0, scale=sigmaH0) - stats.norm.logpdf(Z2, loc=muH0, scale=sigmaH0), axis=0)
    percentileH0 = np.percentile(TZ, PERCENTILE_RANGE)
    ecdfH0 = interp1d(percentileH0, PERCENTILE_RANGE, fill_value=(0.0, 100.0), bounds_error=False)
    p1 = np.sum(ecdfH0(TZ) <= 5) / M

    while 1:

        # for mu in mu_vec:
        #    print("\nTraining mu=" + str(mu))

        mu_vec = 1 * sigma * nprnd.rand(1, M) * (nprnd.rand(1, M) > 0.5)

        X = nprnd.normal(0, sigma, (N, M))
        Y = nprnd.normal(0, sigma, (N, M)) + mu_vec

        # H0 model parameter estimation under H0:
        muXYH0 = np.mean(np.concatenate([X, Y], axis=0), axis=0)
        sigmaXYH0 = np.sqrt(np.var(np.concatenate([X, Y], axis=0), axis=0))
        # H1 model parameter estimation under H0:
        muXH1 = np.mean(X, axis=0)
        muYH1 = np.mean(Y, axis=0)
        sigmaXH1 = np.sqrt(np.var(X, axis=0))
        sigmaYH1 = np.sqrt(np.var(Y, axis=0))

        TXY = np.sum(
            stats.norm.logpdf(X, loc=muXH1, scale=sigmaXH1) + stats.norm.logpdf(Y, loc=muYH1, scale=sigmaYH1) -
            stats.norm.logpdf(X, loc=muXYH0, scale=sigmaXYH0) - stats.norm.logpdf(Y, loc=muXYH0, scale=sigmaXYH0),
            axis=0)
        pValues = 100 - ecdfH0(TXY)

        Xtrain = np.concatenate((X.transpose(), Y.transpose()), axis=1)
        Xtrain -= Xtrain.min(axis=1).reshape((Xtrain.shape[0],1))
        Xtrain /= Xtrain.max(axis=1).reshape((Xtrain.shape[0],1))
        yield (Xtrain, pValues.transpose())


def get_nn_model(input_size):

    main_input = Input(shape=(input_size,), name='main_input')

    x = Dense(10 * input_size, activation='relu')(main_input)
    x = Dense(NN_WIDTH, activation='relu')(x)
    x = Dense(NN_WIDTH, activation='relu')(x)
    x = Dense(NN_WIDTH, activation='relu')(x)

    x = Dense(NN_WIDTH, activation='relu')(x)
    x = Dense(NN_WIDTH, activation='relu')(x)

    x = Dense(1)(x)
    main_output = Activation(activation='relu', name = 'main_output')(x)

    model = Model(input=[main_input], output=[main_output])
    model.compile(optimizer='adam',
                  loss={'main_output': 'mean_squared_error'},
                  metrics={'main_output': "mean_squared_error"})
    return model


gaussianGenerator = myGenerator()
model = get_nn_model(2*N)

# model.fit_generator(gaussianGenerator, steps_per_epoch=10, epochs=10, verbose=1, callbacks=None, validation_data=None, workers=1)

model.fit_generator(gaussianGenerator, samples_per_epoch=M, nb_epoch=1000, verbose=1, nb_worker=1)

X = gaussianGenerator.__next__()
metrics = model.evaluate(x=X[0], y=X[1])

yPred = model.predict(X[0])
a=1;

plt.plot(X[1])
plt.plot(yPred)
plt.show()