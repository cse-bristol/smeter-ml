from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Input, concatenate
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from ml.model.loss import make_qd_loss_fn


def base_cnn(input_shape, loss_fn=None):
    """
    Produce a basic CNN, with the architecture we determined to be pretty good in Phase 1.
    Note: inputs should be channels *last*, i.e. the shape should be (<no. time series entries>, <no. channels>).
    """
    if loss_fn is None:
        loss_fn = make_qd_loss_fn(lam=100., s=0.0104)
    rmse = RootMeanSquaredError()

    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation="relu", input_shape=input_shape))
    model.add(Conv1D(32, kernel_size=3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(2, activation="linear"))
    adam = Adam(learning_rate=0.001)
    model.compile(loss=loss_fn, optimizer=adam, metrics=[rmse])

    return model


def base_cnn_single_output(input_shape):
    """
    Same as base_cnn above except that there is only one output feature (the HTC)
    as opposed to two (the upper and lower bounds).
    This is useful for doing experiments where you only care about the predicted value itself,
    not the prediction interval.
    """
    rmse = RootMeanSquaredError()
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation="relu", input_shape=input_shape))
    model.add(Conv1D(32, kernel_size=3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(loss="mse", optimizer="adam", metrics=[rmse])

    return model


def _base_cnn(input_shape):
    """
    Produce a basic CNN using the Keras functional API so that we can merge it with other
    networks later, if we so wish.
    """
    input = Input(shape=input_shape, name="timeseries")
    model = Conv1D(32, kernel_size=3, activation="relu")(input)
    model = Conv1D(32, kernel_size=3, activation="relu")(model)
    model = Flatten()(model)
    model = Dense(128, activation="relu")(model)
    output = Dense(1, activation="linear")(model)

    return input, output


def base_cnn_with_static_features(cnn_input_shape, static_features_input_shape, loss_fn=None):
    """
    Produce a keras model which is the combination of the CNN with some static features,
    with a couple of MLP layers at the end.
    (At the moment there is only one static feature, floor area).
    """
    if loss_fn is None:
        loss_fn = make_qd_loss_fn(lam=100., s=0.0104)
    rmse = RootMeanSquaredError()

    cnn_input, cnn_output = _base_cnn(cnn_input_shape)
    static_features_input = Input(shape=static_features_input_shape, name="static_features")
    concat = concatenate([cnn_output, static_features_input])
    # Add an extra MLP layer or two for combining the cnn output and static features
    mlp1 = Dense(32, activation="relu")(concat)
    mlp2 = Dense(32, activation="relu")(mlp1)
    # Output layer
    output = Dense(2, activation="linear", name="output")(mlp2)

    model = Model(inputs=[cnn_input, static_features_input], outputs=output)
    model.compile(loss=loss_fn, optimizer="adam", metrics=[rmse])

    return model


def base_cnn_with_static_features_and_single_output(cnn_input_shape, static_features_input_shape):
    """
    Produce a keras model which is the combination of the CNN with some static features,
    with a couple of MLP layers at the end, outputting just one feature (the HTC).
    (At the moment there is only one static feature, floor area).
    """
    rmse = RootMeanSquaredError()

    cnn_input, cnn_output = _base_cnn(cnn_input_shape)
    static_features_input = Input(shape=static_features_input_shape, name="static_features")
    concat = concatenate([cnn_output, static_features_input])
    # Add an extra MLP layer or two for combining the cnn output and static features
    mlp1 = Dense(32, activation="relu")(concat)
    mlp2 = Dense(32, activation="relu")(mlp1)
    # Output layer
    output = Dense(1, activation="linear", name="output")(mlp2)

    model = Model(inputs=[cnn_input, static_features_input], outputs=output)
    model.compile(loss="mse", optimizer="adam", metrics=[rmse])

    return model
