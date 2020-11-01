"""Module for generating an ensemble model from a given config file.

Exports a single class, Ensemble, which provides methods for training the ensemble,
saving the individual model files to disk and making predictions.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as k
from typing import Callable
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session
from ml.model.cnn import (
    base_cnn,
    base_cnn_with_static_features,
    base_cnn_single_output,
    base_cnn_with_static_features_and_single_output
)
from ml.model.loss import make_qd_loss_fn
from ml.features import (
    make_X_and_y,
    make_sqm_X,
    make_htc_proxy_X,
    unscale_outputs
)
from ml.common.paths import MODELS_DIR, MODEL_CONFIG_DIR
from ml.model.helpers import read_config_file


_ENSEMBLE_PATH = os.path.join(MODELS_DIR, "ensemble")


class Ensemble:
    """Class that wraps an ensemble model.

    This consists of the specified number of individual models which are trained
    and saved to disk.

    Ensemble predictions are simply the average of all the individual models' predictions.

    The only argument to the constructor is the name of the ensemble, which must correspend
    to a config file in model-config/.

    Config options are:

    Option                    Type   Description  Default
    ------                    ----   -----------  -------
    ensemble_size             int    Number of ensemble members                     None
    epochs                    int    Number of training epochs                      None
    input_days                int    Number of days to run model on                 None
    training_repeats          int    Number of training samples to make per house   None
    loss_fn_lambda            float  lambda param for loss function                 None
    loss_fn_s                 float  s param for loss function                      None
    training_htc_lower_bound  float  Only train on houses with htc > value          None
    training_htc_upper_bound  float  Only train on houses with htc > value          None
    channels                  list   Which channels the model will use              [mean_temp, outdoor_temp,
                                                                                     gas_kwh, elec_kwh]
    validation_split          float  validation_split param in keras                None
    early_stopping_variable   str    For keras EarlyStopping, e.g. val_loss         None
    early_stopping_patience   int    For keras EarlyStopping, how many iterations   None
                                     to wait before stopping early
    ensemble_type             str    Either regular or bagged                       None
    static_features           list   Either floor_area or proxy                     None
                                     (proxy is sum(gas) / mean(in_temp - out_temp)
                                      over whole data series)
    scale_inputs              bool   Whether or not to scale inputs into (0,1)      False
    scale_targets             bool   Whether or not to scale targets into (0,1)     False
    single_output_only        bool   Whether or not to only output HTC, rather      False
                                     than upper and lower bounds
    building_types            list   Subset of SEMI, MID_TERRACE,END_TERRACE,       []
                                     DETACHED, BUNGALOW. Only these building types
                                     will be used in training data.
    """
    # Set default values for ensemble options
    ensemble_size: int = None
    epochs: int = None
    input_days: int = None
    training_repeats: int = None
    loss_fn_lambda: float = None
    loss_fn_s: float = None
    training_htc_lower_bound: float = None
    training_htc_upper_bound: float = None
    channels: list = ["mean_temp", "outdoor_temp", "gas_kwh", "elec_kwh"]
    validation_split: float = None
    early_stopping_variable: str = None
    early_stopping_patience: int = None
    ensemble_type: str = None
    static_features: list = None
    scale_inputs: bool = False
    scale_targets: bool = False
    single_output_only: bool = False
    building_types: list = []

    def __init__(self, name: str):
        """Constructor. Read all the config options from the file and set them on self."""
        self.name = name
        config_path = os.path.join(MODEL_CONFIG_DIR, f"{ensemble_name}.txt")
        options = read_config_file(config_path)

        for option, value in options.items():
            setattr(self, option, value)

    def train(self):
        """Train all the ensemble members, save them in models/ensemble/<ensemble_name>
        and print out some stats about the training."""
        # Make the basic X and y
        X_train, y_train = self.__make_X_and_y()

        # Filter channels and add static features if necessary
        X_train, y_train, X_static = self.__modify_training_features(X_train, y_train)

        # Add early stopping condition if specified
        if self.early_stopping_variable:
            es = EarlyStopping(monitor=self.early_stopping_variable, patience=self.early_stopping_patience)

        # Make loss function
        loss_fn = make_qd_loss_fn(lam=self.loss_fn_lambda, s=self.loss_fn_s)

        # Make dir to save models in
        self.ensemble_path = os.path.join(_ENSEMBLE_PATH, ensemble_name)
        os.makedirs(self.ensemble_path)

        # Make the individual ensemble members
        for i in range(self.ensemble_size):
            self.__train_ensemble_member(i, X_train, y_train, X_static, loss_fn, es)

        self.__create_training_statistics(X_train, y_train, X_static)

    def predict(self, X: np.ndarray):
        """Make predictions on X."""
        ensemble_predictions = []

        # Load each of the ensemble members and make predictions. Append predictions to ensemble_predictions.
        for f in os.scandir(self.ensemble_path):
            if "json" in f.name:
                member_name = f.name[:-5]
                model = self.__load_ensemble_member(member_name)
                pred = model.predict(X)
                ensemble_predictions.append(pred)

        predictions = np.array(ensemble_predictions).mean(axis=0)
        # The prediction intervals may not be the right way round, so sort them to make sure they are.
        predictions.sort(axis=1)

        return predictions

    def __create_training_statistics(self, X_train: np.ndarray, y_train: np.ndarray, X_static: np.ndarray):
        """
        Computes, some statistics about how the model was trained, prints them out and also
        writes them to models/ensemble/<model name>/trainingsummary.txt.

        For models with a single output feature, just computes the training RMSE.
        For models with upper and lower bounds as output, computes the RMSE + the average
        prediction interval width and prediction interval coverage.
        """
        # Write some statistics about the trained model to file
        has_static_features = self.static_features and len(self.static_features) > 0
        X_pred = [X_train, X_static] if has_static_features else X_train

        if self.single_output_only:
            preds = self.predict(X_pred)
            if self.scale_targets:
                pi_preds = unscale_outputs(preds)
                y_train = unscale_outputs(y_train)

            rmse = np.sqrt(np.mean((preds - y_train) ** 2))
            print(f"Train RMSE: {round(rmse)}")

            with open(os.path.join(self.ensemble_path, "trainingsummary.txt"), "w+") as f:
                f.write(f"Train RMSE: {round(rmse)}")
        else:
            pi_preds = self.predict(X_pred)
            if self.scale_targets:
                pi_preds = unscale_outputs(pi_preds)
                y_train = unscale_outputs(y_train)

            htc_preds = pi_preds.mean(axis=1)
            rmse = np.sqrt(np.mean((htc_preds - y_train) ** 2))
            pi_coverage = np.mean(((y_train >= pi_preds[:,0]) & (y_train <= pi_preds[:,1])).astype(int))
            mean_pi_width = np.mean(pi_preds[:,1] - pi_preds[:,0])

            print("\nTraining summary")
            print("----------------")
            print(f"Train RMSE: {round(rmse)}")
            print(f"PI coverage: {round(pi_coverage * 100)}%")
            print(f"Mean PI width: {round(mean_pi_width)}")

            with open(os.path.join(self.ensemble_path, "trainingsummary.txt"), "w+") as f:
                f.write(f"Train RMSE: {round(rmse)}\n")
                f.write(f"PI coverage: {round(pi_coverage * 100)}%\n")
                f.write(f"Mean PI width: {round(mean_pi_width)}")

    def __make_X_and_y(self):
        """Makes X and y for training."""
        # Make the basic X and y
        X_train, y_train = make_X_and_y(
            days=self.input_days,
            repeats=self.training_repeats,
            scale_inputs=self.scale_inputs,
            scale_targets=self.scale_targets,
            building_types=self.building_types,
            htc_upper_bound=self.training_htc_upper_bound,
            htc_lower_bound=self.training_htc_lower_bound
        )

        return X_train, y_train

    def __modify_training_features(self, X_train: np.ndarray, y_train: np.ndarray):
        """Takes X and y and filters channels and adds static features if necessary."""
        has_static_features = self.static_features and len(self.static_features) > 0

        # If static features are to be used in the model, make the training data arrays
        if has_static_features:
            # This assumes there is only ever one static features
            # TODO deal with the case where there are multiple
            if self.static_features == ["floor_area"]:
                X_static = make_sqm_X(repeats=self.training_repeats, scale=self.scale_inputs)
            elif self.static_features == ["proxy"]:
                X_static = make_htc_proxy_X(X_train)
        else:
            X_static = np.array([])

        # Filter channels if specified
        channels = self.channels.copy()
        if len(channels) < 4:
            all_channels = ["mean_temp", "outdoor_temp", "gas_kwh", "elec_kwh"]

            if "temp_diff" in channels:
                temp_diff_X = X_train[:,:,0] - X_train[:,:,1]
                temp_diff_X = temp_diff_X.reshape((temp_diff_X.shape[0], temp_diff_X.shape[1], 1))
                channels.remove("temp_diff")
                other_channel_indices = [all_channels.index(c) for c in channels]
                other_channels_X = X_train[:,:,other_channel_indices]

                X_train = np.concatenate((temp_diff_X, other_channels_X), axis=2)
            else:
                channel_indices = [all_channels.index(c) for c in channels]
                X_train = X_train[:,:,channel_indices]

        return X_train, y_train, X_static

    def __train_ensemble_member(
        self,
        i: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_static: np.ndarray,
        loss_fn: Callable,
        es: EarlyStopping
    ):
        """Trains an individual member of the ensemble and saves it to file."""
        tf.set_random_seed(i)

        # Instantiate the model
        has_static_features = self.static_features and len(self.static_features) > 0

        if has_static_features and self.single_output_only:
            model = base_cnn_with_static_features_and_single_output(
                cnn_input_shape=X_train[0].shape,
                static_features_input_shape=X_static[0].shape,
            )
        elif has_static_features and not self.single_output_only:
            model = base_cnn_with_static_features(
                cnn_input_shape=X_train[0].shape,
                static_features_input_shape=X_static[0].shape,
                loss_fn=loss_fn
            )
        elif not has_static_features and self.single_output_only:
            model = base_cnn_single_output(input_shape=X_train[0].shape)
        else:
            model = base_cnn(input_shape=X_train[0].shape, loss_fn=loss_fn)

        # Train the model
        # Construct the arguments to fit() as a list and a dict as it makes the logic slightly less messy
        kwargs = {"epochs": self.epochs, "validation_split": self.validation_split, "verbose": 0}
        if self.early_stopping_variable:
            kwargs["callbacks"] = [es]

        # Create training data based on whether we are using bagging and whether there are static features
        # Bagged AND static features
        if self.ensemble_type == "bagged" and has_static_features:
            X_train_, X_static_, y_train_,  = _make_bagged_training_sets(X_train, y_train, X_static, seed=i)
            args = [{"timeseries": X_train_, "static_features": X_static_}, {"output": y_train}]

        # Bagged AND NO static features
        elif self.ensemble_type == "bagged" and not has_static_features:
            args = _make_bagged_training_sets(X_train, y_train, seed=i)

            # Regular AND static features
        elif self.ensemble_type == "regular" and has_static_features:
            args = [{"timeseries": X_train, "static_features": X_static}, {"output", y_train}]

            # Regular AND NO static features
        elif self.ensemble_type == "regular" and not has_static_features:
            args = [X_train, y_train]

        model.fit(*args, **kwargs)

        # Save model
        model.save_weights(os.path.join(self.ensemble_path, f"model{i}.h5"), save_format="h5")
        with open(os.path.join(self.ensemble_path, f"model{i}.json"), "w+") as f:
            f.write(model.to_json())

        print(f"Trained model {i + 1} of {self.ensemble_size}", flush=True)

        # Clear model from memory
        clear_session()

    def __load_ensemble_member(self, member_name: str):
        """Load CNN from saved serialized architecture and weights."""
        with open(os.path.join(self.ensemble_path, f"{member_name}.json")) as f:
            json_str = f.read()
            model = k.models.model_from_json(json_str, custom_objects={"GlorotUniform": k.initializers.glorot_uniform})

        model.load_weights(os.path.join(self.ensemble_path, f"{member_name}.h5"))

        return model


def _make_bagged_training_sets(
    X: np.ndarray,
    y: np.ndarray,
    X_static: np.ndarray = None,
    seed: int = 0
):
    """Generate new copies of X and y by randomly choosing with replacement."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(X.shape[0], X.shape[0], replace=True)

    if X_static is None:
        return X[indices,:,:], y[indices]
    else:
        return X[indices,:,:], X_static[indices,:], y[indices]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Correct usage via make is: make ensemble config=<config file name>")
    else:
        # Allow user to pass in either e.g. `model1.txt` or just `model1`
        ensemble_name = sys.argv[1].replace(".txt", "")
        ensemble = Ensemble(ensemble_name)
        ensemble.train()
