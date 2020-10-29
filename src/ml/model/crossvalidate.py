import os
import sys
import numpy as np
from tensorflow.keras.backend import clear_session
from sklearn.model_selection import KFold
from ml.features import make_X_and_y, make_sqm_X, make_htc_proxy_X
from ml.model.helpers import read_config_file
from ml.model.loss import make_qd_loss_fn
from ml.common.paths import MODEL_CONFIG_DIR
import ml.model.cnn as cnn


def kfold_crossvalidate(model_name: str, k: int = 5):
    """Performs k-fold crossvalidation on a model.

    This is mostly for experimentation purposes.
    The only argument is `model_name`, which is the name of the config file, e.g. model0 for model0.txt.
    It will only train a single model, so the value of ensemble_size in the model config will be ignored.
    """
    config_path = os.path.join(MODEL_CONFIG_DIR, f"{model_name}.txt")
    options = read_config_file(config_path)

    # Make the training feature arrays
    X, y = make_X_and_y(
        days=options["input_days"],
        repeats=options["training_repeats"],
        scale_inputs=options["scale_inputs"],
        scale_targets=options["scale_targets"],
        building_types=options["building_types"],
        htc_upper_bound=options["training_htc_upper_bound"],
        htc_lower_bound=options["training_htc_lower_bound"]
    )

    single_output = bool(options["single_output_only"])
    has_static_features = ("static_features" in options) and len(options["static_features"]) > 0

    # If there are any static features, make the static feature arrays
    if has_static_features:
        # This assumes there is only ever one static features
        # TODO deal with the case where there are multiple
        if options["static_features"] == ["floor_area"]:
            X_static = make_sqm_X(repeats=options["training_repeats"], scale=options["scale_inputs"])
        elif options["static_features"] == ["proxy"]:
            X_static = make_htc_proxy_X(X)
    else:
        X_static = np.array([])

    # Determine which model constructor to use and make its arguments.
    # A new model will be constructed and trained in each of the k loops.
    if has_static_features and single_output:
        model_constructor = cnn.base_cnn_with_static_features_and_single_output
        kwargs = {
            "cnn_input_shape": X[0].shape,
            "static_features_input_shape": X_static[0].shape,
        }
    elif has_static_features and not single_output:
        model_constructor = cnn.base_cnn_with_static_features
        loss_fn = make_qd_loss_fn(lam=options["loss_fn_lambda"], s=options["loss_fn_s"])
        kwargs = {
            "cnn_input_shape": X[0].shape,
            "static_features_input_shape": X_static[0].shape,
            "loss_fn": loss_fn
        }
    elif not has_static_features and single_output:
        model_constructor = cnn.base_cnn_single_output
        kwargs = {"input_shape": X[0].shape}
    else:
        model_constructor = cnn.base_cnn
        kwargs = {"input_shape": X[0].shape, "loss_fn": loss_fn}

    # Instantiate the k-fold generator
    kf = KFold(n_splits=k, shuffle=True, random_state=0)

    # Store the train and test accuracies so we can print out their means later
    train_rmses = []
    test_rmses = []

    for i, (train_ind, test_ind) in enumerate(kf.split(X)):
        print(f"Fold = {i}")
        model = model_constructor(**kwargs)
        X_train, X_test = X[train_ind], X[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]

        if has_static_features:
            X_static_train, X_static_test = X_static[train_ind], X_static[test_ind]
            model.fit(
                {"timeseries": X_train, "static_features": X_static_train},
                {"output": y_train},
                epochs=options["epochs"],
                verbose=0
            )
            # Evaluate the model
            train_mse, train_rmse = model.evaluate(
                {"timeseries": X_train, "static_features": X_static_train},
                {"output": y_train},
                verbose=0
            )
            test_mse, test_rmse = model.evaluate(
                {"timeseries": X_test, "static_features": X_static_test},
                {"output": y_test},
                verbose=0
            )
        else:
            model.fit(X_train, y_train, epochs=options["epochs"], verbose=0)
            # Evaluate the model
            train_mse, train_rmse = model.evaluate(X_train, y_train, verbose=0)
            test_mse, test_rmse = model.evaluate(X_test, y_test, verbose=0)

        print("Training RMSE", train_rmse)
        print("Testing RSME", test_rmse)

        train_rmses.append(train_rmse)
        test_rmses.append(test_rmse)

        # Clear the model from memory
        clear_session()

    print("--------------------------")
    print("Mean training RMSE", np.mean(train_rmses))
    print("Mean testing RMSE", np.mean(test_rmses))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Expected at least one argument. Example: make crossvalidate config=model1")

    kfold_crossvalidate(sys.argv[1])
