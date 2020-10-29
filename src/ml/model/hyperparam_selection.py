import sys
import numpy as np
from ml.model.cnn import base_cnn
from ml.model.loss import make_qd_loss_fn
from ml.features import make_X_and_y
from tensorflow.keras.backend import clear_session


def sample_loss_fn_params(lam_values=[0.1, 1., 10., 100.], s_values=[0.1, 1., 10., 100.]):
    """Try out a range of values for the QD loss function parameters and print out the results.

    The QD loss function has two parameters which we can tweak:
        lamda: a constant factor in the second term of the expression - determines how much weighting
            we should assign to getting the model to capture targets in the prediction interval.
        s: softening factor for the sigmoid, which is used as a differentiable approximation of the sign function.

    This function trains a series of models using all the combinations of the supplied values of lambda and s.
    It prints out the prediction interval capture rate in training, the prediction intervals in test and the
    mean prediction interval widths and accuracies over all the test samples.

    Args:
        lam_values (list): List of values of lambda to try.
        s_values (list): List of values of s to try.

    TODO: At the moment this only uses the base CNN model with the default arguments - it would be useful
        to be able to try variations on the base model.
    """
    X_train, y_train = make_X_and_y(days=21, repeats=4)
    X_test, y_test = make_X_and_y(test=True, days=21, repeats=4)

    for lam in lam_values:
        for s in s_values:
            qd_loss = make_qd_loss_fn(lam=lam, s=s)

            # Try it with 5 different initializations
            pi_widths = []
            accuracies = []

            print("\n")
            print("===========================")
            print(f"= lambda = {lam}; s = {s}")
            print("===========================")

            for j in range(5):
                print(".....")
                model = base_cnn(input_shape=X_train[0].shape, loss_fn=qd_loss)
                model.fit(X_train, y_train, epochs=10, verbose=0, validation_split=0.5)

                # See what proportion of predicted values end up in PIs
                # (Make predictions on training set)
                preds = model.predict(X_train)
                def in_interval(x, interval):
                    interval.sort()
                    return x >= interval[0] and x <= interval[1]

                captured = [1 if in_interval(y_train[i], x) else 0 for i, x in enumerate(preds)]
                print("Percentage in PIs:", (sum(captured) / len(captured)) * 100, "%")

                # Now make predictions on test set
                predictions = model.predict(X_test)

                # Clear the model from memory
                clear_session()

                for i, pred in enumerate(predictions):
                    target = y_test[i]
                    lower, upper = pred
                    pi_width = np.abs(upper - lower)
                    acc = np.abs(np.mean(pred) - target)
                    pi_widths.append(pi_width)
                    accuracies.append(acc)

                    print(f"Target: {round(target)}. Predicted: [{round(lower), round(upper)}]")

            print(".......................")
            print(f"Mean PI width: {round(np.mean(pi_widths))}")
            print(f"Mean accuracy: {round(np.mean(accuracies))}")


if __name__ == "__main__":
    # This can be called with either, both, or neither of the arguments `s` and `lam`,
    # which should be a comma separated list of values to try for these parameters
    # in the loss function.
    kwargs = {}

    s_str = sys.argv[1][2:]
    if len(s_str) > 0:
        kwargs["s_values"] = [float(x) for x in sys.argv[1][2:].split(",")]

    lam_str = sys.argv[2][4:]
    if len(lam_str) > 0:
        kwargs["lam_values"] = [float(x) for x in sys.argv[2][4:].split(",")]

    sample_loss_fn_params(**kwargs)
