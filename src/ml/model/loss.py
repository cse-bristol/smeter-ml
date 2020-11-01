"""Provides a constructor for the QD loss function.

This loss function is used for producing prediction intervals as neural network outputs
rather than just single predictions.
"""

import tensorflow as tf

def make_qd_loss_fn(lam: float = 15.0, s: float = 160.0, alpha: float = 0.05):
    """
    Returns a 'quality-driven' loss function with the given hyper-parameters.
    See this paper for explanation https://arxiv.org/pdf/1802.07167.pdf.
    See also https://github.com/TeaPearce/Deep_Learning_Prediction_Intervals/ for some code.

    Params:
    lam -- a constant factor in the second term of the expression.
    s -- softening factor for the sigmoid, which is used as a differentiable approximation of the sign function.
    alpha -- the level of confidence: alpha = 0.05 <=> 95% confidence.
    """
    def qd_loss(y_true, y_pred):
        """Loss function from https://arxiv.org/pdf/1802.07167.pdf"""
        y_L = y_pred[:,0]
        y_U = y_pred[:,1]
        y_T = y_true[:,0]

        # Sample size
        n = tf.cast(tf.size(y_T), tf.float32)

        sigmoid = lambda x: 1.0 / (1.0 + tf.math.exp(tf.clip_by_value(-x, -10.0, 10.0)))
        k_U = sigmoid((y_U - y_T) * s)
        k_L = sigmoid((y_T - y_L) * s)
        k = tf.multiply(k_U, k_L)

        # Prediction Interval Coverage Probability = the proportion of true y's covered by prediction intervals
        picp = tf.reduce_mean(k)
        # Mean Prediction Interval Width = mean width of PIs, but only those which capture y's
        mpiw = tf.divide(tf.reduce_sum(tf.abs(y_U - y_L) * k), tf.reduce_sum(k) + 0.001)

        second_term = lam * tf.sqrt(n) * tf.square(tf.maximum(0., (1. - alpha) - picp))

        return mpiw + second_term

    return qd_loss
