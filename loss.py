import tensorflow as tf
from keras import backend as K


def loss_mse_ssim(y_true, y_pred):
    ssim_para = 1e-1 # 1e-2
    mse_para = 1

    y_true = (y_true - K.min(y_true)) / (K.max(y_true) - K.min(y_true))
    y_pred = (y_pred - K.min(y_pred)) / (K.max(y_pred) - K.min(y_pred))

    ssim_loss = ssim_para * (1 - K.mean(tf.image.ssim(y_true, y_pred, 1)))
    mse_loss = mse_para * K.mean(K.square(y_pred - y_true))

    return mse_loss + ssim_loss


def loss_huber(y_true, y_pred, delta=1.2, lamda=0.1):
    # nomolization
    y_true = (y_true - K.min(y_true)) / (K.max(y_true) - K.min(y_true))
    y_pred = (y_pred - K.min(y_pred)) / (K.max(y_pred) - K.min(y_pred))

    residual = K.abs(y_true, y_pred)
    ssim = tf.image.ssim(y_true, y_pred, 1)

    large_loss = 0.5 * (K.square(residual) + lamda * K.pow(1.0 - ssim, delta))
    small_loss = (residual + lamda * (1.0 - ssim)) / delta - 0.5 * K.square(delta)
    cond = tf.less(residual, delta)
    final_loss = K.mean(tf.where(cond, large_loss, small_loss))
    return final_loss
