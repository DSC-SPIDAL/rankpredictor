import tensorflow as tf
import numpy as np

def gaussian_likelihood(theta):

    def gaussian_loss(y_true, y_pred):
        sigma = theta[0]
        x = y_true
        mu = y_pred
        F = tf.math

        ll = tf.reduce_mean( (F.log(sigma)  + 0.5 * F.log(2 * np.math.pi) + 0.5 * F.square((x - mu) / sigma)) )
        return ll 
        #return tf.reduce_mean(0.5*tf.math.log(sigma) + 0.5*tf.math.truediv(tf.math.square(y_true - y_pred), sigma)) + 1e-6 + 6
    return gaussian_loss

def gaussian_sampler(theta, num_samples=1):
    """
    input:
        theta 
    return samples.shape = [num_samples]
    """
    mu = theta[0]
    sigma = theta[1]
    #return np.random.normal(loc=mu, scale=np.sqrt(sigma), size=num_samples)[0]
    return np.random.normal(loc=mu, scale=sigma, size=num_samples)[0]

def studentt_likelihood(theta):

    def studentt_loss(y_true, y_pred):

        print('y_true.shape=%s'%y_true.shape)

        sigma, nu = theta[0], theta[1]

        #mu, sigma, nu = self.mu, self.sigma, self.nu
        mu = y_pred
        F = tf.math

        nup1_half = (nu + 1.0) / 2.0
        part1 = 1.0 / nu * F.square((y_true - mu) / sigma)
        Z = (
            F.lgamma(nup1_half)
            - F.lgamma(nu / 2.0)
            - 0.5 * F.log(np.math.pi * nu)
            - F.log(sigma)
        )

        ll = Z - nup1_half * F.log1p(part1)
        return -ll
    return studentt_loss

def studentt_sampler(theta, num_samples=1):
    mu, sigma, nu = theta[0],theta[1], theta[2]
    F = tf.math

    #gammas = tf.random.gamma([1],
    #        alpha=nu / 2.0, beta=2.0 / (nu * F.square(sigma))
    #        )[0]
    #normal = tf.random.normal(
    #        mu=mu, sigma=1.0 / F.sqrt(gammas)
    #       )

    gammas = np.random.gamma(shape = nu/2.0, scale = nu * F.square(sigma)/2.0,size=(1,))[0]

    sigma_s = 1.0 / F.sqrt(gammas)

    #return np.random.normal(loc=mu, scale=np.sqrt(sigma_s), size=num_samples)[0]
    return np.random.normal(loc=mu, scale=sigma_s, size=num_samples)[0]
