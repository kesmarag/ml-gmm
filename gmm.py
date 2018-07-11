import matplotlib.mlab as mlab
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans


class GMM(object):
  def __init__(self, num_mixtures, data_dim):
    self._m = num_mixtures
    self._d = data_dim
    self._pi = np.ones((self._m, ), dtype='float64')/self._m
    # self._mu = np.zeros((self._m, self._d), dtype='float64')
    self._mu = np.random.randn(self._m, self._d)
    self._sigma = np.array([np.identity(self._d, dtype='float64')] * self._m)
    self._gamma = None
    self._post = 0.0

  def fit(self, data, max_steps, tol, use_kmeans=False):
    step = 0
    post_prev = (tol + 1.0) * data.shape[0]
    while np.abs(
      self._post - post_prev) / data.shape[0] > tol and step < max_steps:
      if step == 0 and use_kmeans:
        kmeans = KMeans(
          n_clusters=self._m, random_state=0).fit(data)
        self._mu = kmeans.cluster_centers_
      post_prev = self._post
      self._expectation_maximization(data, tol)
      step += 1
    print(self._post)

  def get_marginal(self, i, interval, plot_interval,
                   scaling_factor=1, num_samples=1000):
    ra = (interval[1] - interval[0]) / (2.0 * scaling_factor)
    c = (interval[0] + interval[1]) / 2.0
    mu = ra * self._mu[:, i] + c
    # print(mu)
    var = self._sigma[:, i, i] * ra ** 2
    # print(var)
    x = np.linspace(plot_interval[0], plot_interval[1], num_samples)
    distr = np.zeros_like(x)
    for m in range(self._m):
      distr += self._pi[m] * mlab.normpdf(x, mu[m], np.sqrt(var[m]))
    return x, distr

  def get_multivariate(self, i, j, intervals,
                       plot_intervals, scaling_factor=1, num_samples=100):
    ra_i = (intervals[0][1] - intervals[0][0]) / (2.0 * scaling_factor)
    ra_j = (intervals[1][1] - intervals[1][0]) / (2.0 * scaling_factor)
    c_i = (intervals[0][0] + intervals[0][1]) / 2.0
    c_j = (intervals[1][0] + intervals[1][1]) / 2.0
    mu_i = ra_i * self._mu[:, i] + c_i
    mu_j = ra_j * self._mu[:, j] + c_j
    x_i = np.linspace(plot_intervals[0][0], plot_intervals[0][1], num_samples)
    x_j = np.linspace(plot_intervals[1][0], plot_intervals[1][1], num_samples)
    x, y = np.meshgrid(x_i, x_j)
    pos = np.dstack((x, y))
    distr = np.zeros((num_samples, num_samples))
    for m in range(self._m):
      mu = [mu_i[m], mu_j[m]]
      cov = [[self._sigma[m, i, i] * ra_i**2, self._sigma[m, i, j] * ra_i * ra_j],
             [self._sigma[m, i, j] * ra_i * ra_j, self._sigma[m, j, j] * ra_j**2]]
      while not self._is_pos_def(cov):
         cov = cov + 0.05 * np.array(
          [np.identity(2, dtype=np.float64)]) * cov
      rv = multivariate_normal(mu, cov)
      distr += self._pi[m] * rv.pdf(pos)
    return x, y, distr


  def __str__(self):
    frame_len = 35
    s = '#' * frame_len
    s += '\n' + '-' * frame_len + '\n'
    s += 'kesmarag.ml.gmm.GMM'
    s += '\n' + '-' * frame_len + '\n'
    s += ' - number of mixtures: ' + str(self._m) + '\n'
    s += ' - observation length: ' + str(self._d)
    s += '\n' + '-' * frame_len + '\n'
    s += ' - mixing coefficients'
    s += '\n' + '-' * frame_len + '\n'
    s += str(self._pi)
    s += '\n' + '-' * frame_len + '\n'
    s += ' - mean values'
    s += '\n' + '-' * frame_len + '\n'
    s += str(self._mu)
    s += '\n' + '-' * frame_len + '\n'
    s += ' - covariances'
    s += '\n' + '-' * frame_len + '\n'
    s += str(self._sigma)
    s += '\n' + '-' * frame_len
    s += '\n' + '#' * frame_len + '\n'
    return s

  def _expectation_maximization(self, data, tol):
    # E - step
    mixture = np.zeros((self._m, data.shape[0]), dtype='float64')
    # self._gamma = np.zeros((self._m, data.shape[0]), dtype='float64')
    for m in range(self._m):
        mixture_m = multivariate_normal.pdf(
          data, self._mu[m], self._sigma[m])
        mixture[m, :] = mixture_m
    # print(mixture)
    pi_mixture = np.multiply(np.expand_dims(self._pi, -1), mixture)
    sum_m_pi_mixture = np.sum(pi_mixture, axis=0)
    self._gamma = pi_mixture/sum_m_pi_mixture
    # M - step
    N_m = np.sum(self._gamma, axis=-1)
    self._pi = N_m/data.shape[0]
    self._mu = np.dot(self._gamma, data)/np.expand_dims(N_m, -1)
    x_m_mu = np.expand_dims(
      np.subtract(np.expand_dims(data, -2), self._mu), -1)
    xi = np.matmul(x_m_mu, np.matrix.transpose(x_m_mu, (0, 1, 3, 2)))
    phi = np.multiply(
      np.expand_dims(np.expand_dims(np.transpose(self._gamma), -1), -1), xi)
    self._sigma = np.sum(
      phi, axis=0)/np.expand_dims(np.expand_dims(N_m, -1), -1)
    # chech if the covariance matrix is positive definite
    for k in range(self._m):
      j = 0
      while not self._is_pos_def(self._sigma[k]):
        j += 1
        # print('.. not positive definite ..')
        self._sigma[k] = self._sigma[k] + 0.05 * np.array(
          [np.identity(self._d, dtype=np.float64)])*self._sigma[k]
    # Evaluate the log likelihood posterior distributions
    self._post = np.sum(np.log(sum_m_pi_mixture))
    # print(self._post)

  def _is_pos_def(self, sigma):
    return np.all(np.linalg.eigvals(sigma) > 0.02)