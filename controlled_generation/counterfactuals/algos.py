import scipy
import numpy as np

def fit_optimal_transport(train_x, train_z, dev_x, dev_z=None, eps=1e-7):

  # ideally maps source --> target, assuming gaussian representations. Ensures the same mean and covariacne after the transformation, while minimally changing the L2 distance.
  # TODO: think of a version of this for mixture-of-gaussians.

  def matrix_squared_root(A):

    evalues, evectors = np.linalg.eig(A)
    sqrt_matrix = evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)
    return sqrt_matrix

  # fit a classifier to predict z from x (to be used on the dev example, in order to decide which representations to move)
  if dev_z is None:
    clf = MLPClassifier(hidden_layer_sizes=(128), max_iter=10000)
    clf.fit(train_x, train_z)
    dev_z_pred = clf.predict(dev_x)
  else:
    # else, use gold labels (`oracle mode`)
    dev_z_pred = dev_z

  source_x = dev_x[dev_z_pred == 0]
  target_x = dev_x[dev_z_pred == 1]

  # cov
  cov_source = np.cov(source_x.T).real + eps
  cov_target = np.cov(target_x.T).real + eps
  # mean
  mean_source = source_x.mean(axis=0)
  mean_target = target_x.mean(axis=0)

  # optimal transport

  cov_source_sqrt = matrix_squared_root(cov_source)
  cov_target_sqrt = matrix_squared_root(cov_target)
  cov_source_sqrt_inv = scipy.linalg.inv(cov_source_sqrt)
  cov_target_sqrt_inv = scipy.linalg.inv(cov_target_sqrt)

  A = cov_source_sqrt_inv @ matrix_squared_root(cov_source_sqrt @ cov_target @ cov_source_sqrt) @ cov_source_sqrt_inv

  dev_x_transformed = dev_x.copy()
  dev_x_transformed[dev_z_pred == 0] = mean_target + (dev_x_transformed[dev_z_pred == 0] - mean_source) @ A

  train_x_transformed = train_x.copy()
  mean_source_train, mean_target_train = train_x_transformed[train_z == 0].mean(axis=0), train_x_transformed[train_z == 1].mean(axis=0)
  train_x_transformed[train_z == 0] = mean_target_train + (train_x_transformed[train_z == 0] - mean_source_train) @ A

  return train_x_transformed, dev_x_transformed, A

def fit_optimal_transport2(train_x, train_z, dev_x, dev_z=None, eps=1e-7):

  # ideally maps source --> target, assuming gaussian representations. Ensures the same mean and covariacne after the transformation, while minimally changing the L2 distance.
  # TODO: think of a version of this for mixture-of-gaussians.

  def matrix_squared_root(A):

    evalues, evectors = np.linalg.eig(A)
    sqrt_matrix = evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)
    return sqrt_matrix

  # fit a classifier to predict z from x (to be used on the dev example, in order to decide which representations to move)
  if dev_z is None:
    clf = MLPClassifier(hidden_layer_sizes=(128), max_iter=10000)
    clf.fit(train_x, train_z)
    dev_z_pred = clf.predict(dev_x)
  else:
    # else, use gold labels (`oracle mode`)
    dev_z_pred = dev_z

  source_x = dev_x.copy()
  target_x = dev_x[dev_z_pred == 1]

  # cov
  cov_source = np.cov(source_x.T).real + eps
  cov_target = np.cov(target_x.T).real + eps
  # mean
  mean_source = source_x.mean(axis=0)
  mean_target = target_x.mean(axis=0)

  # optimal transport

  cov_source_sqrt = matrix_squared_root(cov_source)
  cov_target_sqrt = matrix_squared_root(cov_target)
  cov_source_sqrt_inv = scipy.linalg.inv(cov_source_sqrt)
  cov_target_sqrt_inv = scipy.linalg.inv(cov_target_sqrt)

  A = cov_source_sqrt_inv @ matrix_squared_root(cov_source_sqrt @ cov_target @ cov_source_sqrt) @ cov_source_sqrt_inv

  # dev_x_transformed = dev_x.copy()
  dev_x_transformed = mean_target + (dev_x - mean_source) @ A

  train_x_transformed = train_x.copy()
  mean_source_train, mean_target_train = train_x_transformed.mean(axis=0), train_x_transformed[train_z == 1].mean(axis=0)
  train_x_transformed = mean_target_train + (train_x - mean_source_train) @ A

  return train_x_transformed, dev_x_transformed, A

def fit_optimal_transport3(train_x, train_z, dev_x, target_x=None, eps=1e-7):

  # ideally maps source --> target, assuming gaussian representations. Ensures the same mean and covariacne after the transformation, while minimally changing the L2 distance.
  # TODO: think of a version of this for mixture-of-gaussians.

  def matrix_squared_root(A):

    evalues, evectors = np.linalg.eig(A)
    sqrt_matrix = evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)
    return sqrt_matrix

  # fit a classifier to predict z from x (to be used on the dev example, in order to decide which representations to move)


  source_x = dev_x.copy()

  # cov
  cov_source = np.cov(source_x.T).real + eps
  cov_target = np.cov(target_x.T).real + eps
  # mean
  mean_source = source_x.mean(axis=0)
  mean_target = target_x.mean(axis=0)

  # optimal transport

  cov_source_sqrt = matrix_squared_root(cov_source)
  cov_target_sqrt = matrix_squared_root(cov_target)
  cov_source_sqrt_inv = scipy.linalg.inv(cov_source_sqrt)
  cov_target_sqrt_inv = scipy.linalg.inv(cov_target_sqrt)

  A = cov_source_sqrt_inv @ matrix_squared_root(cov_source_sqrt @ cov_target @ cov_source_sqrt) @ cov_source_sqrt_inv

  # dev_x_transformed = dev_x.copy()
  dev_x_transformed = mean_target + (dev_x - mean_source) @ A

  train_x_transformed = train_x.copy()
  mean_source_train, mean_target_train = train_x_transformed.mean(axis=0), train_x_transformed[train_z == 1].mean(axis=0)
  train_x_transformed = mean_target_train + (train_x - mean_source_train) @ A

  return train_x_transformed, dev_x_transformed, A


# def fit_optimal_transport2(train_x, train_z, dev_x, target_x, dev_z=None, eps=1e-7):

#   # ideally maps source --> target, assuming gaussian representations. Ensures the same mean and covariacne after the transformation, while minimally changing the L2 distance.
#   # TODO: think of a version of this for mixture-of-gaussians.

#   def matrix_squared_root(A):

#     evalues, evectors = np.linalg.eig(A)
#     sqrt_matrix = evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)
#     return sqrt_matrix

#   # fit a classifier to predict z from x (to be used on the dev example, in order to decide which representations to move)
#   if dev_z is None:
#     clf = MLPClassifier(hidden_layer_sizes=(128), max_iter=10000)
#     clf.fit(train_x, train_z)
#     dev_z_pred = clf.predict(dev_x)
#   else:
#     # else, use gold labels (`oracle mode`)
#     dev_z_pred = dev_z

#   source_x = dev_x.copy()

#   # cov
#   cov_source = np.cov(source_x.T).real + eps
#   cov_target = np.cov(target_x.T).real + eps
#   # mean
#   mean_source = source_x.mean(axis=0)
#   mean_target = target_x.mean(axis=0)

#   # optimal transport

#   cov_source_sqrt = matrix_squared_root(cov_source)
#   cov_target_sqrt = matrix_squared_root(cov_target)
#   cov_source_sqrt_inv = scipy.linalg.inv(cov_source_sqrt)
#   cov_target_sqrt_inv = scipy.linalg.inv(cov_target_sqrt)

#   A = cov_source_sqrt_inv @ matrix_squared_root(cov_source_sqrt @ cov_target @ cov_source_sqrt) @ cov_source_sqrt_inv

#   # dev_x_transformed = dev_x.copy()
#   dev_x_transformed = mean_target + (dev_x - mean_source) @ A

#   train_x_transformed = train_x.copy()
#   mean_source_train, mean_target_train = train_x_transformed.mean(axis=0), train_x_transformed[train_z == 1].mean(axis=0)
#   train_x_transformed = mean_target_train + (train_x - mean_source_train) @ A

#   return train_x_transformed, dev_x_transformed, A

