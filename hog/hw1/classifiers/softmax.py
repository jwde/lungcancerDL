import numpy as np
from random import shuffle

def softmax_loss_simple(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  num_classes = W.shape[1]
  num_train = X.shape[0]
  dim = X.shape[1]
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  dScores = np.zeros((num_train,num_classes))
  max_score = np.max(X.dot(W))
  for i in xrange(num_train):
      scores = X[i].dot(W)
      scores -= max_score # normalize for largest score of zero
      log_term = 0
      for j in xrange(num_classes):
          log_term += np.exp(scores[j])
      pk = np.exp(scores[y[i]]) / log_term
      loss += -np.log(pk)
      for k in xrange(num_classes):
          dScores[i,k] = np.exp(scores[k]) / log_term
      dScores[i,y[i]] -= 1
  dW = np.dot(np.transpose(X),dScores)
  dW /= num_train
  dW += reg * W
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  # If you get stuck, don't forget about these resources:                     #
  # http://cs231n.github.io/neural-networks-case-study/#linear                #
  # http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/#
  #############################################################################


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_fast(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  dim = X.shape[1]
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  dScores = np.zeros((num_train,num_classes))
  scores = X.dot(W)
  scores -= np.max(scores)
  e_scores = np.exp(scores)
  probs = e_scores / np.sum(e_scores, axis = 1)[:, np.newaxis]
  loss = np.sum(-np.log(probs[range(num_train), y]))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dScores = probs
  dScores[range(num_train), y] -= 1
  dW = np.dot(np.transpose(X), dScores)
  dW /= num_train
  dW += reg * W


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

