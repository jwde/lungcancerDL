import numpy as np
from random import shuffle

def structured_loss_simple(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  dim = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    count = 0
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if j == y[i]:
        continue
      if margin > 0:
        count += 1
        dW[:,j] += X[i]
        loss += margin
    dW[:,y[i]] -= count * X[i]
  dW /= num_train
  dW += reg * W

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather than first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  ############################################################################# 


  return loss, dW


def structured_loss_fast(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as structured_loss_simple.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]

  scores = X.dot(W)
  margin = scores - scores[np.array(range(len(y))), y][:,np.newaxis] + 1
  margin[margin < 0] = 0
  losses = np.sum(margin, axis=1) - margin[np.array(range(len(y))), y]
  loss = np.sum(losses)
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  # Calculate loss for j=y_i
  correct_class_loss = (np.sum(margin != 0, axis=1) - 1)[:,np.newaxis] * X
  # Calculate loss for j!=y_i
  class_losses = (margin > 0)[:,:,np.newaxis] * X[:,np.newaxis,:]
  class_losses[np.array(range(len(y))), y] = - correct_class_loss
  dW = np.transpose(class_losses.sum(axis=0))
  dW /= num_train
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  # compute the loss and the gradient

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.

  # Add regularization to the loss.

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
