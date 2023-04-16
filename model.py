"""
PFE Boulet Olgiati
"""

import numpy as np

from utils import *


class ESN() : 

  def __init__(self, n_inputs, n_outputs, 
               n_reservoir=400, alpha=0.0001, spectral_radius=1, sparsity=0.5, noise=0.001,
               input_scaling=None, input_shift=None, 
               teacher_scaling=None, teacher_shift=None,
               teacher_forcing=False, feedback_scaling=None, 
               out_activation=identity, inverse_out_activation=identity,
               random_state=None, 
               use_gradient_descent=False, learning_rate=0.0001, l2_rate=0.001, n_epochs=1, batch_size=1, 
               use_rls=False, forget_rate=0.5) :
    
    # spectral radius = spectral radius of W_rec
    # sparsity = proportion of recurrent weights set to zero
    # noise = noise added to each neuron (regularization)
    # random_state = positive seed
    # input_shift: scalar or vector of length n_inputs to add to each input dimension before feeding it to the network.
    # input_scaling: scalar or vector of length n_inputs to multiply with each input dimension before feeding it to the netw.
    # teacher_forcing: if True, feed the target back into output units
    # teacher_scaling: factor applied to the target signal
    # teacher_shift: additive term applied to the target signal
    # out_activation: output activation function (applied to the readout)
    # inverse_out_activation: inverse of the output activation function

    self.n_inputs = n_inputs
    self.n_outputs = n_outputs
    self.n_reservoir = n_reservoir
     
    self.noise = noise
    self.spectral_radius = spectral_radius
    self.alpha = alpha

    self.random_state = random_state
    self.sparsity = sparsity

    self.teacher_forcing = teacher_forcing
    self.teacher_scaling = teacher_scaling
    self.teacher_shift = teacher_shift
    self.feedback_scaling = feedback_scaling
    self.input_shift = correct_dimensions(input_shift, n_inputs)
    self.input_scaling = correct_dimensions(input_scaling, n_inputs)
    
    self.out_activation = identity
    self.inverse_out_activation = identity 

    if isinstance(random_state, np.random.RandomState):
        self.random_state_ = random_state
    elif random_state:
        try:
            self.random_state_ = np.random.RandomState(random_state)
        except TypeError as e:
            raise Exception("Invalid seed: " + str(e))
    else:
        self.random_state_ = np.random.mtrand._rand
    
    self.init_weights()

    self.use_gradient_descent = use_gradient_descent
    self.learning_rate = learning_rate
    self.l2_rate = l2_rate

    self.use_rls = use_rls
    self.forget_rate = forget_rate

    self.n_epochs = n_epochs
    self.batch_size = batch_size


  def init_weights(self) : 

    self.x_0 = np.zeros(self.n_inputs)
    self.y_0 = np.zeros(self.n_outputs)

    self.W_in = np.empty((self.n_reservoir, self.n_inputs))
    self.W = np.empty((self.n_reservoir, self.n_reservoir))
    self.W_back = np.empty((self.n_reservoir, self.n_outputs))
    self.W_bias = np.zeros(self.n_reservoir)
  
    # initialize recurrent weights:
    # begin with a random matrix centered around zero:
    W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5

    # delete the fraction of connections given by (self.sparsity):
    W[self.random_state_.rand(*W.shape) < self.sparsity] = 0
    
    # compute the spectral radius of these weights:
    radius = np.max(np.abs(np.linalg.eigvals(W)))

    # rescale them to reach the requested spectral radius:
    self.W_rec = W * (self.spectral_radius / radius)
    
    # random input weights:
    self.W_in = self.random_state_.rand(
        self.n_reservoir, self.n_inputs) * 2 - 1
    
    # random feedback (teacher forcing) weights:
    self.W_back = self.random_state_.rand(
        self.n_reservoir, self.n_outputs) * 2 - 1
  

  def one_step(self, state, input, output):

    if self.teacher_forcing :
      preact = np.dot(self.W_rec, state) + np.dot(self.W_in, input) + np.dot(self.W_back, output) 
    else :
      preact = np.dot(self.W_rec, state) + np.dot(self.W_in, input)
    
    return forward(self, preact, self.alpha) + self.noise * (self.random_state_.rand(self.n_reservoir) - 0.5) 


  def _scale_inputs(self, inputs):
    """for each input dimension j: multiplies by the j'th entry in the
    input_scaling argument, then adds the j'th entry of the input_shift
    argument."""
    if self.input_scaling is not None:
      inputs = np.dot(inputs, np.diag(self.input_scaling))
    if self.input_shift is not None:
      inputs = inputs + self.input_shift
    return inputs


  def _scale_teacher(self, teacher):
    """multiplies the teacher/target signal by the teacher_scaling argument,
    then adds the teacher_shift argument to it."""
    if self.teacher_scaling is not None:
      teacher = teacher * self.teacher_scaling
    if self.teacher_shift is not None:
      teacher = teacher + self.teacher_shift
    return teacher


  def _unscale_teacher(self, teacher_scaled):
    """inverse operation of the _scale_teacher method."""
    if self.teacher_shift is not None:
      teacher_scaled = teacher_scaled - self.teacher_shift
    if self.teacher_scaling is not None:
      teacher_scaled = teacher_scaled / self.teacher_scaling
    return teacher_scaled


  def fit(self, inputs, outputs, inspect=False):
    if inputs.ndim < 2:
      inputs = np.reshape(inputs, (len(inputs), -1))
    if outputs.ndim < 2:
      outputs = np.reshape(outputs, (len(outputs), -1))
    # transform input and teacher signal:
    inputs_scaled = self._scale_inputs(inputs)
    teachers_scaled = self._scale_teacher(outputs)

    states = np.zeros((inputs.shape[0], self.n_reservoir))

    print("computing states ...")
    for n in range(1, inputs.shape[0]):
      states[n, :] = self.one_step(states[n-1, :], inputs_scaled[n-1, :], teachers_scaled[n-1, :]) + self.alpha*states[n-1, :]
   # including inputs 
    extended = np.hstack((states, inputs_scaled))

    print("computing W_out ...")

    if self.use_gradient_descent : 
      self.W_out = self.random_state_.rand(self.n_outputs, self.n_reservoir+self.n_inputs) * 2 - 1
      
      for epoch in range(self.n_epochs) : 
        for n in range(1, inputs.shape[0], self.batch_size):
          ext = np.hstack((states[n : min(n+self.batch_size, len(states)),:], inputs_scaled[n : min(n+self.batch_size, len(states)),:]))
          self.W_out = self.W_out*(1 - self.learning_rate*self.l2_rate)
          self.W_out += self.learning_rate * (teachers_scaled[n : min(n+self.batch_size, len(states)),:].T - self.W_out@ext.T) @ ext

    elif self.use_rls : 
      self.W_out = np.zeros((self.n_outputs, self.n_reservoir+self.n_inputs))
      P = np.diag([self.l2_rate for k in range(self.batch_size)])

      for epoch in range(self.n_epochs) : 
        for n in range(1, inputs.shape[0], self.batch_size):
          X = states[n : min(n+self.batch_size, len(states)),:]

          if len(X) != self.batch_size : 
            break
            
          Y = teachers_scaled[n : min(n+self.batch_size, len(states)), :]
          ext = np.hstack((X, inputs_scaled[n : min(n+self.batch_size, len(states)),:]))
          K = P@ext @ np.power(self.forget_rate + ext.T @ P @ ext, -1)
          xi = Y.T - self.W_out @ ext.T
          self.W_out += xi@K
          P = self.forget_rate*P - self.forget_rate * K @ ext.T @ P

    else : 
      self.W_out = np.dot(np.linalg.pinv(extended), 
                            self.inverse_out_activation(teachers_scaled)).T

    self.laststate = states[-1,:]
    self.lastinput = inputs[-1,:]
    self.lastoutput = outputs[-1,:]

    print("Training error ...")
    pred_train = self._unscale_teacher(self.out_activation(
        np.dot( extended, self.W_out.T)))
    print(np.sqrt(np.mean((pred_train - outputs)**2)))
    return pred_train


  def score(self, u, v):
    return np.sqrt(np.mean((u - v)**2))


  def predict(self, inputs, continuation=True):
    # if continuation = True, start the RcNN from the last training state

    n_samples = inputs.shape[0]

    if continuation == True : 
      laststate = self.laststate 
      lastinput = self.lastinput
      lastoutput = self.lastoutput 
    
    else :
      laststate = np.zeros(self.n_reservoir)
      lastinput = np.zeros(self.n_inputs)
      lastoutput = np.zeros(self.n_outputs)

    inputs = np.vstack([lastinput, self._scale_inputs(inputs)])
    states = np.vstack([laststate, np.zeros((n_samples, self.n_reservoir))])
    outputs = np.vstack([lastoutput, np.zeros((n_samples, self.n_outputs))])

    for n in range(n_samples): 
      states[n+1, :] = self.one_step(states[n, :], inputs[n+1, :], outputs[n, :])
      outputs[n+1, :] = self.out_activation(np.dot(self.W_out, np.concatenate([states[n+1, :], inputs[n+1, :]])))

    column_preds = self._unscale_teacher(self.out_activation(outputs[1:, :]))
    
    preds = []
    n_images = n_samples // self.n_inputs
    for j in range(n_images) : 
      column_labels = np.argmax(column_preds[j*self.n_inputs : (j+1)*self.n_inputs], axis=1)
      label = np.argmax(np.bincount(column_labels))
      pred = np.zeros((self.n_outputs))
      pred[label] = 1
      preds.append(pred)
    
    return np.array(preds)




class ESNEstimator() : 

  def __init__(self, n_inputs, n_outputs, 
               n_reservoir=400, alpha=0.0001, spectral_radius=1, sparsity=0.5, noise=0.001,
               input_scaling=None, input_shift=None, 
               teacher_scaling=None, teacher_shift=None,
               teacher_forcing=False, feedback_scaling=None, 
               out_activation=identity, inverse_out_activation=identity,
               random_state=None, 
               use_gradient_descent=False, learning_rate=0.0001, l2_rate=0.001, n_epochs=1, batch_size=1, 
               use_rls=False, forget_rate=0.5) :
    
    # spectral radius = spectral radius of W_rec
    # sparsity = proportion of recurrent weights set to zero
    # noise = noise added to each neuron (regularization)
    # random_state = positive seed
    # input_shift: scalar or vector of length n_inputs to add to each input dimension before feeding it to the network.
    # input_scaling: scalar or vector of length n_inputs to multiply with each input dimension before feeding it to the netw.
    # teacher_forcing: if True, feed the target back into output units
    # teacher_scaling: factor applied to the target signal
    # teacher_shift: additive term applied to the target signal
    # out_activation: output activation function (applied to the readout)
    # inverse_out_activation: inverse of the output activation function
    
    self.n_inputs = n_inputs
    self.n_outputs = n_outputs
    self.n_reservoir = n_reservoir
     
    self.noise = noise
    self.spectral_radius = spectral_radius
    self.alpha = alpha
    self.random_state = random_state
    self.sparsity = sparsity

    self.teacher_forcing = teacher_forcing
    self.teacher_scaling = teacher_scaling
    self.teacher_shift = teacher_shift
    self.feedback_scaling = feedback_scaling
    self.input_shift = input_shift
    self.input_scaling = input_scaling
    
    self.out_activation = out_activation
    self.inverse_out_activation = inverse_out_activation 

    if isinstance(self.random_state, np.random.RandomState):
        self.random_state_ = self.random_state
    elif random_state:
        try:
            self.random_state_ = np.random.RandomState(self.random_state)
        except TypeError as e:
            raise Exception("Invalid seed: " + str(e))
    else:
        self.random_state_ = np.random.mtrand._rand
    
    self.use_gradient_descent = use_gradient_descent
    self.learning_rate = learning_rate
    self.l2_rate = l2_rate

    self.use_rls = use_rls
    self.forget_rate = forget_rate

    self.n_epochs = n_epochs
    self.batch_size = batch_size

    #self.init_weights()

  def get_params(self, deep=True):
        # Return the hyperparameters as a dictionary
        return {'n_inputs': self.n_inputs, 'n_outputs': self.n_outputs, 'n_reservoir': self.n_reservoir, 
                'alpha': self.alpha, 'spectral_radius': self.spectral_radius, 'sparsity': self.sparsity, 
                'noise': self.noise, 'input_scaling': self.input_scaling, 
                'input_shift': self.input_shift, 'teacher_forcing': self.teacher_forcing, 
                'teacher_scaling': self.teacher_scaling, 'teacher_shift': self.teacher_shift, 
                'feedback_scaling': self.feedback_scaling, 'out_activation': self.out_activation, 
                'inverse_out_activation': self.inverse_out_activation, 'random_state': self.random_state, 
                'use_gradient_descent': self.use_gradient_descent, 'learning_rate': self.learning_rate, 'l2_rate': self.l2_rate, 
                'n_epochs': self.n_epochs, 'batch_size': self.batch_size, 
                'use_rls' : self.use_rls, 'forget_rate' : self.forget_rate}

  def set_params(self, **params):
        # Set the hyperparameters

        self.n_reservoir = params['n_reservoir']
        self.alpha = params['alpha']
        self.spectral_radius = params['spectral_radius']
        self.sparsity = params['sparsity']
        self.noise = params['noise']
        self.input_scaling = params['input_scaling']
        self.input_shift = params['input_scaling']
        self.teacher_forcing = params['teacher_forcing']
        self.teacher_scaling = params['teacher_scaling']
        self.teacher_shift = params['teacher_shift']
        self.feedback_scaling = params['feedback_scaling']
        self.out_activation = params['out_activation']
        self.inverse_out_activation = params['inverse_out_activation']
        self.random_state = params['random_state']
        self.use_gradient_descent = params['use_gradient_descent']
        self.learning_rate = params['learning_rate']
        self.l2_rate = params['l2_rate']
        self.use_rls = params['use_rls']
        self.forget_rate = params['forget_rate']
        self.n_epochs = params['n_epochs']
        self.batch_size = params['batch_size']
        return self

  def __getstate__(self):
      # Return a dictionary of instance attributes to be pickled
      return self.__dict__

  def __setstate__(self, state):
      # Restore instance attributes from pickled state
      self.__dict__ = state

  def init_weights(self) : 

    self.x_0 = np.zeros(self.n_inputs)
    self.y_0 = np.zeros(self.n_outputs)

    self.W_in = np.empty((self.n_reservoir, self.n_inputs))
    self.W = np.empty((self.n_reservoir, self.n_reservoir))
    self.W_back = np.empty((self.n_reservoir, self.n_outputs))
    self.W_bias = np.zeros(self.n_reservoir)
  
    # initialize recurrent weights:
    # begin with a random matrix centered around zero:
    W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5

    # delete the fraction of connections given by (self.sparsity):
    W[self.random_state_.rand(*W.shape) < self.sparsity] = 0
    
    # compute the spectral radius of these weights:
    radius = np.max(np.abs(np.linalg.eigvals(W)))

    # rescale them to reach the requested spectral radius:
    self.W_rec = W * (self.spectral_radius / radius)
    
    # random input weights:
    self.W_in = self.random_state_.rand(
        self.n_reservoir, self.n_inputs) * 2 - 1
    
    # random feedback (teacher forcing) weights:
    self.W_back = self.random_state_.rand(
        self.n_reservoir, self.n_outputs) * 2 - 1
  
  def one_step(self, state, input, output):

    if self.teacher_forcing :
      preact = np.dot(self.W_rec, state) + np.dot(self.W_in, input) + np.dot(self.W_back, output) 
    
    else :
      preact = np.dot(self.W_rec, state) + np.dot(self.W_in, input)
    
    return forward(self, preact, self.alpha) + self.noise * (self.random_state_.rand(self.n_reservoir) - 0.5)

  def _scale_inputs(self, inputs):
    """for each input dimension j: multiplies by the j'th entry in the
    input_scaling argument, then adds the j'th entry of the input_shift
    argument."""
    if self.input_scaling is not None:
      inputs = np.dot(inputs, np.diag(self.input_scaling))
    if self.input_shift is not None:
      inputs = inputs + self.input_shift
    return inputs

  def _scale_teacher(self, teacher):
    """multiplies the teacher/target signal by the teacher_scaling argument,
    then adds the teacher_shift argument to it."""
    if self.teacher_scaling is not None:
      teacher = teacher * self.teacher_scaling
    if self.teacher_shift is not None:
      teacher = teacher + self.teacher_shift
    return teacher

  def _unscale_teacher(self, teacher_scaled):
    """inverse operation of the _scale_teacher method."""
    if self.teacher_shift is not None:
      teacher_scaled = teacher_scaled - self.teacher_shift
    if self.teacher_scaling is not None:
      teacher_scaled = teacher_scaled / self.teacher_scaling
    return teacher_scaled

  def fit(self, inputs, outputs, inspect=False):

    self.init_weights()
    self.input_shift = correct_dimensions(self.input_shift, self.n_inputs)
    self.input_scaling = correct_dimensions(self.input_scaling, self.n_inputs)

    if inputs.ndim < 2:
      inputs = np.reshape(inputs, (len(inputs), -1))
    if outputs.ndim < 2:
      outputs = np.reshape(outputs, (len(outputs), -1))
    # transform input and teacher signal:
    inputs_scaled = self._scale_inputs(inputs)
    teachers_scaled = self._scale_teacher(outputs)

    if self.use_gradient_descent : 
      self.W_out = self.random_state_.rand(self.n_outputs, self.n_reservoir+self.n_inputs) * 2 - 1

    states = np.zeros((inputs.shape[0], self.n_reservoir))

    print("computing states ...")
    for n in range(1, inputs.shape[0]):
      states[n, :] = self.one_step(states[n-1, :], inputs_scaled[n, :], teachers_scaled[n-1, :]) + (self.alpha)*states[n-1, :]

    # including inputs 
    extended = np.hstack((states, inputs_scaled))
    acc = []
    acc_rls = []
    print("computing W_out ...")
    if self.use_gradient_descent : 
      self.W_out = self.random_state_.rand(self.n_outputs, self.n_reservoir+self.n_inputs) * 2 - 1
      for epoch in range(self.n_epochs) : 
        for n in range(1, inputs.shape[0], self.batch_size):
          ext = np.hstack((states[n : min(n+self.batch_size, len(states)),:], inputs_scaled[n : min(n+self.batch_size, len(states)),:]))
          self.W_out = self.W_out*(1 - self.learning_rate*self.l2_rate)
          self.W_out += self.learning_rate * (teachers_scaled[n : min(n+self.batch_size, len(states)),:].T - self.W_out@ext.T) @ ext
          pred_train = self._unscale_teacher(self.out_activation(np.dot( extended, self.W_out.T)))
        acc.append(np.sqrt(np.linalg.norm((pred_train - outputs)**2, ord=2, keepdims=False)))
    
    elif self.use_rls : 
      self.W_out = np.zeros((self.n_outputs, self.n_reservoir+self.n_inputs))
      P = np.diag([self.l2_rate for k in range(self.batch_size)])
      for epoch in range(self.n_epochs) : 
        for n in range(1, inputs.shape[0], self.batch_size):
          X = states[n : min(n+self.batch_size, len(states)),:]
          if len(X) != self.batch_size : 
            break
          Y = teachers_scaled[n : min(n+self.batch_size, len(states)), :]
          ext = np.hstack((X, inputs_scaled[n : min(n+self.batch_size, len(states)),:]))
          K = P@ext @ np.power(self.forget_rate + ext.T @ P @ ext, -1)
          xi = Y.T - self.W_out @ ext.T
          self.W_out += xi@K
          P = self.forget_rate*P - self.forget_rate * K @ ext.T @ P
        acc_rls.append(self._unscale_teacher(self.out_activation(np.dot(ext, self.W_out.T))))

    else : 
      self.W_out = np.dot(np.linalg.pinv(extended), 
                            self.inverse_out_activation(teachers_scaled)).T

    self.laststate = states[-1,:]
    self.lastinput = inputs[-1,:]
    self.lastoutput = outputs[-1,:]

    
  def score(self, u, v):
    return np.sqrt(np.linalg.norm((u-v)**2, ord=2, keepdims=False))

  def predict(self, inputs, continuation=True):
    # if continuation = True, start the RcNN from the last training state

    n_samples = inputs.shape[0]

    if continuation == True : 
      laststate = self.laststate 
      lastinput = self.lastinput
      lastoutput = self.lastoutput 
    
    else :
      laststate = np.zeros(self.n_reservoir)
      lastinput = np.zeros(self.n_inputs)
      lastoutput = np.zeros(self.n_outputs)

    inputs = np.vstack([lastinput, self._scale_inputs(inputs)])
    states = np.vstack([laststate, np.zeros((n_samples, self.n_reservoir))])
    outputs = np.vstack([lastoutput, np.zeros((n_samples, self.n_outputs))])

    for n in range(n_samples): 
      states[n+1, :] = self.one_step(states[n, :], inputs[n+1, :], outputs[n, :]) + self.alpha*states[n, :]
      outputs[n+1, :] = self.out_activation(np.dot(self.W_out, np.concatenate([states[n+1, :], inputs[n+1, :]])))

    #return self._unscale_teacher(self.out_activation(outputs[1:]))

    column_preds = self._unscale_teacher(self.out_activation(outputs[1:]))
    preds = []
    n_images = n_samples // self.n_inputs
    for j in range(n_images) : 
      column_labels = np.argmax(column_preds[j*self.n_inputs : (j+1)*self.n_inputs], axis=1)
      label = np.argmax(np.bincount(column_labels))
      preds.append(label)
    
    return np.array(preds)