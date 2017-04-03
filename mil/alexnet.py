from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

#from keras.layers.core import Flatten
#from keras.models import Sequential, Model
#from keras.layers import Dense, Dropout, Reshape, Permute, Activation, \
#    Input, merge
#from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
#from keras.optimizers import SGD
#import numpy as np
#from scipy.misc import imread, imresize, imsave
#from keras.engine import Layer
#from keras.layers.core import  Lambda#, Merge

#from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
#    splittensor, Softmax4D, Recalc, ReRank, ExtractDim, SoftReRank, ActivityRegularizerOneDim, RecalcExpand
#from convnetskeras.imagenet_tool import synset_to_id, id_to_synset,synset_to_dfs_ids
#from keras.regularizers import l1l2, activity_l1l2

#from keras import backend as K

#def crosschannelnormalization(alpha = 1e-4, k=2, beta=0.75, n=5,**kwargs):
#    """
#    This is the function used for cross channel normalization in the original
#    Alexnet
#    """
#    def f(X):
#        b, ch, r, c = X.shape
#        half = n // 2
#        square = K.square(X)
#        extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0,2,3,1))
#                                              , (0,half))
#        extra_channels = K.permute_dimensions(extra_channels, (0,3,1,2))
#        scale = k
#        for i in range(n):
#            scale += alpha * extra_channels[:,i:i+ch,:,:]
#        scale = scale ** beta
#        return X / scale
#
#    return Lambda(f, output_shape=lambda input_shape:input_shape,**kwargs)

def pretrained(weights_file):
    model = definition()
    model.load_weights(weights_file)

def definition(l1factor=0, l2factor=0):
    model = Sequential()

    model.add(Conv2D(96, (11, 11), strides=(4,4), input_shape=(3, 227, 227)))
    model.add(BatchNormalization((64,226,226)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))
    
    model.add(Convolution2D(128, 5, 5, subsample(1,1), border_mode='full'))
    model.add(BatchNormalization((128,115,115)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))
    
    model.add(Convolution2D(192, 128, 3, 3, border_mode='full'))
    model.add(BatchNormalization((128,112,112)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))
    
    model.add(Convolution2D(256, 192, 3, 3, border_mode='full'))
    model.add(BatchNormalization((128,108,108)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))
    
    model.add(Flatten())
    model.add(Dense(12*12*256, 4096, init='normal'))
    model.add(BatchNormalization(4096))
    model.add(Activation('relu'))
    model.add(Dense(4096, 4096, init='normal'))
    model.add(BatchNormalization(4096))
    model.add(Activation('relu'))
    model.add(Dense(4096, 1000, init='normal'))
    model.add(BatchNormalization(1000))
    model.add(Activation('softmax'))
   
    return model


###    inputs = Input(shape=(3,227,227))
###
###    conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor),
###                           name='conv_1')(inputs)
###
###    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
###    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
###    conv_2 = ZeroPadding2D((2,2))(conv_2)
###    conv_2 = merge([
###        Convolution2D(128,5,5,activation="relu",name='conv_2_'+str(i+1), W_regularizer=l1l2(l1=l1factor, l2=l2factor))(
###            splittensor(ratio_split=2,id_split=i)(conv_2)
###        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")
###
###    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
###    conv_3 = crosschannelnormalization()(conv_3)
###    conv_3 = ZeroPadding2D((1,1))(conv_3)
###    conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3', W_regularizer=l1l2(l1=l1factor, l2=l2factor))(conv_3)
###
###    conv_4 = ZeroPadding2D((1,1))(conv_3)
###    conv_4 = merge([
###        Convolution2D(192,3,3,activation="relu",name='conv_4_'+str(i+1), W_regularizer=l1l2(l1=l1factor, l2=l2factor))(
###            splittensor(ratio_split=2,id_split=i)(conv_4)
###        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")
###
###    conv_5 = ZeroPadding2D((1,1))(conv_4)
###    conv_5 = merge([
###        Convolution2D(128,3,3,activation="relu",name='conv_5_'+str(i+1), W_regularizer=l1l2(l1=l1factor, l2=l2factor))(
###            splittensor(ratio_split=2,id_split=i)(conv_5)
###        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")
###
###    dense_1 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)
###
#####    if heatmap:
#####        dense_1 = Convolution2D(4096,6,6,activation="relu",name="dense_1",W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
#####        dense_2 = Convolution2D(4096,1,1,activation="relu",name="dense_2",W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
#####        dense_3 = Convolution2D(outdim, 1,1,name="dense_3",W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_2)
#####        prediction = Softmax4D(axis=1,name="softmax")(dense_3)
#####    elif usemil:
#####        dense_1 = Convolution2D(128,1,1,activation='relu',name='mil_1',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
#####        dense_2 = Convolution2D(128,1,1,activation='relu',name='mil_2',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
#####        dense_3 = Convolution2D(outdim,1,1,name='mil_3',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_2)
#####        prediction_1 = Softmax4D(axis=1, name='softmax')(dense_3)
#####        #prediction = Flatten(name='flatten')(prediction_1)
#####        #dense_3 = Dense(outdim,name='dense_3',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(prediction)
#####        #prediction = Activation("softmax",name="softmax2")(dense_3)
#####        
#####        prediction_1 = MaxPooling2D((6,6), name='output')(prediction_1)
#####        prediction = Flatten(name='flatten')(prediction_1)
#####        prediction = Recalc(axis=1, name='Recalcmil')(prediction)
#####    elif usemymil:
#####        dense_1 = Convolution2D(128,1,1,activation='relu',name='mil_1',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
#####        dense_2 = Convolution2D(128,1,1,activation='relu',name='mil_2',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
#####        dense_3 = Convolution2D(1,1,1,activation='sigmoid',name='mil_3',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_2)
#####        #prediction_1 = Softmax4D(axis=1, name='softmax')(dense_3)
#####        #prediction = ExtractDim(axis=1, name='extract')(prediction_1)
#####        prediction = Flatten(name='flatten')(dense_3)
#####        prediction = ReRank(k=k, label=1, name='output')(prediction)
#####    elif usemysoftmil:
#####        dense_1 = Convolution2D(128,1,1,activation='relu',name='mil_1',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
#####        dense_2 = Convolution2D(128,1,1,activation='relu',name='mil_2',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
#####        dense_3 = Convolution2D(1,1,1,activation='sigmoid',name='mil_3',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_2)
#####        #prediction_1 = Softmax4D(axis=1, name='softmax')(dense_3)
#####        #prediction = ExtractDim(axis=1, name='extract')(prediction_1)
#####        prediction = Flatten(name='flatten')(dense_3)
#####        prediction = SoftReRank(softmink=softmink, softmaxk=softmaxk, label=1, name='output')(prediction)
#####    elif sparsemil:
#####        dense_1 = Convolution2D(128,1,1,activation='relu',name='mil_1',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
#####        dense_2 = Convolution2D(128,1,1,activation='relu',name='mil_2',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
#####        prediction_1 = Convolution2D(1,1,1,activation='sigmoid',name='mil_3',W_regularizer=l1l2(l1=l1factor, l2=l2factor),\
#####            activity_regularizer=activity_l1l2(l1=sparsemill1, l2=sparsemill2))(dense_2)
####        prediction_1 = Softmax4D(axis=1, name='softmax')(prediction_1)
###        #dense_3 = Convolution2D(outdim,1,1,name='mil_3',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_2)
###        #prediction_1 = Softmax4D(axis=1, name='softmax')(dense_3)
###        #prediction_1 = ActivityRegularizerOneDim(l1=sparsemill1, l2=sparsemill2)(prediction_1)
###        #prediction = MaxPooling2D((6,6), name='output')(prediction_1)
####        prediction_1 = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same', name='smooth', \
####            W_regularizer=l1l2(l1=l1factor, l2=l2factor), activity_regularizer=activity_l1l2(l1=sparsemill1, l2=sparsemill2))(prediction_1)
#####        prediction = Flatten(name='flatten')(prediction_1)
#####        if saveact:
#####          model = Model(input=inputs, output=prediction)
#####          return model
#####        prediction = RecalcExpand(axis=1, name='Recalcmil')(prediction)
#####    else:
###    dense_1 = Flatten(name="flatten")(dense_1)
###    dense_1 = Dense(4096, activation='relu',name='dense_1',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
###    dense_2 = Dropout(0.5)(dense_1)
###    dense_2 = Dense(4096, activation='relu',name='dense_2',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_2)
###    dense_3 = Dropout(0.5)(dense_2)
###    dense_3 = Dense(outdim,name='dense_3',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_3)
###    prediction = Activation("softmax",name="softmax")(dense_3)
###
###    model = Model(input=inputs, output=prediction)
###
###    if weights_path:
###        model.load_weights(weights_path)
###
###    return model
###
###
###
###class Regularizer(object):
###    """Regularizer base class.
###    """
###
###    def __call__(self, x):
###        return 0
###
###    def get_config(self):
###        return {'name': self.__class__.__name__}
###
###    def set_param(self, _):
###        warnings.warn('The `set_param` method on regularizers is deprecated. '
###                      'It no longer does anything, '
###                      'and it will be removed after 06/2017.')
###
###    def set_layer(self, _):
###        warnings.warn('The `set_layer` method on regularizers is deprecated. '
###                      'It no longer does anything, '
###                      'and it will be removed after 06/2017.')
###
###
###class EigenvalueRegularizer(Regularizer):
###    """Regularizer based on the eignvalues of a weight matrix.
###
###    Only available for tensors of rank 2.
###
###    # Arguments
###        k: Float; modulates the amount of regularization to apply.
###    """
###
###    def __init__(self, k):
###        self.k = k
###
###    def __call__(self, x):
###        if K.ndim(x) != 2:
###            raise ValueError('EigenvalueRegularizer '
###                             'is only available for tensors of rank 2.')
###        covariance = K.dot(K.transpose(x), x)
###        dim1, dim2 = K.eval(K.shape(covariance))
###
###        # Power method for approximating the dominant eigenvector:
###        power = 9  # Number of iterations of the power method.
###        o = K.ones([dim1, 1])  # Initial values for the dominant eigenvector.
###        main_eigenvect = K.dot(covariance, o)
###        for n in range(power - 1):
###            main_eigenvect = K.dot(covariance, main_eigenvect)
###        covariance_d = K.dot(covariance, main_eigenvect)
###
###        # The corresponding dominant eigenvalue:
###        main_eigenval = (K.dot(K.transpose(covariance_d), main_eigenvect) /
###                         K.dot(K.transpose(main_eigenvect), main_eigenvect))
###        # Multiply by the given regularization gain.
###        regularization = (main_eigenval ** 0.5) * self.k
###        return K.sum(regularization)
###
###
###class L1L2Regularizer(Regularizer):
###    """Regularizer for L1 and L2 regularization.
###
###    # Arguments
###        l1: Float; L1 regularization factor.
###        l2: Float; L2 regularization factor.
###    """
###
###    def __init__(self, l1=0., l2=0.):
###        self.l1 = K.cast_to_floatx(l1)
###        self.l2 = K.cast_to_floatx(l2)
###
###    def __call__(self, x):
###        regularization = 0
###        if self.l1:
###            regularization += K.sum(self.l1 * K.abs(x))
###        if self.l2:
###            regularization += K.sum(self.l2 * K.square(x))
###        return regularization
###
###    def get_config(self):
###        return {'name': self.__class__.__name__,
###                'l1': float(self.l1),
###                'l2': float(self.l2)}
###
###
#### Aliases.
###
###WeightRegularizer = L1L2Regularizer
###ActivityRegularizer = L1L2Regularizer
###
###
###def l1(l=0.01):
###    return L1L2Regularizer(l1=l)
###
###
###def l2(l=0.01):
###    return L1L2Regularizer(l2=l)
###
###
###def l1l2(l1=0.01, l2=0.01):
###    return L1L2Regularizer(l1=l1, l2=l2)
###
###
###def activity_l1(l=0.01):
###    return L1L2Regularizer(l1=l)
###
###
###def activity_l2(l=0.01):
###    return L1L2Regularizer(l2=l)
###
###
###def activity_l1l2(l1=0.01, l2=0.01):
###    return L1L2Regularizer(l1=l1, l2=l2)
###
###
###def get(identifier, kwargs=None):
###    return get_from_module(identifier, globals(), 'regularizer',
###                           instantiate=True, kwargs=kwargs)
###
###def crosschannelnormalization(alpha = 1e-4, k=2, beta=0.75, n=5,**kwargs):
###    """
###    This is the function used for cross channel normalization in the original
###    Alexnet
###    """
###    def f(X):
###        b, ch, r, c = X.shape
###        half = n // 2
###        square = K.square(X)
###        extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0,2,3,1))
###                                              , padding=((0,0),(half, half)))
###        extra_channels = K.permute_dimensions(extra_channels, (0,3,1,2))
###        scale = k
###        for i in range(n):
###            scale += alpha * extra_channels[:,i:i+ch,:,:]
###        scale = scale ** beta
###        return X / scale
###
###    return Lambda(f, output_shape=lambda input_shape:input_shape,**kwargs)
###
###
###
###def splittensor(axis=1, ratio_split=1, id_split=0,**kwargs):
###    def f(X):
###        div = X.shape[axis] // ratio_split
###
###        if axis == 0:
###            output =  X[id_split*div:(id_split+1)*div,:,:,:]
###        elif axis == 1:
###            output =  X[:, id_split*div:(id_split+1)*div, :, :]
###        elif axis == 2:
###            output = X[:,:,id_split*div:(id_split+1)*div,:]
###        elif axis == 3:
###            output == X[:,:,:,id_split*div:(id_split+1)*div]
###        else:
###            raise ValueError("This axis is not possible")
###
###        return output
###
###    def g(input_shape):
###        output_shape=list(input_shape)
###        output_shape[axis] = output_shape[axis] // ratio_split
###        return tuple(output_shape)
###
###    return Lambda(f,output_shape=lambda input_shape:g(input_shape),**kwargs)
###
###
###
###
####def convolution2Dgroup(n_group, nb_filter, nb_row, nb_col, **kwargs):
####    def f(input):
####        return Merge([
####            Convolution2D(nb_filter//n_group,nb_row,nb_col)(
####                splittensor(axis=1,
####                            ratio_split=n_group,
####                            id_split=i)(input))
####            for i in range(n_group)
####        ],mode='concat',concat_axis=1)
####
####    return f
###
###
###class Softmax4D(Layer):
###    def __init__(self, axis=-1,**kwargs):
###        self.axis=axis
###        super(Softmax4D, self).__init__(**kwargs)
###
###    def build(self,input_shape):
###        pass
###
###    def call(self, x,mask=None):
###        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
###        s = K.sum(e, axis=self.axis, keepdims=True)
###        return e / s
###
###    def get_output_shape_for(self, input_shape):
###        return input_shape
###        #axis_index = self.axis % len(input_shape)
###        #return tuple([input_shape[i] for i in range(len(input_shape)) \
###        #              if i != axis_index ])
###
###class Recalc(Layer):
###    def __init__(self, axis=-1,**kwargs):
###        self.axis=axis
###        super(Recalc, self).__init__(**kwargs)
###
###    def build(self,input_shape):
###        pass
###
###    def call(self, x,mask=None):
###        response = K.reshape(x[:,self.axis], (-1,1))
###        return K.concatenate([1-response, response], axis=self.axis)
###        #e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
###        #s = K.sum(e, axis=self.axis, keepdims=True)
###        #return e / s
###
###    def get_output_shape_for(self, input_shape):
###        return input_shape
###        #axis_index = self.axis % len(input_shape)
###        #return tuple([input_shape[i] for i in range(len(input_shape)) \
###        #              if i != axis_index ])
###
###class RecalcExpand(Layer):
###    def __init__(self, axis=-1,**kwargs):
###        self.axis=axis
###        super(RecalcExpand, self).__init__(**kwargs)
###
###    def build(self,input_shape):
###        pass
###
###    def call(self, x,mask=None):
###        response = K.max(x, axis=-1, keepdims=True) #K.reshape(x, (-1,1))
###        return K.concatenate([1-response, response], axis=self.axis)
###        #e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
###        #s = K.sum(e, axis=self.axis, keepdims=True)
###        #return e / s
###
###    def get_output_shape_for(self, input_shape):
###        return tuple([input_shape[0], 2])
###
###class ExtractDim(Layer):
###    def __init__(self, axis=-1,**kwargs):
###        self.axis=axis
###        super(ExtractDim, self).__init__(**kwargs)
###
###    def build(self,input_shape):
###        pass
###
###    def call(self, x,mask=None): # batchsize*2*6*6
###        return x[:,self.axis,:,:]
###
###    def get_output_shape_for(self, input_shape):
###        return tuple([input_shape[0],1,input_shape[2],input_shape[3]])
###
###class ReRank(Layer):
###    # Rerank is difficult. It is equal to the number of points (>0.5) to be fixed number.
###    def __init__(self,k=1,label=1,**kwargs):
###        # k is the factor we force to be 1
###        self.k = k*1.0
###        self.label = label
###        super(ReRank, self).__init__(**kwargs)
###
###    def build(self,input_shape):
###        pass
###
###    def call(self, x,mask=None):
###        import theano.tensor as T
###        newx = T.sort(x)
###        #response = K.reverse(newx, axes=1)
###        #response = K.sum(x> 0.5, axis=1) / self.k
###        return newx
###        #response = K.reshape(newx,[-1,1])
###        #return K.concatenate([1-response, response], axis=self.label)
###        #response = K.reshape(x[:,self.axis], (-1,1))
###        #return K.concatenate([1-response, response], axis=self.axis)
###        #e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
###        #s = K.sum(e, axis=self.axis, keepdims=True)
###        #return e / s
###
###    def get_output_shape_for(self, input_shape):
###        #return tuple([input_shape[0],input_shape[1],2])
###        return input_shape
###
###class SoftReRank(Layer):
###    # Rerank is difficult. It is equal to the number of points (>0.5) to be fixed number.
###    def __init__(self,softmink=1, softmaxk=1,label=1,**kwargs):
###        # k is the factor we force to be 1
###        self.softmink=softmink
###        self.softmaxk=softmaxk
###        self.label = label
###        super(SoftReRank, self).__init__(**kwargs)
###
###    def build(self,input_shape):
###        pass
###
###    def call(self, x,mask=None):
###        newx = K.sort(x)
###        #response = K.reverse(newx, axes=1)
###        #response = K.sum(x> 0.5, axis=1) / self.k
###        return K.concatenate([newx[:,:self.softmink], newx[:,newx.shape[1]-self.softmaxk:]], axis=-1)
###        #response = K.reshape(newx,[-1,1])
###        #return K.concatenate([1-response, response], axis=self.label)
###        #response = K.reshape(x[:,self.axis], (-1,1))
###        #return K.concatenate([1-response, response], axis=self.axis)
###        #e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
###        #s = K.sum(e, axis=self.axis, keepdims=True)
###        #return e / s
###
###    def get_output_shape_for(self, input_shape):
###        #return tuple([input_shape[0],input_shape[1],2])
###        return tuple([input_shape[0], self.softmink+self.softmaxk])
###
###class ActivityRegularizerOneDim(Regularizer):
###    def __init__(self, l1=0., l2=0.,**kwargs):
###        self.l1 = K.cast_to_floatx(l1)
###        self.l2 = K.cast_to_floatx(l2)
###        self.uses_learning_phase = True
###        super(ActivityRegularizerOneDim, self).__init__(**kwargs)
###        #self.layer = None
###
###    def set_layer(self, layer):
###        if self.layer is not None:
###            raise Exception('Regularizers cannot be reused')
###        self.layer = layer
###
###    def __call__(self, loss):
###        #if self.layer is None:
###        #    raise Exception('Need to call `set_layer` on '
###        #                    'ActivityRegularizer instance '
###        #                    'before calling the instance.')
###        regularized_loss = loss
###        for i in range(len(self.layer.inbound_nodes)):
###            output = self.layer.get_output_at(i)
###            if self.l1:
###                regularized_loss += K.sum(self.l1 * K.abs(output[:,:,:,1]))
###            if self.l2:
###                regularized_loss += K.sum(self.l2 * K.square(output[:,:,:,1]))
###        return K.in_train_phase(regularized_loss, loss)
###
###    def get_config(self):
###        return {'name': self.__class__.__name__,
###                'l1': float(self.l1),
###                'l2': float(self.l2)}
###
