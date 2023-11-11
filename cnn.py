import numpy as np


def im2col(X, filter_h, filter_w, padding=1, stride=1):
    '''
    Construct the im2col matrix of intput feature map X.
    X: 4D tensor of shape [N, C, H, W], input feature map
    k_height, k_width: height and width of convolution kernel
    return a 2D array of shape (C*k_height*k_width, H*W*N)
    The axes ordering need to be (C, k_height, k_width, H, W, N) here, while in
    reality it can be other ways if it weren't for autograding tests.
    '''
    # pass

    N, C, H, W = X.shape
    out_h = (H + 2 * padding - filter_h) // stride + 1
    out_w = (W + 2 * padding - filter_w) // stride + 1
    # print("out:", out_h, out_w, " input before pad if any:", X.shape)
    img = np.pad(X, [(0,0), (0,0), (padding, padding), (padding, padding)], 'constant')
    # print("img shape after padding:", img.shape)
    # col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    
    first = 1
    for y in range(out_h):
        y_max = y * stride + filter_h
        # print("ymax:", y, y_max)
        for x in range(out_w):
            x_max = x * stride + filter_w
            # print("xmax:", x, x_max)
            for i in range(N):
                if first == 1:
                    first = 0
                    # print(img[i, :, y* stride:y_max, x * stride:x_max])
                    col = img[i, :, y* stride:y_max, x * stride:x_max].reshape(filter_w * filter_h * C, -1)
                else:
                    # print(img[i, :, y* stride:y_max, x * stride:x_max])
                    new_img = img[i, :, y* stride:y_max, x* stride:x_max].reshape(filter_w * filter_h * C, -1)
                    col = np.hstack((col, new_img))
    # print("col_array:", col.shape)      
    # col = col.transpose(1, 2, 3, 0, 4, 5).reshape(C*filter_h*filter_w, -1)

    return col


def im2col_bw(grad_X_col, X_shape, filter_h, filter_w, padding=1, stride=1):
    '''
    Map gradient w.r.t. im2col output back to the feature map.
    grad_X_col: a 2D array
    return X_grad as a 4D array in X_shape
    '''
    # pass
    N, C, H, W = X_shape
    out_h = (H + 2*padding - filter_h)//stride + 1
    out_w = (W + 2*padding - filter_w)//stride + 1
    
    # grad_X = grad_X_col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*padding + stride - 1, W + 2*padding + stride - 1))
    # first = 1
    start_points = 0
    # print("img after padding:", img.shape, "inputs grad_X_col:", grad_X_col.shape)
    
    for y in range(out_h):
        y_max = y * stride + filter_h
        for x in range(out_w):
            x_max = x * stride + filter_w
            for i in range(N):
                # print(grad_X_col[:, i].reshape(C, filter_h, filter_w))
                # print(" img:",  img[i%N, :, y:y_max:stride, x:x_max:stride])
                # try:
                img[i%N, :, y*stride:y_max, x*stride:x_max] += grad_X_col[:, start_points+i].reshape(C, filter_h, filter_w)
            start_points+=N
            # img = np.asarray(img, dtype=int)
            # print(img)
            # print("next")
                # except:
                    # print("i:", i, " x_max:", x_max, " y_max:", y_max)

    # print("out:", out_h, out_w)
     
    return img[:, :, padding:H + padding, padding:W + padding]


class Transform:
    """
    This is the base class. You do not need to change anything.
    Read the comments in this class carefully.
    """
    def __init__(self):
        """
        Initialize any parameters
        """
        pass

    def forward(self, x):
        """
        x should be passed as column vectors
        """
        pass

    def backward(self, grad_wrt_out):
        """
        Unlike Problem 5 MLP, here we no longer accumulate the gradient values,
        we assign new gradients directly. This means we should call update()
        every time we do forward and backward, which is fine. Consequently, for
        Problem 6, zerograd() is not needed any more.
        Compute and save the gradients wrt the parameters for update()
        Read comments in each class to see what to return.
        """
        pass

    def update(self, learning_rate, momentum_coeff):
        """
        Apply gradients to update the parameters
        """
        pass


class ReLU(Transform):
    """
    Implement this class
    """
    def forward(self, x, train=True):
        """
        returns ReLU(x)
        """
        # pass
        self.act_values = np.clip(x, a_min=0, a_max=None)
        self.act_state = self.act_values
        return self.act_values

    def backward(self, dLoss_dout):
        """
        dLoss_dout is the gradients wrt the output of ReLU
        returns gradients wrt the input to ReLU
        """
        # pass
        d_act = np.where(self.act_state > 0, 1, self.act_state)
        dl = np.multiply(dLoss_dout, d_act)
        return dl     

class Flatten(Transform):
    """
    Implement this class which reshapes a multi-dimensional array input (batch_size, dim_1, dim_2, ...)
    to a 2-dimensional array (batch_size, dim1 x dim 2 x ...). Not Autograded but will be useful for 
    implementing the ConvNet and ConvNetThree classes.
    """
    def forward(self, x):
        """
        returns Flatten(x)
        """
        # pass
        self.x_shape = x.shape
        x_reshaped = x.reshape(x.shape[0], -1)
        return x_reshaped

    def backward(self, dloss):
        """
        dLoss is the gradients wrt the output of Flatten
        returns gradients wrt the input to Flatten
        """
        # pass
        reshaped_dloss = dloss.reshape(self.x_shape)
        return reshaped_dloss
        


class Conv(Transform):
    """
    Implement this class - Convolution Layer
    """
    def __init__(self, input_shape, filter_shape, rand_seed=0):
        """
        input_shape is a tuple: (channels, height, width)
        filter_shape is a tuple: (num of filters, filter height, filter width)
        weights shape (number of filters, number of input channels, filter height, filter width)
        Use Xavier initialization for weights, as instructed on handout
        Initialze biases as an array of zeros in shape of (num of filters, 1)
        """
        np.random.seed(rand_seed) # keep this line for autograding; you may remove it for training
        self.channel, self.height, self.width = input_shape
        self.num_filters, self.k_height, self.k_width = filter_shape
        b = np.sqrt(6) / np.sqrt((self.channel + self.num_filters) * self.k_height * self.k_width)
        self.W = np.random.uniform(-b, b, (self.num_filters, self.channel, self.k_height, self.k_width))
        self.b = np.zeros((self.num_filters, 1))
        self.momentum_W = np.zeros_like(self.W)
        self.momentum_b = np.zeros_like(self.b)

    def forward(self, inputs, stride=1, pad=2):
        """
        Forward pass of convolution between input and filters
        inputs is in the shape of (batch_size, num of channels, height, width)
        Return the output of convolution operation in shape (batch_size, num of filters, height, width)
        use im2col here to vectorize your computations
        """
        # pass
        self.x, self.stride, self.pad = inputs, stride, pad
        batch_size = inputs.shape[0]
        
        out_height = (self.height + 2 * pad - self.k_height) / stride + 1
        out_width = (self.width + 2 * pad - self.k_width) / stride + 1
        
        output = np.zeros((batch_size, self.num_filters, int(out_height), int(out_width)), dtype=inputs.dtype) #correct reshape
        
        self.x_col = im2col(inputs, self.k_height, self.k_width, pad, stride)
        
        w_col = self.W.reshape((self.num_filters, -1))
        output = np.dot(w_col, self.x_col) + self.b.reshape(-1, 1)
        # print("forward WO HO:", WO, HO)        
        
        output = output.reshape((self.num_filters, int(out_height), int(out_width), batch_size)) #correct reshape method
        output = output.transpose(3, 0, 1, 2)
        
        return output

    def backward(self, dloss):
        """
        Read Transform.backward()'s docstring in this file
        dloss shape (batch_size, num of filters, output height, output width)
        Return [gradient wrt weights, gradient wrt biases, gradient wrt input to this layer]
        use im2col_bw here to vectorize your computations
        """
        # pass
        self.db = np.sum(dloss, axis = (0, 2, 3)).reshape(-1, 1)
        # print("db:", self.db.shape)
        
        num_filters, num_channel, filter_height, filter_width = self.W.shape
        dloss_reshaped = dloss.transpose(1, 2, 3, 0).reshape(num_filters, -1) #correct reshape format
        # print("self.x_col:", self.x_col.shape, "dloss_reshape:", dloss_reshaped.shape)

        self.dW = np.dot(dloss_reshaped, self.x_col.T).reshape(self.W.shape) #correct reshape format 

        dx_cols = np.dot(self.W.reshape(dloss.shape[1], -1).T, dloss_reshaped)
        # print("dx_cols:", dx_cols.shape, "dw:", self.W.shape)
        
        dx = im2col_bw(dx_cols, self.x.shape, self.W.shape[2], self.W.shape[3], self.pad, self.stride)        
        
        return [self.dW, self.db, dx]

    def update(self, learning_rate=0.001, momentum_coeff=0.5):
        """
        Update weights and biases with gradients calculated by backward()
        Use the same momentum formula as in Problem 5.
        """
        # pass
        
        self.momentum_W = momentum_coeff * self.momentum_W - learning_rate * self.dW
        self.momentum_b = momentum_coeff * self.momentum_b - learning_rate * self.db.T
        
        self.W += self.momentum_W 
        self.b += self.momentum_b 
   
    def get_wb_conv(self):
        """
        Return weights and biases
        """
        return self.W, self.b


class MaxPool(Transform):
    """
    Implement this class - MaxPool layer
    """
    def __init__(self, filter_shape, stride):
        """
        filter_shape is (filter_height, filter_width)
        stride is a scalar
        """
        # pass
        self.pool_height = filter_shape[0]
        self.pool_width = filter_shape[1]
        self.stride = stride

    def forward(self, inputs):
        """
        forward pass of MaxPool
        inputs: (N, C, H, W)
        """
        # pass
        self.inputs = inputs
        batch_size, num_channel, inputs_height, inputs_width = self.inputs.shape
        reshaped_inputs = self.inputs.reshape(batch_size * num_channel, 1, inputs_height, inputs_width)
        
        output_height = (inputs_height - self.pool_height)/self.stride + 1
        output_width = (inputs_width - self.pool_width) / self.stride + 1
        
        # print("max output:", output_height, output_width)
        # print("filter shape:", self.pool_height, self.pool_width)
        # print("max reshaped_inputs:", reshaped_inputs.shape, inputs.shape, self.stride)
        
        self.x_col = im2col(reshaped_inputs, self.pool_height, self.pool_width, padding=0, stride=self.stride)
        # print("max x_col:", self.x_col.shape)
        
        self.x_col_argmax = np.argmax(self.x_col, axis=0)
        x_cols_max = self.x_col[self.x_col_argmax, np.arange(self.x_col.shape[1])]
        self.output = x_cols_max.reshape(int(output_height), int(output_width), batch_size, num_channel).transpose(2, 3, 0, 1)
        
        return self.output        

    def backward(self, dloss):
        """
        dloss is the gradients wrt the output of forward()
        """
        # pass
        batch_size, num_channel, in_height, in_width = self.inputs.shape
        # print("dloss:", dloss.shape)
        dloss_reshaped = dloss.transpose(2, 3, 0, 1).flatten()
        dx_cols = np.zeros_like(self.x_col)
        dx_cols[self.x_col_argmax, np.arange(dx_cols.shape[1])] = dloss_reshaped
        dx = im2col_bw(dx_cols, (batch_size * num_channel, 1, in_height, in_width), self.pool_height, self.pool_width, padding=0, stride=self.stride)
        # print(dx.shape, dx_cols.shape)
        dx = dx.reshape(self.inputs.shape)      
        return dx            
    
class LinearLayer(Transform):
    """
    Implement this class - Linear layer
    """
    def __init__(self, indim, outdim, rand_seed=0):
        """
        indim, outdim: input and output dimensions
        weights shape (indim,outdim)
        Use Xavier initialization for weights, as instructed on handout
        Initialze biases as an array of ones in shape of (outdim,1)
        """
        np.random.seed(rand_seed) # keep this line for autograding; you may remove it for training
        b = np.sqrt(6) / np.sqrt(indim + outdim)
        self.W = np.random.uniform(-b, b, (indim, outdim))
        self.b = np.zeros((outdim, 1))
        self.dW = np.zeros((indim, outdim))
        self.db = np.zeros((1, outdim))
        
        #check don't we need following momentum params
        self.momentum_W = np.zeros_like(self.W)
        self.momentum_b = np.zeros_like(self.b)

    def forward(self, inputs):
        """
        Forward pass of linear layer
        inputs shape (batch_size, indim)
        """
        # pass
        self.x = inputs
        yhat = np.dot(self.x, self.W) + self.b.T
        return yhat        

    def backward(self, dloss):
        """
        Read Transform.backward()'s docstring in this file
        dloss shape (batch_size, outdim)
        Return [gradient wrt weights, gradient wrt biases, gradient wrt input to this layer]
        """
        # pass
        self.dW = self.x.T @ dloss
        self.db = np.sum(dloss, axis=0, keepdims=True)
        dx = dloss @ self.W.T
        return [self.dW, self.db.T, dx]

    def update(self, learning_rate=0.001, momentum_coeff=0.5):
        """
        Similar to Conv.update()
        """
        # pass
        self.momentum_W = momentum_coeff * self.momentum_W - learning_rate * self.dW
        self.momentum_b = momentum_coeff * self.momentum_b - learning_rate * self.db.T
        
        self.W += self.momentum_W 
        self.b += self.momentum_b 

    def get_wb_fc(self):
        """
        Return weights and biases
        """
        return self.W, self.b


class SoftMaxCrossEntropyLoss():
    """
    Implement this class
    """
    def forward(self, logits, labels, get_predictions=False):
        """
        logits are pre-softmax scores, labels are true labels of given inputs
        labels are one-hot encoded
        logits and labels are in the shape of (batch_size, num_classes)
        returns loss as scalar

        (your loss should be the mean loss over the batch)
        """
        # pass
        self.labels = labels
        logits = logits - np.amax(logits, axis=1).reshape(logits.shape[0], 1)
        self.softmax = np.exp(logits)/np.sum(np.exp(logits), axis=1, keepdims=True)
        loss = -np.sum(labels * np.log(self.softmax), axis=1)
        return np.mean(loss), self.softmax
        #check about get_predictions
    
    def backward(self):
        """
        return shape (batch_size, num_classes)
        Remeber to divide by batch_size so the gradients correspond to the mean loss
        """
        # pass
        dl = self.softmax - self.labels
        return dl/dl.shape[0]  #dividing by batch_size

    def getAccu(self):
        """
        Implement as you wish, not autograded.
        """
        # pass
        pred = np.argmax(self.softmax, axis=1)
        actual = np.argmax(self.labels, axis=1)
        acc = np.sum(pred==actual)/len(pred)
        return acc


class ConvNet:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self):
        """
        Initialize Conv, ReLU, MaxPool, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 1x5x5 (or 5x5x5)
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then initialize linear layer with output 10 neurons
        Initialize SotMaxCrossEntropy object
        """
        # pass
        self.input_shape = (3, 32, 32)
        self.filter_shape = (1, 5, 5)
        self.conv1 = Conv(self.input_shape, self.filter_shape)
        self.activation = ReLU()
        self.maxpool = MaxPool(filter_shape=(2, 2), stride=2)
        self.flatten = Flatten()
        self.linearlayer = LinearLayer(256, 10)
        self.softmax = SoftMaxCrossEntropyLoss()

    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape (batch_size x channels x height x width)
        2. labels => True labels (batch_size, num_classes)

        Return loss and predicted labels after one forward pass
        """
        # pass
        h1 = self.conv1.forward(inputs)
        z1 = self.activation.forward(h1)
        pool1 = self.maxpool.forward(z1)
        flatten1 = self.flatten.forward(pool1)
        h2 = self.linearlayer.forward(flatten1)
        loss, outputs = self.softmax.forward(h2, y_labels)
        
        return loss, outputs
        
    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        # pass
        grad_wrt_out = self.softmax.backward()
        dlinear = self.linearlayer.backward(grad_wrt_out)
        dflatten = self.flatten.backward(dlinear[-1])
        dmaxpool = self.maxpool.backward(dflatten)
        dact = self.activation.backward(dmaxpool)
        dconv1 = self.conv1.backward(dact)

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        # pass
        self.conv1.update(learning_rate, momentum_coeff)
        self.linearlayer.update(learning_rate, momentum_coeff)


class ConvNetThree:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self):
        """
        Initialize Conv, ReLU, MaxPool, Conv, ReLU, Conv, ReLU, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 5x5x5
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then Conv with filter size of 5x5x5
        then apply Relu
        then Conv with filter size of 5x5x5
        then apply Relu
        then initialize linear layer with output 10 neurons
        Initialize SotMaxCrossEntropy object
        """
        pass

    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => True labels

        Return loss and predicted labels after one forward pass
        """
        pass

    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        pass

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        pass


class MLP:
    """
    Implement as you wish, not autograded
    """
    def __init__(self):
        pass

    def forward(self, inputs, y_labels):
        pass

    def backward(self):
        pass

    def update(self,learning_rate,momentum_coeff):
        pass


# Implement the training as you wish. This part will not be autograded.
if __name__ == '__main__':
    # This part may be helpful to write the training loop
    from argparse import ArgumentParser
    import matplotlib.pyplot as plt
    from load_svhn import train_X, train_y, test_X, test_y

    # Training parameters
    parser = ArgumentParser(description='CNN')
    parser.add_argument('--batch_size', type=int, default = 128)
    parser.add_argument('--learning_rate', type=float, default = 0.001)
    parser.add_argument('--momentum', type=float, default = 0.9)
    parser.add_argument('--num_epochs', type=int, default = 50)
    parser.add_argument('--conv_layers', type=int, default = 1)
    parser.add_argument('--filters', type=int, default = 1)
    parser.add_argument('--title', type=str, default=None)
    args = parser.parse_args()
    print('\n'.join([f'{k}: {v}' for k, v in vars(args).items()]))

    train_label = np.array([[i==lab for i in range(10)] for lab in train_y], np.int32)
    test_label = np.array([[i==lab for i in range(10)] for lab in test_y], np.int32)

    num_train = len(train_X)
    num_test = len(test_X)
    batch_size = args.batch_size
    train_iter = num_train // batch_size + 1
    test_iter = num_test // batch_size + 1

    if args.conv_layers == 1:
        cnn = ConvNet()
    else:
        raise NotImplementedError('Not implemented yet')
