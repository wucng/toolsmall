import numpy as np
import tensorrt as trt

class ModelData(object):
    INPUT_NAME = "input"
    OUTPUT_NAME = "output"  #
    BATCH_SIZE = 128 # 要与 pytorch2onnx 时设置的batch size 对应
    INPUT_SHAPE = (128,3, 64, 64)
    OUTPUT_SIZE = 10
    DTYPE = trt.float16  # 使用半精度 half-float
    NP_DTYPE = np.float16
    # DTYPE = trt.float32
    # NP_DTYPE = np.float32


def trt_bn(network,input_size,name,weights,dtype,belta=1e-5):
    g0 = weights[name+'.weight']  # .reshape(-1)
    b0 = weights[name+'.bias']  # .reshape(-1)
    m0 = weights[name+'.running_mean']  # .reshape(-1)
    v0 = weights[name+'.running_var']  # .reshape(-1)
    scale0 = g0 / np.sqrt(v0 + belta)
    shift0 = -m0 / np.sqrt(v0 + belta) * g0 + b0
    power0 = np.ones(len(g0), dtype=dtype)
    # bn1 = network.add_scale(input=input_size, mode=trt.ScaleMode.CHANNEL, shift=shift0, scale=scale0,power=power0)
    bn1 = network.add_scale(input=input_size, mode=trt.ScaleMode.CHANNEL, shift=shift0, scale=scale0)
    return bn1

def trt_bn1d(network,input_size,name,weights,new_shape,belta=1e-5):
    g0 = weights[name+'.weight']  # .reshape(-1)
    b0 = weights[name+'.bias']  # .reshape(-1)
    m0 = weights[name+'.running_mean']  # .reshape(-1)
    v0 = weights[name+'.running_var']  # .reshape(-1)
    scale0 = g0 / np.sqrt(v0 + belta)
    shift0 = -m0 / np.sqrt(v0 + belta) * g0 + b0

    # reshape to 2D
    shuffle = network.add_shuffle(input_size)
    shuffle.reshape_dims = (new_shape, new_shape, 1)

    # power0 = np.ones(len(g0), dtype=dtype)
    # bn1 = network.add_scale(input=input_size, mode=trt.ScaleMode.CHANNEL, shift=shift0, scale=scale0,power=power0)
    bn1 = network.add_scale(input=input_size, mode=trt.ScaleMode.CHANNEL, shift=shift0, scale=scale0)

    # reshape to 1D
    shuffle = network.add_shuffle(bn1.get_output(0))
    shuffle.reshape_dims = (new_shape, new_shape, 1)

    return shuffle

def trt_conv(network,input_size,name,weights,dtype,kernel_shape,stride,padding):
    conv1_w = weights[name+'.weight']
    conv1_b = weights[name+'.bias'] if name+'.bias' in weights else np.zeros([conv1_w.shape[0]], dtype=dtype)
    conv1 = network.add_convolution(input=input_size, num_output_maps=conv1_w.shape[0],
                                    kernel_shape=kernel_shape, kernel=conv1_w, bias=conv1_b)
    conv1.stride = stride
    conv1.padding = padding

    return conv1

def trt_pool(network,input_size,type,pool_size,stride,padding=(0,0)):
    pool1 = network.add_pooling(input=input_size, type=type, window_size=pool_size)
    pool1.stride = stride
    pool1.padding = padding
    return pool1

def trt_active(network,input_size,type=trt.ActivationType.RELU):
    return network.add_activation(input=input_size, type=type)

def trt_add(network,input_size1,input_size2):
    return network.add_elementwise(input1=input_size1,
                            input2=input_size2,
                            op=trt.ElementWiseOperation.SUM)

def trt_fc(network,input_size,name,weights,dtype):
    fc1_w = weights[name + '.weight']
    fc1_b = weights[name + '.bias'] if name + '.bias' in weights else np.zeros([conv1_w.shape[0]], dtype=dtype)

    fc2 = network.add_fully_connected(input=input_size, num_outputs=fc1_w.shape[0], kernel=fc1_w, bias=fc1_b)
    return fc2

def populate_network(network, weights):
    dtype = ModelData.NP_DTYPE
    # Configure the network layers based on the weights provided.
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

    conv1 = trt_conv(network,input_tensor,"conv1",weights,dtype,(5,5),(1,1),(0,0))
    bn1 = trt_bn(network,conv1.get_output(0),"bn1",weights,dtype)
    relu1 = trt_active(network,bn1.get_output(0))
    maxpool1 = trt_pool(network,relu1.get_output(0),trt.PoolingType.MAX,(2,2),(2,2))

    conv2 = trt_conv(network, maxpool1.get_output(0), "conv2", weights, dtype, (5, 5), (1, 1), (0, 0))
    bn2 = trt_bn(network, conv2.get_output(0), "bn2", weights, dtype)
    relu2 = trt_active(network, bn2.get_output(0))
    maxpool2 = trt_pool(network, relu2.get_output(0), trt.PoolingType.MAX, (2, 2), (2, 2))

    conv3 = trt_conv(network, maxpool2.get_output(0), "conv3", weights, dtype, (5, 5), (1, 1), (0, 0))
    bn3 = trt_bn(network, conv3.get_output(0), "bn3", weights, dtype)
    relu3 = trt_active(network, bn3.get_output(0))
    maxpool3 = trt_pool(network, relu3.get_output(0), trt.PoolingType.MAX, (2, 2), (2, 2))

    conv4 = trt_conv(network, maxpool3.get_output(0), "conv4", weights, dtype, (3, 3), (1, 1), (1, 1))
    bn4 = trt_bn(network, conv4.get_output(0), "bn4", weights, dtype)
    relu4 = trt_active(network, bn4.get_output(0))
    maxpool4 = trt_pool(network, relu4.get_output(0), trt.PoolingType.MAX, (2, 2), (2, 2))

    fc1 = trt_fc(network,maxpool4.get_output(0),"fc1",weights,dtype)
    relufc1 = trt_active(network, fc1.get_output(0))

    fc2 = trt_fc(network, relufc1.get_output(0), "fc2", weights, dtype)

    """
    fc2.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=fc2.get_output(0))
    """
    # or add softmax layer
    softmax = network.add_softmax(fc2.get_output(0))
    softmax.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=softmax.get_output(0))
    # """