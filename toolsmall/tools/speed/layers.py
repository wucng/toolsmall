"""
直接解析 .npz文件（结合tensorrt api）
实现 resnet18 （注：并没有完全还原原始的resnet18，有些细节对不上，虽然可以编译 但精度很低，待改进）
"""

import numpy as np
import tensorrt as trt

class ModelData(object):
    INPUT_NAME = "input"
    OUTPUT_NAME = "output"  #
    BATCH_SIZE = 32 # 要与 pytorch2onnx 时设置的batch size 对应
    INPUT_SHAPE = (32,3, 224, 224)
    # INPUT_SHAPE = (3, 224, 224)
    OUTPUT_SIZE = 1000
    # DTYPE = trt.float16  # 使用半精度 half-float
    # NP_DTYPE = np.float16
    DTYPE = trt.float32
    NP_DTYPE = np.float32

    # torch_model_path = "./model.pth"
    npz_path = "./model.npz"
    engine_file_path = "./model.engine"
    data_dir = "/media/wucong/225A6D42D4FA828F1/datas/samples" # 推理的文件路径



def trt_bn(network,input_size,name,weights,dtype,belta=1e-3):
    g0 = weights[name+'.weight']  # .reshape(-1)
    b0 = weights[name+'.bias']  # .reshape(-1)
    m0 = weights[name+'.running_mean']  # .reshape(-1)
    v0 = weights[name+'.running_var']  # .reshape(-1)
    scale0 = g0 / np.sqrt(v0 + belta)
    shift0 = -m0 / np.sqrt(v0 + belta) * g0 + b0
    power0 = np.ones(len(g0), dtype=dtype)
    bn1 = network.add_scale(input=input_size, mode=trt.ScaleMode.CHANNEL, shift=shift0, scale=scale0,
                            power=power0)
    return bn1

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

def layer_common(network,weights,input_size,dtype,
                 conv_config=[],
                 bn_name="",
                 activation_type=None,
                 pool_type=trt.PoolingType.MAX,
                 pool_config=[]):
    """
    conv + batch_norm + [relu] +maxpool
    :return:
    """
    conv1 = trt_conv(network, input_size, conv_config[0], weights, dtype,
                              conv_config[1], conv_config[2],conv_config[3])
    bn1 = trt_bn(network, conv1.get_output(0), bn_name, weights, dtype)
    if activation_type!=None:
        relu1 = network.add_activation(input=bn1.get_output(0), type=activation_type)
        maxpool1 = trt_pool(network, relu1.get_output(0), pool_type, pool_config[0],
                                     pool_config[1], pool_config[2])
    else:
        maxpool1 = trt_pool(network, bn1.get_output(0), pool_type, pool_config[0],
                            pool_config[1], pool_config[2])

    return maxpool1

def bottleNet(network,weights,input_size,dtype,
              conv1_config=[],
              bn1_name="",relu1_type=trt.ActivationType.RELU,
              conv2_config=[],
              bn2_name="",relu2_type=trt.ActivationType.RELU,
              downsample=[]):
    """
    resnet 瓶颈层
    """
    conv1 = trt_conv(network, input_size, conv1_config[0],
                                       weights, dtype, conv1_config[1], conv1_config[2], conv1_config[3])
    bn1 = trt_bn(network, conv1.get_output(0), bn1_name, weights,dtype)

    relu1 = network.add_activation(input=bn1.get_output(0),type=relu1_type)
    # ----------------------------------------------------------------------------------------------
    conv2 = trt_conv(network, relu1.get_output(0), conv2_config[0],
                                       weights, dtype, conv2_config[1], conv2_config[2], conv2_config[3])
    bn2 = trt_bn(network, conv2.get_output(0),bn2_name, weights,dtype)

    # 是否需要做downsample
    if len(downsample)>0:
        downsample_conv = trt_conv(network, input_size,downsample[0],
                           weights, dtype, downsample[1], downsample[2], downsample[3])
        downsample_bn = trt_bn(network, downsample_conv.get_output(0),
                             downsample[4], weights, dtype)

        backbone_layer1 = network.add_elementwise(input1=downsample_bn.get_output(0),
                                                  input2=bn2.get_output(0),
                                                  op=trt.ElementWiseOperation.SUM)
    else:
        backbone_layer1 = network.add_elementwise(input1=input_size,
                                                  input2=bn2.get_output(0),
                                                  op=trt.ElementWiseOperation.SUM)

    relu2 = network.add_activation(input=backbone_layer1.get_output(0),type=relu2_type)

    return relu2

# resnet 18
def populate_network2(network, weights):
    # Configure the network layers based on the weights provided.
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

    # ----------------backbone-------------------------------------------------
    backbone_maxpool1=layer_common(network,weights,input_tensor,ModelData.NP_DTYPE,
                                   ["backbone.conv1",(7, 7), (2, 2),(3, 3)],
                                   'backbone.bn1',trt.ActivationType.RELU,
                                   trt.PoolingType.MAX,[(3, 3), (2, 2), (1, 1)])

    # --------------------backbone.layer1.0---------------------------------------------
    backbone_layer1_0=bottleNet(network,weights,backbone_maxpool1.get_output(0),ModelData.NP_DTYPE,
                                ['backbone.layer1.0.conv1',(3, 3), (1, 1), (1, 1)],
                                'backbone.layer1.0.bn1',trt.ActivationType.RELU,
                                ['backbone.layer1.0.conv2',(3, 3), (1, 1), (1, 1)],
                                'backbone.layer1.0.bn2',
                                trt.ActivationType.RELU)

    # ----------------backbone.layer1.1-------------------------------------------------
    backbone_layer1_1=bottleNet(network,weights,backbone_layer1_0.get_output(0),ModelData.NP_DTYPE,
                                ['backbone.layer1.1.conv1',(3, 3), (1, 1), (1, 1)],
                                'backbone.layer1.1.bn1',trt.ActivationType.RELU,
                                ['backbone.layer1.1.conv2',(3, 3), (1, 1), (1, 1)],
                                'backbone.layer1.1.bn2',trt.ActivationType.RELU,
                                )

    # ----------------backbone.layer2.0-------------------------------------------------
    backbone_layer2_0 = bottleNet(network, weights, backbone_layer1_1.get_output(0), ModelData.NP_DTYPE,
                                  ['backbone.layer2.0.conv1', (3, 3), (2, 2), (1, 1)],
                                  'backbone.layer2.0.bn1', trt.ActivationType.RELU,
                                  ['backbone.layer2.0.conv2', (3, 3), (1, 1), (1, 1)],
                                  'backbone.layer2.0.bn2', trt.ActivationType.RELU,
                                  ["backbone.layer2.0.downsample.0",(1, 1), (2, 2), (0, 0),
                                   "backbone.layer2.0.downsample.1"]
                                  )

    # ----------------backbone.layer2.1-------------------------------------------------
    backbone_layer2_1 = bottleNet(network, weights, backbone_layer2_0.get_output(0), ModelData.NP_DTYPE,
                                  ['backbone.layer2.1.conv1', (3, 3), (1, 1), (1, 1)],
                                  'backbone.layer2.1.bn1', trt.ActivationType.RELU,
                                  ['backbone.layer2.1.conv2', (3, 3), (1, 1), (1, 1)],
                                  'backbone.layer2.1.bn2', trt.ActivationType.RELU,
                                  )

    # ----------------backbone.layer3.0-------------------------------------------------
    backbone_layer3_0 = bottleNet(network, weights, backbone_layer2_1.get_output(0), ModelData.NP_DTYPE,
                                  ['backbone.layer3.0.conv1', (3, 3), (2, 2), (1, 1)],
                                  'backbone.layer3.0.bn1', trt.ActivationType.RELU,
                                  ['backbone.layer3.0.conv2', (3, 3), (1, 1), (1, 1)],
                                  'backbone.layer3.0.bn2', trt.ActivationType.RELU,
                                  ["backbone.layer3.0.downsample.0", (1, 1), (2, 2), (0, 0),
                                   "backbone.layer3.0.downsample.1"]
                                  )

    # ----------------backbone.layer3.1-------------------------------------------------
    backbone_layer3_1 = bottleNet(network, weights, backbone_layer3_0.get_output(0), ModelData.NP_DTYPE,
                                  ['backbone.layer3.1.conv1', (3, 3), (1, 1), (1, 1)],
                                  'backbone.layer3.1.bn1', trt.ActivationType.RELU,
                                  ['backbone.layer3.1.conv2', (3, 3), (1, 1), (1, 1)],
                                  'backbone.layer3.1.bn2', trt.ActivationType.RELU,
                                  )

    # ----------------backbone.layer4.0-------------------------------------------------
    backbone_layer4_0 = bottleNet(network, weights, backbone_layer3_1.get_output(0), ModelData.NP_DTYPE,
                                  ['backbone.layer4.0.conv1', (3, 3), (2, 2), (1, 1)],
                                  'backbone.layer4.0.bn1', trt.ActivationType.RELU,
                                  ['backbone.layer4.0.conv2', (3, 3), (1, 1), (1, 1)],
                                  'backbone.layer4.0.bn2', trt.ActivationType.RELU,
                                  ["backbone.layer4.0.downsample.0", (1, 1), (2, 2), (0, 0),
                                   "backbone.layer4.0.downsample.1"]
                                  )

    # ----------------backbone.layer4.1-------------------------------------------------
    backbone_layer4_1 = bottleNet(network, weights, backbone_layer4_0.get_output(0), ModelData.NP_DTYPE,
                                  ['backbone.layer4.1.conv1', (3, 3), (1, 1), (1, 1)],
                                  'backbone.layer4.1.bn1', trt.ActivationType.RELU,
                                  ['backbone.layer4.1.conv2', (3, 3), (1, 1), (1, 1)],
                                  'backbone.layer4.1.bn2', trt.ActivationType.RELU,
                                  )

    # --------_conv1---------------------------------
    _conv=layer_common(network, weights, backbone_layer4_1.get_output(0), ModelData.NP_DTYPE,
                 ["_conv1.1", (1, 1), (1, 1), (0, 0)],
                 "_conv1.2", trt.ActivationType.RELU, trt.PoolingType.AVERAGE,
                 [(1, 1), (7, 7), (0, 0)])

    # add softmax or not
    softmax_layer=network.add_softmax(_conv.get_output(0))

    softmax_layer.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=softmax_layer.get_output(0))


def populate_network(network, weights):
    # Configure the network layers based on the weights provided.
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

    # ----------------backbone-------------------------------------------------
    backbone_maxpool1=layer_common(network,weights,input_tensor,ModelData.NP_DTYPE,
                                   ["conv1",(7, 7), (2, 2),(3, 3)],
                                   'bn1',trt.ActivationType.RELU,
                                   trt.PoolingType.MAX,[(3, 3), (2, 2), (1, 1)])

    # --------------------backbone.layer1.0---------------------------------------------
    backbone_layer1_0=bottleNet(network,weights,backbone_maxpool1.get_output(0),ModelData.NP_DTYPE,
                                ['layer1.0.conv1',(3, 3), (1, 1), (1, 1)],
                                'layer1.0.bn1',trt.ActivationType.RELU,
                                ['layer1.0.conv2',(3, 3), (1, 1), (1, 1)],
                                'layer1.0.bn2',
                                trt.ActivationType.RELU)

    # ----------------backbone.layer1.1-------------------------------------------------
    backbone_layer1_1=bottleNet(network,weights,backbone_layer1_0.get_output(0),ModelData.NP_DTYPE,
                                ['layer1.1.conv1',(3, 3), (1, 1), (1, 1)],
                                'layer1.1.bn1',trt.ActivationType.RELU,
                                ['layer1.1.conv2',(3, 3), (1, 1), (1, 1)],
                                'layer1.1.bn2',trt.ActivationType.RELU,
                                )

    # ----------------backbone.layer2.0-------------------------------------------------
    backbone_layer2_0 = bottleNet(network, weights, backbone_layer1_1.get_output(0), ModelData.NP_DTYPE,
                                  ['layer2.0.conv1', (3, 3), (2, 2), (1, 1)],
                                  'layer2.0.bn1', trt.ActivationType.RELU,
                                  ['layer2.0.conv2', (3, 3), (1, 1), (1, 1)],
                                  'layer2.0.bn2', trt.ActivationType.RELU,
                                  ["layer2.0.downsample.0",(1, 1), (2, 2), (0, 0),
                                   "layer2.0.downsample.1"]
                                  )

    # ----------------backbone.layer2.1-------------------------------------------------
    backbone_layer2_1 = bottleNet(network, weights, backbone_layer2_0.get_output(0), ModelData.NP_DTYPE,
                                  ['layer2.1.conv1', (3, 3), (1, 1), (1, 1)],
                                  'layer2.1.bn1', trt.ActivationType.RELU,
                                  ['layer2.1.conv2', (3, 3), (1, 1), (1, 1)],
                                  'layer2.1.bn2', trt.ActivationType.RELU,
                                  )

    # ----------------backbone.layer3.0-------------------------------------------------
    backbone_layer3_0 = bottleNet(network, weights, backbone_layer2_1.get_output(0), ModelData.NP_DTYPE,
                                  ['layer3.0.conv1', (3, 3), (2, 2), (1, 1)],
                                  'layer3.0.bn1', trt.ActivationType.RELU,
                                  ['layer3.0.conv2', (3, 3), (1, 1), (1, 1)],
                                  'layer3.0.bn2', trt.ActivationType.RELU,
                                  ["layer3.0.downsample.0", (1, 1), (2, 2), (0, 0),
                                   "layer3.0.downsample.1"]
                                  )

    # ----------------backbone.layer3.1-------------------------------------------------
    backbone_layer3_1 = bottleNet(network, weights, backbone_layer3_0.get_output(0), ModelData.NP_DTYPE,
                                  ['layer3.1.conv1', (3, 3), (1, 1), (1, 1)],
                                  'layer3.1.bn1', trt.ActivationType.RELU,
                                  ['layer3.1.conv2', (3, 3), (1, 1), (1, 1)],
                                  'layer3.1.bn2', trt.ActivationType.RELU,
                                  )

    # ----------------backbone.layer4.0-------------------------------------------------
    backbone_layer4_0 = bottleNet(network, weights, backbone_layer3_1.get_output(0), ModelData.NP_DTYPE,
                                  ['layer4.0.conv1', (3, 3), (2, 2), (1, 1)],
                                  'layer4.0.bn1', trt.ActivationType.RELU,
                                  ['layer4.0.conv2', (3, 3), (1, 1), (1, 1)],
                                  'layer4.0.bn2', trt.ActivationType.RELU,
                                  ["layer4.0.downsample.0", (1, 1), (2, 2), (0, 0),
                                   "layer4.0.downsample.1"]
                                  )

    # ----------------backbone.layer4.1-------------------------------------------------
    backbone_layer4_1 = bottleNet(network, weights, backbone_layer4_0.get_output(0), ModelData.NP_DTYPE,
                                  ['layer4.1.conv1', (3, 3), (1, 1), (1, 1)],
                                  'layer4.1.bn1', trt.ActivationType.RELU,
                                  ['layer4.1.conv2', (3, 3), (1, 1), (1, 1)],
                                  'layer4.1.bn2', trt.ActivationType.RELU,
                                  )

    # -------------------------------------------------------------------------------------------------------------
    avg_pool = trt_pool(network, backbone_layer4_1.get_output(0), trt.PoolingType.AVERAGE,(1, 1), (7, 7), (0, 0))
    fc1_w = weights['fc.weight']
    fc1_b = weights['fc.bias']
    fc1 = network.add_fully_connected(input=avg_pool.get_output(0), num_outputs=1000, kernel=fc1_w, bias=fc1_b)

    # add softmax or not
    softmax_layer=network.add_softmax(fc1.get_output(0))

    softmax_layer.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=softmax_layer.get_output(0))
