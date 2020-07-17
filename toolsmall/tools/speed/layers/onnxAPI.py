"""
使用onnx python api 将 pytorch模型转成 onnx格式
更多细节可以参考 torch.onnx.export 输出的打印信息
https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md
"""
import onnx
from onnx import helper,TensorProto
import numpy as np

class OnnxAPI:
    def __init__(self,model_path="model.npz",output_file_path='model.onnx',
                 alpha_lrelu = 0.1,epsilon_bn = 1e-5,momentum_bn = 0.99):
        self.weights = np.load(model_path)
        self.nodes = list()
        self.inputs = list()
        self.outputs = list()
        self.initializer = list()

        self.alpha_lrelu = alpha_lrelu
        self.epsilon_bn = epsilon_bn
        self.momentum_bn = momentum_bn
        self.graph_name = "model_test"
        self.output_file_path = output_file_path

    def layer_input(self,layer_name,input_shape):
        # input_shape:[batch_size, channels, height, width]
        input_tensor = helper.make_tensor_value_info(
            str(layer_name), TensorProto.FLOAT,input_shape)
        self.inputs.append(input_tensor)
        return self

    def layer_output(self,previous_node_name,output_shape):
        # output_shape:[batch_size, channels, height, width]
        output_tensor = helper.make_tensor_value_info(
            str(previous_node_name), TensorProto.FLOAT,output_shape)
        self.outputs.append(output_tensor)
        return self

    def layer_conv(self,layer_name,previous_node_name,kernel_shape=(3,3),strides=(1,1),pads=(1,1,1,1)):
        inputs = [previous_node_name]
        dilations = [1, 1]
        weights_name = layer_name+".weight"
        inputs.append(weights_name)
        self._create_param_tensors(weights_name)
        bias_name = layer_name + ".bias"
        if bias_name in self.weights:
            inputs.append(bias_name)
            self._create_param_tensors(bias_name)

        conv_node = helper.make_node(
            'Conv',
            inputs=inputs,
            outputs=[layer_name],
            kernel_shape=kernel_shape,
            strides=strides,
            # auto_pad='SAME_LOWER',
            pads=pads,
            dilations=dilations,
            group=1,
            name=layer_name
        )
        self.nodes.append(conv_node)
        return self

    def layer_deconv(self,layer_name,previous_node_name,kernel_shape=(3,3),strides=(2,2),pads=(1,1,1,1),output_padding=(1, 1)):
        inputs = [previous_node_name]
        dilations = [1, 1]
        weights_name = layer_name+".weight"
        inputs.append(weights_name)
        self._create_param_tensors(weights_name)
        bias_name = layer_name + ".bias"
        if bias_name in self.weights:
            inputs.append(bias_name)
            self._create_param_tensors(bias_name)

        deconv_node = helper.make_node(
            'ConvTranspose',
            inputs=inputs,
            outputs=[layer_name],
            kernel_shape=kernel_shape,
            strides=strides,
            # auto_pad='SAME_LOWER',
            pads=pads,
            output_padding=output_padding,
            group=1,
            dilations=dilations,
            name=layer_name
        )
        self.nodes.append(deconv_node)
        return self

    def layer_bn(self,layer_name,previous_node_name):
        inputs = [previous_node_name]
        param_names = [layer_name+".weight",layer_name+".bias",layer_name+".running_mean",layer_name+".running_var"]
        inputs.extend(param_names)
        for param in param_names:
            self._create_param_tensors(param)

        batchnorm_node = helper.make_node(
            'BatchNormalization',
            inputs=inputs,
            outputs=[layer_name],
            epsilon=self.epsilon_bn,
            momentum=self.momentum_bn,
            name=layer_name
        )
        self.nodes.append(batchnorm_node)

        return self

    def layer_lrelu(self,layer_name,previous_node_name):
        inputs = [previous_node_name]
        lrelu_node = helper.make_node(
            'LeakyRelu',
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name,
            alpha=self.alpha_lrelu
        )
        self.nodes.append(lrelu_node)
        return self

    def layer_relu(self,layer_name,previous_node_name):
        inputs = [previous_node_name]
        relu_node = helper.make_node(
            'Relu',
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name
        )
        self.nodes.append(relu_node)
        return self

    def layer_tanh(self,layer_name,previous_node_name):
        inputs = [previous_node_name]
        tanh_node = helper.make_node(
            'Tanh',
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name
        )
        self.nodes.append(tanh_node)
        return self

    def layer_maxpool(self,layer_name,previous_node_name,kernel_shape=(2,2),strides=(2,2),pads=(0,0,0,0)):
        inputs = [previous_node_name]
        maxpool_node = helper.make_node(
            'MaxPool',
            inputs=inputs,
            outputs=[layer_name],
            kernel_shape=kernel_shape,
            strides=strides,
            # auto_pad='SAME_LOWER',
            pads=pads,
            name=layer_name
        )

        self.nodes.append(maxpool_node)
        return self

    def layer_avgpool(self,layer_name,previous_node_name,kernel_shape=(2,2),strides=(2,2),pads=(0,0,0,0)):
        inputs = [previous_node_name]
        avgpool_node = helper.make_node(
            'AveragePool',
            inputs=inputs,
            outputs=[layer_name],
            kernel_shape=kernel_shape,
            strides=strides,
            # auto_pad='SAME_LOWER',
            pads=pads,
            name=layer_name
        )

        self.nodes.append(avgpool_node)
        return self

    def layer_flatten(self,layer_name,previous_node_name,axis=1):
        inputs = [previous_node_name]
        flatten_node = helper.make_node(
            'Flatten',
            inputs=inputs,
            outputs=[layer_name],
            axis=axis,
            name=layer_name
        )

        self.nodes.append(flatten_node)
        return self

    def layer_fc(self,layer_name,previous_node_name):
        inputs = [previous_node_name]
        weights_name = layer_name + ".weight"
        inputs.append(weights_name)
        self._create_param_tensors(weights_name)
        bias_name = layer_name + ".bias"
        if bias_name in self.weights:
            inputs.append(bias_name)
            self._create_param_tensors(bias_name)
        fc_node = helper.make_node(
            'Gemm',
            inputs=inputs,
            outputs=[layer_name],
            alpha=1.,
            beta=1.,
            transB=1,
            name=layer_name
        )

        self.nodes.append(fc_node)
        return self

    def layer_sigmoid(self,layer_name,previous_node_name):
        inputs = [previous_node_name]
        sigmoid_node = helper.make_node(
            'Sigmoid',
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name
        )
        self.nodes.append(sigmoid_node)
        return self

    def layer_softmax(self,layer_name,previous_node_name,axis=1):
        inputs = [previous_node_name]
        softmax_node = helper.make_node(
            'Softmax',
            inputs=inputs,
            axis=axis,
            outputs=[layer_name],
            name=layer_name
        )
        self.nodes.append(softmax_node)
        return self

    def layer_add(self,layer_name,first_node_name, second_node_name):
        inputs = [first_node_name, second_node_name]
        shortcut_node = helper.make_node(
            'Add',
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name,
        )
        self.nodes.append(shortcut_node)
        return self

    def layer_concat(self,layer_name,route_node_name=[]):
        inputs = route_node_name
        route_node = helper.make_node(
            'Concat',
            axis=1,
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name,
        )
        self.nodes.append(route_node)

        return self

    def layer_upsample(self,layer_name,previous_node_name,resize_scale_factors=2):
        scales = np.array([1.0, 1.0, resize_scale_factors, resize_scale_factors]).astype(np.float32)
        scale_name = layer_name + ".scale"
        roi_name = layer_name + ".roi"
        inputs = [previous_node_name,roi_name,scale_name]

        resize_node = helper.make_node(
            'Resize',
            coordinate_transformation_mode='asymmetric',
            mode='nearest',
            nearest_mode='floor',
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name,
        )
        self.nodes.append(resize_node)

        # 获取权重信息
        shape = scales.shape
        scale_init = helper.make_tensor(
            scale_name, TensorProto.FLOAT,shape, scales)
        scale_input = helper.make_tensor_value_info(
            scale_name, TensorProto.FLOAT, shape)
        self.initializer.append(scale_init)
        self.inputs.append(scale_input)

        # In opset 11 an additional input named roi is required. Create a dummy tensor to satisfy this.
        # It is a 1D tensor of size of the rank of the input (4)
        rank = 4
        roi_input = helper.make_tensor_value_info(roi_name, TensorProto.FLOAT, [rank])
        roi_init = helper.make_tensor(roi_name, TensorProto.FLOAT, [rank], [0, 0, 0, 0])
        self.initializer.append(roi_init)
        self.inputs.append(roi_input)

        return self

    def _create_param_tensors(self,param_name):
        param_data = self.weights[param_name]
        param_data_shape = param_data.shape
        initializer_tensor = helper.make_tensor(
            param_name, TensorProto.FLOAT, param_data_shape, param_data.ravel())
        input_tensor = helper.make_tensor_value_info(
            param_name, TensorProto.FLOAT, param_data_shape)

        self.initializer.append(initializer_tensor)
        self.inputs.append(input_tensor)
        return self

    def save_model(self):
        graph_def = helper.make_graph(
            nodes=self.nodes,
            name=self.graph_name,
            inputs=self.inputs,
            outputs=self.outputs,
            initializer=self.initializer
        )
        print(helper.printable_graph(graph_def))

        model_def = helper.make_model(graph_def, producer_name='NVIDIA TensorRT sample')
        onnx.checker.check_model(model_def)
        onnx.save(model_def, self.output_file_path)


if __name__=="__main__":
    from torch import nn
    from toolsmall.tools.speed.modelTansform import onnx2engine, torch2onnx, torch2npz,to_numpy

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
            self.bn = nn.BatchNorm2d(32)
            self.lrelu = nn.LeakyReLU(0.1)
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
            self.maxpool = nn.MaxPool2d(2, 2, 0)
            self.deconv = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)
            # self.gn = nn.GroupNorm(8,32)
            self.tanh = nn.Tanh()
            self.avgpool = nn.AvgPool2d(112, 1, 0)
            self.fc = nn.Linear(32, 8)
            self.sigmoid = nn.Sigmoid()
            self.softmax = nn.Softmax(-1)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.lrelu(x)
            x = self.upsample(x)
            x = self.maxpool(x)
            x = self.deconv(x)
            # x = self.gn(x)
            x = self.tanh(x)
            x = self.maxpool(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            x = self.sigmoid(x)
            x = self.softmax(x)
            return x


    model = Net().eval()

    # x = torch.randn([1,3,224,224])
    x = torch.ones([1, 3, 224, 224])
    # print(model(x))
    torch2npz(model)
    torch2onnx(model, x) # torch.onnx.export

    # -------------使用onnx python api 转成 onnx文件--------------------------------------------

    input_shape = [1,3,224,224]
    output_shape = [1, 8]

    oapi = OnnxAPI()
    oapi.layer_input("input",input_shape)

    oapi.layer_conv("conv","input",(3,3),(2,2))
    oapi.layer_bn("bn","conv")
    oapi.layer_lrelu('lrelu','bn')
    oapi.layer_upsample('upsample','lrelu',2)
    oapi.layer_maxpool('maxpool','upsample')
    oapi.layer_deconv('deconv','maxpool',(3,3),(2,2))
    oapi.layer_tanh('tanh','deconv')
    oapi.layer_maxpool('maxpool2', 'tanh')
    oapi.layer_avgpool('avgpool','maxpool2',(112,112),(1,1))
    oapi.layer_flatten('flatten','avgpool',1)
    oapi.layer_fc('fc','flatten')
    oapi.layer_sigmoid('sigmoid','fc')
    oapi.layer_softmax('softmax',"sigmoid",1)

    oapi.layer_output('softmax', output_shape)

    oapi.save_model()


    # onnx2engine('model.onnx',ModelData=ModelData)

    # ort_session = onnxruntime.InferenceSession("model.onnx")
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    # ort_outs = ort_session.run(None, ort_inputs)
    # print(ort_outs[0])