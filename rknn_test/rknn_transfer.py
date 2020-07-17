from rknn.api import RKNN

from keras.models import load_model
import tensorflow as tf
import os
import os.path as osp
from keras import backend as K

model_name = 'fer2013_mini_XCEPTION.102-0.66'
#model_name = 'ck_mini_XCEPTION.95-0.89'
#model_name = 'ck_mini_XCEPTION.98-0.90'

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a prunned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    prunned so subgraphs that are not neccesary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

def pb_transfer():
    #input_fld = sys.path[0]
    input_fld = './trained_models/emotion_models/'
    weight_file = model_name + '.hdf5'

    output_fld = './'
    output_graph_name = model_name + '.pb'

    if not os.path.isdir(output_fld):
        os.mkdir(output_fld)
    weight_file_path = osp.join(input_fld, weight_file)

    K.set_learning_phase(0)
    net_model = load_model(weight_file_path)

    print('input is :', net_model.input.name)
    print ('output is:', net_model.output.name)

    sess = K.get_session()

    frozen_graph = freeze_session(K.get_session(), output_names=[net_model.output.op.name])

    from tensorflow.python.framework import graph_io

    graph_io.write_graph(frozen_graph, output_fld, output_graph_name, as_text=False)

    print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, output_graph_name))


INPUT_WIDTH = 64
INPUT_HEIGHT = 64

def transfer(pb_path, rknn_name):
    # 创建RKNN执行对象
    #rknn = RKNN(verbose=True, verbose_file='./mini_XCEPTION_build.log')
    rknn = RKNN()
# 配置模型输入，用于NPU对数据输入的预处理
# channel_mean_value='0 0 0 255'，那么模型推理时，将会对RGB数据做如下转换
# (R - 0)/255, (G - 0)/255, (B - 0)/255。推理时，RKNN模型会自动做均值和归一化处理
# reorder_channel=’0 1 2’用于指定是否调整图像通道顺序，设置成0 1 2即按输入的图像通道顺序不做调整
# reorder_channel=’2 1 0’表示交换0和2通道，如果输入是RGB，将会被调整为BGR。如果是BGR将会被调整为RGB
#图像通道顺序不做调整
    #rknn.config(channel_mean_value='0 0 0 255', reorder_channel='0 1 2')
    rknn.config(quantized_dtype='dynamic_fixed_point-8')
 
# 加载TensorFlow模型
# tf_pb='digital_gesture.pb'指定待转换的TensorFlow模型
# inputs指定模型中的输入节点
# outputs指定模型中输出节点
# input_size_list指定模型输入的大小
    print('--> Loading model')
    ret = rknn.load_tensorflow(tf_pb=pb_path,
                         inputs=['input_1'],
                         outputs=['predictions/Softmax'],
                         input_size_list=[[INPUT_WIDTH, INPUT_HEIGHT, 1]])
    if ret != 0:
        print('Load Model failed!')
        exit(ret)
    print('done')
 
# 创建解析pb模型
# do_quantization=False指定不进行量化
# 量化会减小模型的体积和提升运算速度，但是会有精度的丢失
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build Model failed!')
        exit(ret)
    print('done')
 
    # 导出保存rknn模型文件
    print('--> Export RKNN model')
    ret = rknn.export_rknn(rknn_name)
    if ret != 0:
        print('Export Model failed!')
        exit(ret)
    print('done')
 
    # Release RKNN Context
    rknn.release()

if __name__ == '__main__':
    pb_transfer()
    transfer(model_name + '.pb', model_name + '.rknn')
