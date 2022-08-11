import os, sys
import numpy as np
sys.path.append(os.getcwd())
import onnxruntime
import onnx
import pdb
import cv2
def _to_ltrb_bbox(
    vertices: np.array,
    scale: int = 8,
    size_w_prior: float = 1,
    size_h_prior: float = 1,
    with_rotation_angle: bool = False,
):

    _, _, output_height, output_width = vertices.shape

    # Construct a 2D grid of cell x and y coordinates and add offset.
    # HACK(oandrienko): To avoid the TRT/ONNX `Range` op during export. We create the
    #   the range operation in Numpy then convert to a torch Tensor constant when
    #   exporting. Please remove this when we update to TRT 8.

    lin_x_np = np.arange(0, output_width, dtype=np.int32)
    lin_y_np = np.arange(0, output_height, dtype=np.int32)
    lin_x = np.asarray(lin_x_np, dtype=np.int32) * scale
    lin_y = np.asarray(lin_y_np, dtype=np.int32) * scale
    grid_x, grid_y = np.meshgrid(lin_x, lin_y)

    # Unstack final network Offset Predictions.
    conf,center_x, center_y, width, height = np.array_split(vertices, [1,2,3,4],1)

    center_x = center_x + grid_x
    center_y = center_y + grid_y
    height[np.where(height>8)] = 8
    width[np.where(width>8)] = 8
    # Build the LTRB Bbox for Detection Evaluation.
    y_offset = 0.5 * size_h_prior * np.exp(height)
    x_offset = 0.5 * size_w_prior * np.exp(width)
    ltrb = np.concatenate(
        [
            center_x - x_offset,
            center_y - y_offset,
            center_x + x_offset,
            center_y + y_offset,
        ],
        1,
    )

    # Add optional rotational angle.
    return conf,ltrb


class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path,providers = ['CUDAExecutionProvider'])
        print(self.onnx_session.get_providers())
        # self.onnx_session.set_providers(['CUDAExecutionProvider'], [ {'device_id': 0}])
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_tensor):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param image_tensor:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_tensor
        return input_feed

    def forward(self, image_tensor):
        '''
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_tensor:
        :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_tensor})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_tensor})
        input_feed = self.get_input_feed(self.input_name, image_tensor)
        scores = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return scores
def YUV444(frm):
    m,n,_ = frm.shape

model = ONNXModel('mnist_qat.onnx')
#a = np.load('test.npy')
#a = a[1:2,:,:,:]*255
#b = a[0,0,:,:]*255
#b = b.astype('uint8')
#print(a)
a = cv2.imread('t.jpg')
a = a[:,:,0:1]
a = np.expand_dims(a,0)
a = a.transpose((0,3,1,2))
a = a.astype('float32')
result = model.forward(a)
print(result)
