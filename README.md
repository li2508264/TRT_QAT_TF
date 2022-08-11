# QAT_demo
Show how to use QAT in TF1.15 to train a network for mnist and convert it into TensorRT engine.
In my tests, result is correct.
Do it step by step.

`docker pull nvcr.io/nvidia/tensorflow:22.06-tf1-py3` with Trt8

 `pip install -r requirements.txt`



- 1 `python3 qat_training.py`
- 2 `python3 export_freezn_graph.py`
- 3 `python3 fold_constants.py -i saved_results/frozen_graph.pb`
- 4 
```python 
  python3 -m tf2onnx.convert --input saved_results/folded_mnist.pb --output saved_results/mnist_qat.onnx --inputs input_0:0 --outputs softmax_1:0 --opset 11 
  ```
- 5 `python3 build_engine.py --onnx saved_results/mnist_qat.onnx --engine saved_results/mnist_qat.trt -v`

- 6 `python3 infer.py -e saved_results/mnist_qat.trt -b 1`

can infer onnx model by python onnx_infer1.py

