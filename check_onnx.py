import onnxruntime as ort

# 加载模型
session = ort.InferenceSession('models/pangu_weather_24.onnx', providers=['CPUExecutionProvider'])

# 获取输入层信息
for node in session.get_inputs():
    print(f"输入节点名称: {node.name}")
    print(f"输入节点形状: {node.shape}") # 重点看这里