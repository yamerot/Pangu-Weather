import numpy as np
import torch
import onnxruntime as ort

def session_load(dir):
    options = ort.SessionOptions()
    # 保持你的稳定性设置
    options.enable_mem_pattern = True
    options.enable_mem_reuse = False 
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    cuda_provider_options = {
        'device_id': 0, 
        # 改为 kSameAsRequested，防止 ORT 预占过多“无用”显存
        'arena_extend_strategy': 'kSameAsRequested', 
        'cudnn_conv_algo_search': 'DEFAULT', 
        'do_copy_in_default_stream': True, 
    }
    return ort.InferenceSession(
        dir, 
        sess_options=options, 
        providers=[('CUDAExecutionProvider', cuda_provider_options)]
    )

# 1. 在显存中预先开辟好“共享”的输入输出容器
# 这样 4 个模型可以轮流使用这两块地盘，不再额外申请空间
device = 'cuda:0'
input_gpu = torch.empty((5, 13, 721, 1440), device=device, dtype=torch.float32)
input_surface_gpu = torch.empty((4, 721, 1440), device=device, dtype=torch.float32)
output_gpu = torch.empty((5, 13, 721, 1440), device=device, dtype=torch.float32)
output_surface_gpu = torch.empty((4, 721, 1440), device=device, dtype=torch.float32)

# 加载模型
input_t = {
    # 24: [0], 
    6: [0, 6], 
    3: [0, 6], 
    1: [0, 1]
}
sessions = {lead: session_load(f'models/pangu_weather_{lead}.onnx') for lead in input_t}

# 预绑定
bindings = {lead: sess.io_binding() for lead, sess in sessions.items()}
for lead, b in bindings.items():
    b.bind_input('input', 'cuda', 0, np.float32, input_gpu.shape, input_gpu.data_ptr())
    b.bind_input('input_surface', 'cuda', 0, np.float32, input_surface_gpu.shape, input_surface_gpu.data_ptr())
    b.bind_output('output', 'cuda', 0, np.float32, output_gpu.shape, output_gpu.data_ptr())
    b.bind_output('output_surface', 'cuda', 0, np.float32, output_surface_gpu.shape, output_surface_gpu.data_ptr())

# 测试跑通
for lead in input_t:
    print(f'正在运行 {lead}h 模型...')
    sessions[lead].run_with_iobinding(bindings[lead])
    print(f'{lead}h 模型运行成功')

print('所有模型过关')