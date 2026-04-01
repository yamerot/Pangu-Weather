import os
import sys
import xarray as xr
import numpy as np
import onnxruntime as ort
import logging
from zarr.codecs import BloscCodec

# nohup python run_local_single.py &

def get_data(ds: xr.Dataset, time_range, vars, dims):
    """从 dataset 中读取需要的变量和时间范围, 将不同变量连接为通道维度, 组织成 array"""
    # 在索引 1 处增加通道维度，确认：该操作不会改变传入的 dims 本身
    dims_copy = list(dims)
    dims_copy.insert(1, 'channel')
    try:
        selected =  ds.sel(time=time_range)
    except KeyError: # 如果 timerange 越界, 改用 reindex 方法
        logger.warning('time out of range')
        selected =  ds.reindex(time=time_range)
    return selected[vars].to_array(dim='channel').transpose(*dims_copy).values.astype(np.float32)

def save_data(data, filepath, vars, coords, dims, write_mode, chunk):
    """将通道维度拆分回不同的变量, 保存为 zarr"""
    data_arrs = []
    for i, var in enumerate(vars):
        data_arrs.append(xr.DataArray(data[:, i], coords, dims=dims, name=var))
    dataset = xr.merge(data_arrs)
    if write_mode == 'w':
        compressor = BloscCodec(cname='zstd', clevel=7, shuffle='shuffle')
        encoding = {
            var: {
                'compressors': [compressor],
                'chunks': chunk, 
                "dtype": "float32"
            } for var in vars}
        dataset.to_zarr(filepath, mode='w', zarr_format=3, encoding=encoding)
    else:
        dataset.to_zarr(filepath, mode='a', append_dim='time', zarr_format=3)

def session_load(dir):
    options = ort.SessionOptions()
    options.enable_mem_pattern = True
    options.enable_mem_reuse = False
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    cuda_provider_options = {
        'device_id': device_id, 
        'arena_extend_strategy': 'kNextPowerOfTwo', 
        'cudnn_conv_algo_search': 'DEFAULT', 
        'do_copy_in_default_stream': True, 
    }
    return ort.InferenceSession(
        dir, 
        sess_options=options, 
        providers=[('CUDAExecutionProvider', cuda_provider_options)]
        )

def setup_logging(level=logging.INFO):
    logger = logging.getLogger('pangu')
    logger.setLevel(level)

    # 文件处理器
    file_handler = logging.FileHandler('./run_local.log', mode='w')
    file_handler.setLevel(level)

    # 格式器：包含时间、级别、消息
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger

os.chdir('/home/chengzy/pangu')

# 日志设置
logger = setup_logging()
def handle_exception(exc_type, exc_value, exc_traceback):
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
sys.excepthook = handle_exception

# 设备信息
device_type, device_id = 'cuda', 0
device = f'{device_type}:{device_id}'

# 以下代码中，_p 代表 upper (高空), _s 代表 surface (表面)

# 输入时刻与数据路径
data_dir = '/data/chengzy'
t_in = 0
input_file_p = f'{data_dir}/era5_zarr/plev.zarr' if t_in == 0 else f'{data_dir}/pangu_zarr/{t_in}/plev.zarr'
input_file_s = f'{data_dir}/era5_zarr/slev.zarr' if t_in == 0 else f'{data_dir}/pangu_zarr/{t_in}/slev.zarr'

# 模型预测时效
lead = 24
model_file = f'models/pangu_weather_{lead}.onnx'

# 输出时刻与数据路径
t_out = t_in + lead
os.makedirs(f'{data_dir}/pangu_zarr/{t_out}', exist_ok=True)
output_file_p = f'{data_dir}/pangu_zarr/{t_out}/plev.zarr'
output_file_s = f'{data_dir}/pangu_zarr/{t_out}/slev.zarr'

# 输入 dataset 及其数据变量、维度、chunk 等信息
# 其中 chunk 各维度与 dims 中的顺序是对应的，且只有第一个时间维度有切分，其余维度均是完整的
input_ds_p = xr.open_dataset(input_file_p)
coords_p = dict(input_ds_p.coords)
vars_p = ['z', 'q', 't', 'u', 'v']
chunk_p = (12, 13, 721, 1440)
dim_names_p = ['time', 'pressure_level', 'latitude', 'longitude']

input_ds_s = xr.open_dataset(input_file_s)
coords_s = dict(input_ds_s.coords)
vars_s = ['msl', 'u10', 'v10', 't2m']
chunk_s = (120, 721, 1440)
dim_names_s = ['time', 'latitude', 'longitude']

# 分配内存
io_size_p = (12, len(vars_p), 13, 721, 1440)
input_p = np.empty(io_size_p, dtype=np.float32)
output_p = np.empty(io_size_p, dtype=np.float32)

io_size_s = (120, len(vars_s), 721, 1440)
input_s = np.empty(io_size_s, dtype=np.float32)
output_s = np.empty(io_size_s, dtype=np.float32)

# 分配显存
slice_size_p = io_size_p[1:]
gpu_input_p = ort.OrtValue.ortvalue_from_shape_and_type(slice_size_p, np.float32, device_type, device_id)
gpu_output_p = ort.OrtValue.ortvalue_from_shape_and_type(slice_size_p, np.float32, device_type, device_id)

slice_size_s = io_size_s[1:]
gpu_input_s = ort.OrtValue.ortvalue_from_shape_and_type(slice_size_s, np.float32, device_type, device_id)
gpu_output_s = ort.OrtValue.ortvalue_from_shape_and_type(slice_size_s, np.float32, device_type, device_id)

logger.info(f'memory initialized')

# 加载模型，初始化 io_binding
session = session_load(model_file)
binding = session.io_binding()

logger.info(f'session loaded')

# 以小时为步, 运行预测模型
time = np.datetime64('2023-12-31T00')
dt = np.timedelta64(1, 'h')
end_time = np.datetime64('2023-12-31T23')
nt = 0
write_mode_p = 'w' if not os.path.exists(output_file_p) else 'a'
write_mode_s = 'w' if not os.path.exists(output_file_s) else 'a'

while time <= end_time:
    
    # 获取当前时间步在 chunk 中的位置
    nt_p = nt % chunk_p[0]
    nt_s = nt % chunk_s[0]

    # 以 chunk 为单位读取数据, 硬盘 -> 内存
    if nt_p == 0:
        logger.debug(f'proceeding {time}, updating input_p chunk')
        time_range_p = np.arange(time, time + chunk_p[0] * dt, dt)
        np.copyto(input_p, get_data(input_ds_p, time_range_p, vars_p, dim_names_p))

    if nt_s == 0:
        logger.info(f'proceeding {time}, updating input_s chunk')
        time_range_s = np.arange(time, time + chunk_s[0] * dt, dt)
        np.copyto(input_s, get_data(input_ds_s, time_range_s, vars_s, dim_names_s))

    # 内存（提取切片）-> 显存 [input[nt] -> gpu_input]
    gpu_input_p.update_inplace(input_p[nt_p])
    gpu_input_s.update_inplace(input_s[nt_s])

    # 绑定并运行 [gpu_input -> gpu_output]
    binding.bind_ortvalue_input('input', gpu_input_p)
    binding.bind_ortvalue_input('input_surface', gpu_input_s)
    binding.bind_ortvalue_output('output', gpu_output_p)
    binding.bind_ortvalue_output('output_surface', gpu_output_s)
    session.run_with_iobinding(binding)

    # 显存 -> 内存（填充到切片对应位置）[gpu_output -> output[nt]]
    output_p[nt_p] = gpu_output_p.numpy()
    output_s[nt_s] = gpu_output_s.numpy()

    # output chunk 写满时, 保存到 zarr 文件, 内存 -> 硬盘
    if nt_p == chunk_p[0] - 1:
        logger.debug(f'proceeding {time}, saving input_p chunk')
        coords_p['time'] = time_range_p
        save_data(output_p, output_file_p, vars_p, coords_p, dim_names_p, write_mode_p, chunk_p)
        # 只有首次保存需要写入, 后续均为追加
        write_mode_p = 'a'

    if nt_s == chunk_s[0] - 1:
        logger.debug(f'proceeding {time}, saving input_s chunk')
        coords_s['time'] = time_range_s
        save_data(output_s, output_file_s, vars_s, coords_s, dim_names_s, write_mode_s, chunk_s)
        write_mode_s = 'a'

    time += dt
    nt += 1

logger.info(f'prediction all done')

# 若存在未填满一个 chunk 的数据，将其裁切并保存
if nt_p != chunk_p[0] - 1:
    coords_p['time'] = time_range_p[:nt_p + 1]
    logger.warning(f'upper output not enough to fill a chunk: {time_range_p[0]} - {time_range_p[nt_p]}')
    save_data(output_p[:nt_p + 1], output_file_p, vars_p, coords_p, dim_names_p, write_mode_p, chunk_p)

if nt_s != chunk_s[0] - 1:
    coords_s['time'] = time_range_s[:nt_s + 1]
    logger.warning(f'surface output not enough to fill a chunk: {time_range_s[0]} - {time_range_s[nt_s]}')
    save_data(output_s[:nt_s + 1], output_file_s, vars_s, coords_s, dim_names_s, write_mode_s, chunk_s)