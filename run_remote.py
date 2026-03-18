import os
import sys
import subprocess
import xarray as xr
import numpy as np
import torch
import onnxruntime as ort
import logging

def get_file(host, remote_path, local_path, chances=2):
    cmd = ['scp', '-q', f'{host}:{remote_path}', local_path]
    for attempt in range(chances):
        try:
            # 执行命令，检查返回码
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                logger.debug(f"[file grabbed] {remote_path.split('/')[-1]}")
                return True
            else:
                logger.error(f"[scp failed] {host}:{remote_path} -> {local_path}")
                logger.error(f"{result.stderr}")
        except Exception as e:
            logger.error(f'[command failed] {host}:{remote_path} -> {local_path}')
            logger.error(f'{e}')
    return False

def get_data(filepath, vars):
    ds = xr.open_dataset(filepath, chunks={'valid_time': -1, 'lat': 180, 'lon': 360})
    stacked = ds[vars].to_array(dim='channel')
    dims = list(stacked.dims)
    dims.remove('valid_time')
    dims.remove('channel')
    new_dims = ['valid_time', 'channel'] + dims
    stacked = stacked.transpose(*new_dims)
    return stacked.values.astype(np.float32)

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
        # 'gpu_mem_limit': 8 * 1024 * 1024 * 1024,    # 设置软限制 8 GB
    }
    return ort.InferenceSession(
        dir, 
        sess_options=options, 
        providers=[('CUDAExecutionProvider', cuda_provider_options)]
        )

def cut_gpu(upper, surface, lat_index, lon_index):
    """
    在显存中选取需要的数据，最后转移到 cpu
    最好将 index 预装载至 gpu
    """
    with torch.no_grad():
    # 地面层处理，将通道重排为 [u10, v10, t2m, msl]，然后空间切片
        cut_surface = surface[:, [1, 2, 3, 0], :, :][:, :, lat_index, :][:, :, :, lon_index]

        # 高空层处理，取前 6 个气压层（1000 hPa - 500 hPa），然后空间切片
        cut_upper = upper[:, :, :6, :, :][:, :, :, lat_index, :][:, :, :, :, lon_index]
        # 将气压维度合并到通道维度
        shape = cut_upper.shape
        cut_upper = cut_upper.reshape(shape[0], shape[1] * shape[2], shape[3], shape[4])

    return torch.cat([cut_surface, cut_upper], dim=1).cpu().numpy()

def final_save(da):
    for lead in output_t:
        logger.info(f'lead: {lead}')

        # 检测全年数据是否完整
        flist = sorted(os.listdir(f'data/daily_saves/{lead}'))
        if len(flist) < 365:
            missing_dates = [str(x) for x in np.arange('2022-01-01', '2023-01-01', dtype=np.datetime64)]
            avail_dates = [file.split('.')[0] for file in flist]
            [missing_dates.remove(x) for x in avail_dates]
            logger.warning('missing dates: ')
            logger.warning(', '.join(missing_dates))
            logger.warning('not making dataset')
            return 0
        
        # 获取每日数据，按时间拼接
        data = []
        for file in flist:
            data.append(np.load(f'data/daily_saves/{lead}/{file}'))
        data = np.concatenate(data, axis=0)

        # 根据预测时效，将实际预测时间向后推移，缺失的用nan填充
        if lead != 0:
            nanfill = np.full([lead, data.shape[1], data.shape[2], data.shape[3]], np.nan)
            data = np.concatenate([nanfill, data[:-lead]], axis=0)

        # 从模板dataarray (da) 复制坐标轴等信息，替换为预测数据
        da_t = da.copy(data=data).to_dataset(name='input')
        logger.info(da_t)
        da_t.to_netcdf(f'data/output/pangu_22_{lead:02}.nc')

        # 删除临时数据
        cmd = ['rm', '-rf', f'data/daily_saves/{lead}']
        subprocess.run(cmd, capture_output=True, text=True, check=False)

def isort(a, A):
    index_map = {val: i for i, val in enumerate(A)}
    return np.array([index_map[val] for val in a])

def setup_logging(level=logging.INFO):
    logger = logging.getLogger('pangu')
    logger.setLevel(level)

    # 文件处理器
    file_handler = logging.FileHandler('./run_remote.log', mode='w')
    file_handler.setLevel(level)

    # 格式器：包含时间、级别、消息
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger

device_type = 'cuda'
device_id = 0
device = f'{device_type}:{device_id}'

# 模板 dataarray，以及需要的经纬度范围在完整范围（ERA5 格式）中所对应的索引
da = xr.open_dataset('/home/chengzy/hubwind/dataset/test_22.nc')['input']
lat_index = isort(da['lat'].data, np.arange(90, -90, -0.25))
lon_index = isort(da['lon'].data, np.arange(0, 360, 0.25))
gpu_lat_idx = torch.from_numpy(lat_index).to(device)
gpu_lon_idx = torch.from_numpy(lon_index).to(device)

os.chdir('/home/chengzy/pangu')

# 日志设置
logger = setup_logging()
def handle_exception(exc_type, exc_value, exc_traceback):
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
sys.excepthook = handle_exception

# 各个时效模型需要的输入时刻
input_t = {
    24: np.array([0]), 
    # 6: np.array([0, 6, 12]), 
    # 3: np.array([0, 6, 12, 18]),
    1: np.array([0, 1])
}
# 产生的预测时效及其对应的索引，0即初始场数据
output_t = sorted(np.concatenate([np.array([0])] + [input_t[lead] + lead for lead in input_t]))
output_t_dict = {x: i for i, x in enumerate(output_t)}

# 生成需要的路径
os.makedirs('data/initial', exist_ok=True)
os.makedirs('data/output', exist_ok=True)
for lead in output_t:
    os.makedirs(f'data/daily_saves/{lead}', exist_ok=True)

# 加载所有模型，并为其初始化 io_binding
logger.debug('loading models ...')
sessions = {lead: session_load(f'models/pangu_weather_{lead}.onnx') for lead in input_t.keys()}
bindings = {lead: sess.io_binding() for lead, sess in sessions.items()}
logger.info('models loaded')

date = np.datetime64('2022-01-01')
curr_mon = 0
while date.item().year == 2022:
    logger.debug(f'grabbing data for {date} ...')

    # 如果月份变更，远程下载新月份的地面数据，下载失败跳到下一月
    if date.item().month > curr_mon:
        curr_mon = date.item().month
        if get_file(
            host='landata', 
            remote_path=f'/stu02/chengzy25/era5_pangu/surface_lev/2022/2022-{curr_mon:02}.nc', 
            local_path='./data/initial/surface_input.nc', 
        ):
            surface_data = get_data('./data/initial/surface_input.nc', ['msl', 'u10', 'v10', 't2m'])
        else:
            date = (date.astype('datetime64[M]') + np.timedelta64(1, 'M')).astype('datetime64[D]')
            logger.error(f'failed to get monthly surface data, skipping to {date}')
            continue
            
    # 下载当天气压层数据，下载失败跳到下一天
    if get_file(
        host='landata', 
        remote_path=f'/stu02/chengzy25/era5_pangu/pressure_lev/2022/2022-{curr_mon:02}/{str(date)}.nc', 
        local_path='./data/initial/upper_input.nc', 
    ):
        upper_data = get_data('./data/initial/upper_input.nc', ['z', 'q', 't', 'u', 'v'])
    else:
        date = date + np.timedelta64(1, 'D')
        logger.error(f'failed to get daily upper data, skipping to {date}')
        continue

    # 在 gpu 上初始化所有预测时效的总输出，每预测 1 小时完全更新一次
    gpu_out_surface = torch.empty((len(output_t), 4, 721, 1440), device=device, dtype=torch.float32)
    gpu_out_upper = torch.empty((len(output_t), 5, 13, 721, 1440), device=device, dtype=torch.float32)
    daily_data = []

    logger.debug(f'prediction of {date} starts')

    for hour in range(0, 24):
        logger.debug(f'running on hour {hour + 1}')
        # 保存初始数据以复用
        gpu_out_surface[0].copy_(torch.from_numpy(surface_data[24 * (date.item().day - 1) + hour]))
        gpu_out_upper[0].copy_(torch.from_numpy(upper_data[hour]))

        # 运行预测模型
        for lead in input_t.keys():
            logger.debug(f'running {lead}-h model')
            torch.cuda.empty_cache()

            for t_i in input_t[lead]:
                # 预测时效在 gpu_out 中的实际位置
                idx_i = output_t_dict[t_i]
                idx_o = output_t_dict[t_i + lead]

                b = bindings[lead]
                # 绑定输入：指向 gpu_out 的输入时效对应切片
                b.bind_input(
                    name='input', device_type=device_type, device_id=device_id, element_type=np.float32, shape=gpu_out_upper[idx_i].shape, buffer_ptr=gpu_out_upper[idx_i].data_ptr()
                )
                b.bind_input(
                    name='input_surface', device_type=device_type, device_id=device_id, element_type=np.float32, shape=gpu_out_surface[idx_i].shape, buffer_ptr=gpu_out_surface[idx_i].data_ptr()
                )
                # 绑定输出：写进 gpu_out 的输出时效对应切片
                b.bind_output(
                    name='output', device_type=device_type, device_id=device_id, element_type=np.float32, shape=gpu_out_upper[idx_o].shape, buffer_ptr=gpu_out_upper[idx_o].data_ptr()
                )
                b.bind_output(
                    name='output_surface', device_type=device_type, device_id=device_id, element_type=np.float32, shape=gpu_out_surface[idx_o].shape, buffer_ptr=gpu_out_surface[idx_o].data_ptr()
                )
                sessions[lead].run_with_iobinding(b)

        # 裁切需要的数据
        daily_data.append(cut_gpu(gpu_out_upper, gpu_out_surface, gpu_lat_idx, gpu_lon_idx))

    # 每跑完一天数据，保存一次临时数据
    data = np.stack(daily_data, axis=1)
    for lead, idx in output_t_dict.items():
        np.save(f'data/daily_saves/{lead}/{date}.npy', data[idx])
    logger.info(f'prediction of {date} done, outcomes saved')

    date = date + np.timedelta64(1, 'D')

logger.info('prediction all done, making dataset')
# 将全年数据打包为nc文件
final_save(da)
                

