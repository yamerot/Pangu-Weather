import os
import sys
import subprocess
import xarray as xr
import numpy as np
import onnxruntime as ort
import logging

def get_file(host, remote_path, local_path, chances=2):
    return True
    cmd = ['scp', '-q', f'{host}:{remote_path}', local_path]
    for attempt in range(chances):
        try:
            # 执行命令，检查返回码
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                logger.info(f"[file grabbed] {remote_path.split('/')[-1]}")
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
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = True
    options.enable_mem_reuse = True
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    cuda_provider_options = {
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 9 * 1024 * 1024 * 1024,    # 设置软限制9GB
        'cudnn_conv_algo_search': 'HEURISTIC',
    }
    return ort.InferenceSession(
        dir, 
        sess_options=options, 
        providers=[('CUDAExecutionProvider', cuda_provider_options)]
        )

def cut(upper, surface, lat_index, lon_index):
    cut1 = surface[:, :, np.array((1, 2, 3, 0)), :, :][:, :, :, lat_index, lon_index]
    cut2 = upper[:, :, :, np.arange(6), :, :][:, :, :, :, lat_index, lon_index] 
    cut2 = cut2.reshape(upper.shape[0], upper.shape[1], 30, lat_index.shape[0], lon_index.shape[0])
    return np.concatenate([cut1, cut2], axis=1)

def daily_save(date, daily_data, output_t_dict):
    data = np.stack(daily_data, axis=1)
    for lead in output_t_dict:
        np.save(f'data/daily_saves/{lead}/{str(date)}.npy', data[output_t_dict[lead]])

def final_save(da):
    for lead in output_t:
        logger.info(f'\nlead: {lead}')

        # 检测全年数据是否完整
        flist = sorted(os.listdir(f'data/daily_saves/{lead}'))
        if len(flist) < 365:
            missing_dates = [str(x) for x in np.arange('2022-01-01', '2023-01-01', dtype=np.datetime64)]
            avail_dates = [file.split('.')[0] for file in flist]
            [missing_dates.remove(x) for x in avail_dates]
            logger.error('missing dates: ', *missing_dates)
            logger.error('not making dataset')
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

def setup_logging():
    logger = logging.getLogger('pangu')
    logger.setLevel(logging.INFO)

    # 文件处理器（追加模式，每日轮转可选）
    file_handler = logging.FileHandler('./run_remote.log', mode='w')
    file_handler.setLevel(logging.INFO)

    # 格式器：包含时间、级别、消息
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger

da = xr.open_dataset('/home/chengzy/hubwind/dataset/test_22.nc')['input']
lat_index = isort(da['lat'].data, np.arange(90, -90, -0.25))
lon_index = isort(da['lon'].data, np.arange(0, 360, 0.25))

os.chdir('/home/chengzy/pangu')

logger = setup_logging()

def handle_exception(exc_type, exc_value, exc_traceback):
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

# 各个时效模型需要的输入时刻
input_t = {
    24: np.array([0]), 
    6: np.array([0, 6, 12]), 
    3: np.array([0, 6]),
    1: np.array([0, 1, 3, 4])
}
# 需要的预测时效及其对应的索引，0即初始场数据
output_t = sorted(np.concatenate([np.array([0])] + [input_t[lead] + lead for lead in input_t]))
output_t_dict = {x: i for i, x in enumerate(output_t)}

# 生成需要的路径
os.makedirs('data/initial', exist_ok=True)
os.makedirs('data/output', exist_ok=True)
for lead in output_t:
    os.makedirs(f'data/daily_saves/{lead}', exist_ok=True)

date = np.datetime64('2022-01-01')
curr_mon = 0
while date.item().year == 2022:
    # 如果月份变更，远程下载新月份的地面数据，下载失败跳到下一月
    if date.item().month > curr_mon:
        curr_mon = date.item().month
        logger.info(f'running on month: {curr_mon}')
        if get_file(
            host='landata', 
            remote_path=f'/stu02/chengzy25/era5_pangu/surface_lev/2022-{curr_mon:02}.nc', 
            local_path='./data/initial/surface_input.nc', 
            ):
            surface_data = get_data('./data/initial/surface_input.nc', ['msl', 'u10', 'v10', 't2m'])
        else:
            date = date + np.timedelta64(1, 'M')
            continue
            
    # 下载当天气压层数据，下载失败跳到下一天
    if get_file(
        host='landata', 
        remote_path=f'/stu02/chengzy25/era5_pangu/pressure_lev/2022-{curr_mon:02}/{str(date)}.nc', 
        local_path='./data/initial/upper_input.nc', 
        ):
        upper_data = get_data('./data/initial/upper_input.nc', ['z', 'q', 't', 'u', 'v'])
    else:
        date = date + np.timedelta64(1, 'D')
        continue
    
    # 初始化当日所有时刻、所有预测时效的总输出
    output_surface = np.empty([24, len(output_t), 4, 721, 1440], dtype=np.float32)
    output_upper = np.empty([24, len(output_t), 5, 13, 721, 1440], dtype=np.float32)

    for model_lead in input_t:
        # 加载模型
        session = session_load(f'models/pangu_weather_{model_lead}.onnx')

        for hour in range(0, 24):
            # 保存初始数据以复用
            output_surface[hour, 0] = surface_data[24 * (date.item().day - 1) + hour].copy()
            output_upper[hour, 0] = upper_data[hour].copy()

            for t_i in input_t[model_lead]:
                idx_i = output_t_dict[t_i]
                pred_upper, pred_surface = session.run(None, {'input':output_upper[hour, idx_i], 'input_surface':output_surface[hour, idx_i]})

                idx_o = output_t_dict[t_i + model_lead]
                output_surface[hour, idx_o] = pred_surface.copy()
                output_upper[hour, idx_o] = pred_upper.copy()

    # 裁切需要的数据，按照对应的预测时效保存
    daily_data.append(cut(output_upper, output_surface, lat_index, lon_index))

    # 每跑完一天数据，保存一次临时数据，只保存需要的预测时效
    daily_save(date, daily_data, output_t_dict)
    date = date + np.timedelta64(1, 'D')

    break

logger.info('prediction all done, making dataset')
# 将全年数据打包为nc文件
final_save(da)
                

