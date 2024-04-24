import os
import numpy as np
import pickle
from collections import OrderedDict
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


TIME_LIST = [x / 100 for x in range(5, 31)]

USE_360VidStr = False  # whether to use 360VidStr dataset  # NOTE: 废弃, 请固定为False
HW_SIZE = 20  # historical window size (100Hz)  # NOTE: from 100 to 20 (性能更好)
SAMPLE = 4  # 历史数据采样步长  # NOTE: from 10 to 4 (性能更好)
STRIDE = 5   # 生成数据时时间窗口的移动步长  # NOTE: from 10 to 5 (性能更好)

TRAIN_FLAG = True  # whether to train the model


# copy from e3po.utils.motion_trace import read_client_log:
def read_client_log(client_log_path, interval, client_log_user_index):
    """
    Read and process client logs, return client_record dictionary.

    Parameters
    ----------
    client_log_path : str
        Path to client log file.
    interval : int
        The motion sampling interval of the original file, in milliseconds.
    client_log_user_index : int
        Indicate which user's data to use。

    Returns
    -------
    dict
        Client record before frame filling. Its insertion order is in ascending order of timestamp.
            key = timestamp,
            value = {'yaw': yaw, 'pitch': pitch, 'scale': scale}
    """
    client_record = OrderedDict()
    with open(client_log_path, 'r') as f:
        _ = f.readline()
        index = 0
        while True:
            line_pitch = f.readline()[:-1].split(' ')
            line_yaw = f.readline()[:-1].split(' ')
            if len(line_pitch) <= 1:
                break
            index += 1
            if index != client_log_user_index:
                continue

            for i in range(len(line_yaw)):
                if i < 280:
                    scale = 2
                elif i < 300:
                    scale = 4
                elif i < 400:
                    scale = 2
                elif i < 500:
                    scale = 4
                else:
                    scale = 2
                client_record[i * interval] = {'yaw': float(line_yaw[i]), 'pitch': float(line_pitch[i]), 'roll': 0, 'scale': scale}

    return client_record


# theta in the range 0, to 2*pi, theta can be negative, e.g. cartesian_to_eulerian(0, -1, 0) = (-pi/2, pi/2) (is equal to (3*pi/2, pi/2))
# phi in the range 0 to pi (0 being the north pole, pi being the south pole)
def eulerian_to_cartesian(phi, theta):
    x = np.cos(theta)*np.sin(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(phi)
    return np.array([x, y, z])

def cartesian_to_eulerian(x, y, z):
    r = np.sqrt(x*x+y*y+z*z)
    theta = np.arctan2(y, x)
    phi = np.arccos(z/r)
    # remainder is used to transform it in the positive range (0, 2*pi)
    theta = np.remainder(theta, 2*np.pi)
    return phi, theta


def get_data(opt, pp_time):
    client_log_path = os.path.join('e3po', 'source', 'motion_trace', opt['motion_trace']['motion_file'])
    assert os.path.exists(client_log_path), f'[error] {client_log_path} doesn\'t exist'
    interval = int(1000 / opt['motion_trace']['sample_frequency'])
    client_log_user_index = opt['motion_trace']['column_idx']
    client_record = read_client_log(client_log_path, interval, client_log_user_index)
    # print(client_record)  # OrderedDict([(0, {'yaw': 0.0, 'pitch': 0.0, 'roll': 0, 'scale': 2}), ...])

    # 将client_record转换为np.array
    client_record = np.array([[x['pitch'], x['yaw']] for x in list(client_record.values())])
    # print(client_record.shape)  # (6299, 2)

    X = []
    y = []
    pp_pos  = int(pp_time * 100)  # prediction point position (1s)

    for i in range(HW_SIZE, len(client_record) - pp_pos, STRIDE):
        X.append(client_record[i-HW_SIZE+SAMPLE-1:i:SAMPLE, :2])
        y.append(client_record[i+pp_pos, :2])
    X, y = np.array(X), np.array(y)
    # print(X.shape, y.shape)  # (600, HW_SIZE//SAMPLE, 2) (600, 2)
    # print(X[5], y[5])
    
    # NOTE: (pitch, yaw) --> (x, y, z):
    # pitch --> phi:
    X[:, :, 0] += np.pi / 2
    y[:, 0] += np.pi / 2
    # yaw --> theta: (no need to change)
    X = np.array([eulerian_to_cartesian(x[0], x[1]) for x in X.reshape(-1, 2)]).reshape(X.shape[0], X.shape[1], 3)
    y = np.array([eulerian_to_cartesian(x[0], x[1]) for x in y])
    # print(X.shape, y.shape)  # (600, HW_SIZE//SAMPLE, 3) (600, 3)
    # print(X[5], y[5])
    
    # # NOTE: (x, y, z) --> (pitch, yaw): (测试函数cartesian_to_eulerian是否能够将eulerian_to_cartesian的结果逆转回去)
    # X = np.array([cartesian_to_eulerian(x[0], x[1], x[2]) for x in X.reshape(-1, 3)]).reshape(X.shape[0], X.shape[1], 2)
    # y = np.array([cartesian_to_eulerian(x[0], x[1], x[2]) for x in y])
    # # phi --> pitch:
    # X[:, :, 0] -= np.pi / 2
    # y[:, 0] -= np.pi / 2
    # # theta --> yaw: (no need to change)
    # print(X.shape, y.shape)  # (600, HW_SIZE//SAMPLE, 2) (600, 2)
    # print(X[5], y[5])
    
    return X, y


if __name__ == '__main__':
    opt1 = {
        'motion_trace': {
            'motion_file': 'release_video_1_motion_1.txt',  # full name of motion trace file
            'sample_frequency': 100,              # the sample frequency of motion trace file
            'motion_frequency': 100,             # the update frequency of e3po
            'column_idx': 1,                     # the user index in the log file
        }
    }

    opt2 = {
        'motion_trace': {
            'motion_file': 'release_video_1_motion_2.txt',  # full name of motion trace file
            'sample_frequency': 100,              # the sample frequency of motion trace file
            'motion_frequency': 100,             # the update frequency of e3po
            'column_idx': 1,                     # the user index in the log file
        }
    }

    for pp_time in TIME_LIST:
        X1, y1 = get_data(opt1, pp_time)
        X2, y2 = get_data(opt2, pp_time)

        test_size = 100
        X_test, y_test = X1[:test_size], y1[:test_size]
        X_train, y_train = X1[test_size:], y1[test_size:]
        # X_train, y_train = X_test, y_test


        if USE_360VidStr:  # 更改训练数据为: 读取 AggregatedDataset 中的文件生成训练数据:
            data_folder = os.path.join('e3po', 'source', 'motion_trace', 'AggregatedDataset')
            assert os.path.exists(data_folder), f'[error] {data_folder} doesn\'t exist'
            hw_size = 10  # historical window size (1s) (10Hz)
            pp_pos  = int(pp_time * 10)  # prediction point position (1s)
            stride  = 5   # 生成数据时时间窗口的移动步长 (0.5s)
            sample  = 1   # 对历史数据的采样步长 (10Hz -> 10Hz)

            X = []
            y = []
            for file_name in os.listdir(data_folder):
                if file_name == 'readMe.txt':
                    continue
                try:  # 读取 .txt 文件, 以 ' ' 为分隔符
                    data = np.loadtxt(os.path.join(data_folder, file_name), delimiter=' ')
                except ValueError:  # ValueError("Wrong number of columns at line %d"
                    continue
                for uid in range(len(data)//2):
                    user_data = data[uid*2+1 : uid*2+3].T  # (610, 2) (pitch, yaw)
                    for i in range(hw_size, len(user_data) - pp_pos, stride):
                        X.append(user_data[i-hw_size:i:sample, :2])
                        y.append(user_data[i+pp_pos, :2])
            X, y = np.array(X), np.array(y)
            # pitch --> phi:
            X[:, :, 0] += np.pi / 2
            y[:, 0] += np.pi / 2
            # yaw --> theta: (no need to change)
            X = np.array([eulerian_to_cartesian(x[0], x[1]) for x in X.reshape(-1, 2)]).reshape(X.shape[0], X.shape[1], 3)
            y = np.array([eulerian_to_cartesian(x[0], x[1]) for x in y])
            X_train, y_train = X, y
            # print(X_train.shape, y_train.shape)  # (814704, 10, 2) (814704, 2)

        model_folder = os.path.join('e3po', 'approaches', 'bitedance', 'vp')
        model_path = os.path.join(model_folder, 'lr_xyz_' + str(pp_time) + ('' if not USE_360VidStr else '_360VidStr') + '.pkl')
        if TRAIN_FLAG:
            # 训练模型:
            model = LinearRegression()
            model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
            # 保存模型:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        else:
            # 加载模型:
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
        
        # 使用测试集测试模型性能:
        y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))

        print('pp_time:', pp_time)
        print('mse (v):', mean_squared_error(y_test, y_pred))
        print('mae (v):', mean_absolute_error(y_test, y_pred))
        print('r2 score (^):', r2_score(y_test, y_pred))
        print('-'*50)