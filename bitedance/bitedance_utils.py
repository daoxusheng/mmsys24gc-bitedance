import os
import os.path as osp
import pickle
import shutil
from copy import deepcopy

import cv2
import numpy as np
import yaml
from e3po.approaches.bitedance.vp.lr import (
    SAMPLE,
    TIME_LIST,
    USE_360VidStr,
    cartesian_to_eulerian,
    eulerian_to_cartesian,
)
from e3po.utils.data_utilities import (
    extract_frame,
    remove_temp_files,
    transform_projection,
)
from e3po.utils.json import get_video_json_size
from e3po.utils.projection_utilities import (
    _3d_polar_coord_to_pixel_coord,
    fov_to_3d_polar_coord,
    pixel_coord_to_tile,
)

STAGE_TIME = 0.04
MAX_TIMES = 1.1
TOLERANCE = 0.02

# load vp models
vp_models = {}
vp_model_folder = os.path.join("e3po", "approaches", "bitedance", "vp")
for pp_time in TIME_LIST:
    with open(
        os.path.join(
            vp_model_folder,
            "lr_xyz_"
            + str(pp_time)
            + ("" if not USE_360VidStr else "_360VidStr")
            + ".pkl",
        ),
        "rb",
    ) as file:
        vp_models[pp_time] = pickle.load(file)


def predict_motion_tile(
    motion_history, motion_history_size, motion_prediction_size, pp_time
):
    """
    Predicting motion with given historical information and prediction window size.
    (As an example, users can implement their customized function.)

    Parameters
    ----------
    motion_history: dict
        a dictionary recording the historical motion, with the following format:

    motion_history_size: int (单位: 个)
        the size of motion history to be used for predicting
    motion_prediction_size: int (单位: 个)
        the size of motion to be predicted

    Returns
    -------
    list
        The predicted record list, which sequentially store the predicted motion of the future pw chunks.
         Each motion dictionary is stored in the following format:
            {'yaw ': yaw,' pitch ': pitch,' scale ': scale}
    """
    # Use exponential smoothing to predict the angle of each motion within pw for yaw and pitch.
    a = 0.3  # Parameters for exponential smoothing prediction
    hw = [d["motion_record"] for d in motion_history]
    if len(hw) < motion_history_size:
        predicted_motion = list(hw)[0]
        for motion_record in list(hw)[-motion_history_size:]:
            predicted_motion["yaw"] = (
                a * predicted_motion["yaw"] + (1 - a) * motion_record["yaw"]
            )
            predicted_motion["pitch"] = (
                a * predicted_motion["pitch"] + (1 - a) * motion_record["pitch"]
            )
            predicted_motion["scale"] = (
                a * predicted_motion["scale"] + (1 - a) * motion_record["scale"]
            )

        # The current prediction method implemented is to use the same predicted motion for all chunks in pw.
        predicted_record = []
        for i in range(motion_prediction_size):
            predicted_record.append(deepcopy(predicted_motion))

        return predicted_record

    X = np.array([[x["pitch"], x["yaw"]] for x in hw[-motion_history_size:]])[
        SAMPLE - 1 :: SAMPLE
    ]

    X = X[np.newaxis, :, :]
    X[:, :, 0] += np.pi / 2
    X = np.array([eulerian_to_cartesian(x[0], x[1]) for x in X.reshape(-1, 2)]).reshape(
        X.shape[0], X.shape[1], 3
    )

    vp_model = vp_models[round(pp_time, 2)]
    y = vp_model.predict(X.reshape(X.shape[0], -1))
    y = np.array([cartesian_to_eulerian(x[0], x[1], x[2]) for x in y])
    y[:, 0] -= np.pi / 2

    return [{"pitch": y[0][0], "yaw": y[0][1]}]


def tile_decision(predicted_record, video_size, range_fov, chunk_idx, user_data):
    """
    Deciding which tiles should be transmitted for each chunk, within the prediction window
    (As an example, users can implement their customized function.)

    Parameters
    ----------
    predicted_record: dict
        the predicted motion, with format {yaw: , pitch: , scale:}, where
        the parameter 'scale' is used for transcoding approach
    video_size: dict
        the recorded whole video size after video preprocessing
    range_fov: list
        degree range of fov, with format [height, width]
    chunk_idx: int
        index of current chunk
    user_data: dict
        user related data structure, necessary information for tile decision

    Returns
    -------
    tile_record: list
        the decided tile list of current update, each item is the chunk index
    """
    # The current tile decision method is to sample the fov range corresponding to the predicted motion of each chunk,
    # and the union of the tile sets mapped by these sampling points is the tile set to be transmitted.
    config_params = user_data["config_params"]
    tile_record = []

    if config_params["background_flag"]:
        if -1 not in user_data["latest_decision"]:
            tile_record.append(-1)

    stride = 0.1
    for times in np.arange(stride, MAX_TIMES + stride, stride):
        _range_fov = [int(x * times) for x in range_fov]
        sampling_size = [int(50 * times), int(50 * times)]
        converted_width = user_data["config_params"]["converted_width"]
        converted_height = user_data["config_params"]["converted_height"]

        for predicted_motion in predicted_record:
            _3d_polar_coord = fov_to_3d_polar_coord(
                [float(predicted_motion["yaw"]), float(predicted_motion["pitch"]), 0],
                _range_fov,
                sampling_size,
            )
            pixel_coord = _3d_polar_coord_to_pixel_coord(
                _3d_polar_coord,
                config_params["projection_mode"],
                [converted_height, converted_width],
            )
            coord_tile_list = pixel_coord_to_tile(
                pixel_coord, config_params["total_tile_num"], video_size, chunk_idx
            )
            unique_tile_list = [int(item) for item in np.unique(coord_tile_list)]
            for tile_idx in unique_tile_list:
                if tile_idx not in tile_record:
                    tile_record.append(tile_idx)

    return tile_record


def init_user(user_data, video_info):
    """
    Initialization function, users initialize their parameters based on the content passed by E3PO

    Parameters
    ----------
    user_data: None
        the initialized user_data is none, where user can store their parameters
    video_info: dict
        video information of original video, user can perform preprocessing according to their requirement

    Returns
    -------
    user_data: dict
        the updated user_data
    """

    user_data = user_data or {}
    user_data["video_info"] = video_info
    user_data["config_params"] = read_config(video_info)
    user_data["chunk_idx"] = -1
    user_data["download_stage"] = 0
    user_data["pp_time"] = TIME_LIST[0]
    user_data = update_pd_time(user_data, TIME_LIST[-1])

    return user_data


def read_config(video_info):
    """
    read the user-customized configuration file as needed

    Returns
    -------
    config_params: dict
        the corresponding config parameters
    """

    config_path = os.path.dirname(os.path.abspath(__file__)) + "/bitedance.yml"
    with open(config_path, "r", encoding="UTF-8") as f:
        opt = yaml.safe_load(f.read())["approach_settings"]

    background_flag = opt["background"]["background_flag"]
    projection_mode = opt["approach"]["projection_mode"]
    converted_projection_mode = opt["video"]["converted"]["projection_mode"]
    src_projection_mode = video_info["projection"]

    if src_projection_mode == converted_projection_mode:
        opt["video"]["converted"]["height"] = video_info["height"]
        opt["video"]["converted"]["width"] = video_info["width"]
    converted_height = opt["video"]["converted"]["height"]
    converted_width = opt["video"]["converted"]["width"]
    background_height = opt["background"]["height"]
    background_width = opt["background"]["width"]
    tile_height_num = opt["video"]["tile_height_num"]
    tile_width_num = opt["video"]["tile_width_num"]
    total_tile_num = tile_height_num * tile_width_num
    tile_width = int(opt["video"]["converted"]["width"] / tile_width_num)
    tile_height = int(opt["video"]["converted"]["height"] / tile_height_num)
    if background_flag:
        background_info = {
            "width": opt["background"]["width"],
            "height": opt["background"]["height"],
            "background_projection_mode": opt["background"]["projection_mode"],
        }
    else:
        background_info = {}

    motion_history_size = int(opt["video"]["hw_size"] * 100)  # 1s * 100Hz = 100 个
    motino_prediction_size = opt["video"]["pw_size"]  # 1 个
    ffmpeg_settings = opt["ffmpeg"]
    if not ffmpeg_settings["ffmpeg_path"]:
        assert shutil.which("ffmpeg"), "[error] ffmpeg doesn't exist"
        ffmpeg_settings["ffmpeg_path"] = shutil.which("ffmpeg")
    else:
        assert os.path.exists(
            ffmpeg_settings["ffmpeg_path"]
        ), f'[error] {ffmpeg_settings["ffmpeg_path"]} doesn\'t exist'

    config_params = {
        "background_flag": background_flag,
        "converted_height": converted_height,
        "converted_width": converted_width,
        "background_height": background_height,
        "background_width": background_width,
        "tile_height_num": tile_height_num,
        "tile_width_num": tile_width_num,
        "total_tile_num": total_tile_num,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "background_info": background_info,
        "motion_history_size": motion_history_size,
        "motion_prediction_size": motino_prediction_size,
        "ffmpeg_settings": ffmpeg_settings,
        "projection_mode": projection_mode,
        "converted_projection_mode": converted_projection_mode,
    }

    return config_params


def update_decision_info(user_data, tile_record, curr_ts):
    """
    update the decision information

    Parameters
    ----------
    user_data: dict
        user related parameters and information
    tile_record: list
        recode the tiles should be downloaded
    curr_ts: int
        current system timestamp

    Returns
    -------
    user_data: dict
        updated user_data
    """
    # update latest_decision
    for i in range(len(tile_record)):
        if tile_record[i] not in user_data["latest_decision"]:
            user_data["latest_decision"].append(tile_record[i])
    if user_data["config_params"]["background_flag"]:
        if -1 not in user_data["latest_decision"]:
            user_data["latest_decision"].append(-1)

    # update chunk_idx & latest_decision
    if (
        curr_ts
        >= user_data["video_info"]["pre_download_duration"]
        + user_data["next_download_idx"]
        * user_data["video_info"]["chunk_duration"]
        * 1000
        + (user_data["video_info"]["chunk_duration"] - user_data["pd_time"]) * 1000
    ):
        user_data["next_download_idx"] += 1
        user_data["latest_decision"] = []

    left = (
        user_data["video_info"]["pre_download_duration"]
        + user_data["next_download_idx"]
        * user_data["video_info"]["chunk_duration"]
        * 1000
    )
    if curr_ts >= left:
        user_data["download_stage"] = 1 + int((curr_ts - left) / (STAGE_TIME * 1000))
    else:
        user_data["download_stage"] = 0

    return user_data


def get_total_delay(dl_list, video_size, network_stats):
    if len(dl_list) == 0:
        return 0
    else:
        assert len(dl_list) == 1
    row = dl_list[0]
    chunk_idx = row["chunk_idx"]
    chunk_size = 0
    for tile_id in row["decision_data"]["tile_info"]:
        chunk_size += get_video_json_size(video_size, chunk_idx, tile_id)
    download_delay = chunk_size / network_stats[0]["bandwidth"] / 1000
    rendering_delay = 10
    total_delay = download_delay + network_stats[0]["rtt"] + rendering_delay
    return total_delay


def modify_dl_list(dl_list, video_size, network_stats, user_data):
    """
    Modify the dl_list to meet the delay requirement
    """
    if not dl_list:
        return dl_list

    while (
        dl_list[0]["decision_data"]["tile_info"]
        and get_total_delay(dl_list, video_size, network_stats)
        > user_data["max_delay"] * 1000
    ):
        print(f'pop from: {dl_list[0]["decision_data"]["tile_info"]}')
        dl_list[0]["decision_data"]["tile_info"].pop()

    if not dl_list[0]["decision_data"]["tile_info"]:
        dl_list = []

    return dl_list


# NOTE: copy from e3po/utils/data_utilities.py
def generate_nc_dl_list(chunk_idx, tile_record, user_data):
    """
    Based on the decision result, generate the required dl_list to be returned in the specified format.
    (As an example, users can implement their corresponding function.)

    Parameters
    ----------
    chunk_idx: int
        the index of current chunk
    tile_record: list
        the decided tile list of current update, each list item is the chunk index
    latest_result: list
        recording the latest decision result
    dl_list: list
        the decided tile list

    Returns
    -------
    dl_list: list
        updated dl_list
    """
    nc_tile_result = []  # next chunk tile result

    for i in range(len(tile_record)):
        tile_idx = tile_record[i]

        if tile_idx != -1:
            tile_id = f"chunk_{str(chunk_idx+1).zfill(4)}_tile_{str(tile_idx).zfill(3)}"
        else:
            tile_id = f"chunk_{str(chunk_idx+1).zfill(4)}_background"
        nc_tile_result.append(tile_id)

    if user_data["config_params"]["background_flag"] and -1 not in tile_record:
        nc_tile_result.append(f"chunk_{str(chunk_idx+1).zfill(4)}_background")

    if (
        chunk_idx + 1
        >= user_data["video_info"]["duration"]
        / user_data["video_info"]["chunk_duration"]
    ):
        nc_dl_list = []
    else:
        nc_dl_list = [
            {"chunk_idx": chunk_idx + 1, "decision_data": {"tile_info": nc_tile_result}}
        ]

    return nc_dl_list


def update_pd_time(user_data, pd_time):
    user_data["pd_time"] = pd_time  # NOTE: pre-download time (s)
    motion_frequency = 100
    user_data["max_delay"] = user_data["pd_time"] - 1 / motion_frequency
    return user_data


def segment_video(
    ffmpeg_settings, ori_video_uri, dst_video_folder, segmentation_info, tile_info
):
    out_w = segmentation_info["segment_out_info"]["width"]
    out_h = segmentation_info["segment_out_info"]["height"]
    start_w = segmentation_info["start_position"]["width"]
    start_h = segmentation_info["start_position"]["height"]

    result_frame_path = osp.join(dst_video_folder, f"%d.png")

    chunk_idx = tile_info["chunk_idx"]
    chunk_duration = tile_info["chunk_duration"]

    s_1 = str(chunk_idx * chunk_duration % 60).zfill(2)
    m_1 = str(chunk_idx * chunk_duration // 60).zfill(2)
    h_1 = str(chunk_idx * chunk_duration // 3600).zfill(2)
    s_2 = str(((chunk_idx + 1) * chunk_duration) % 60).zfill(2)
    m_2 = str(((chunk_idx + 1) * chunk_duration) // 60).zfill(2)
    h_2 = str(((chunk_idx + 1) * chunk_duration) // 3600).zfill(2)

    cmd = (
        f"{ffmpeg_settings['ffmpeg_path']} "
        f"-i {ori_video_uri} "
        f"-threads {ffmpeg_settings['thread']} "
        f"-ss {h_1}:{m_1}:{s_1} "
        f"-to {h_2}:{m_2}:{s_2} "
        f'-vf "crop={out_w}:{out_h}:{start_w}:{start_h}" '
        f"-q:v 2 -f image2 {result_frame_path} "
        f"-loglevel {ffmpeg_settings['loglevel']}"
    )

    os.system(cmd)


def resize_video(
    ffmpeg_settings, ori_video_uri, dst_video_folder, dst_video_info, tile_info
):
    dst_video_w = dst_video_info["width"]
    dst_video_h = dst_video_info["height"]

    result_frame_path = osp.join(dst_video_folder, f"%d.png")

    chunk_idx = tile_info["chunk_idx"]
    chunk_duration = tile_info["chunk_duration"]

    s_1 = str(chunk_idx * chunk_duration % 60).zfill(2)
    m_1 = str(chunk_idx * chunk_duration // 60).zfill(2)
    h_1 = str(chunk_idx * chunk_duration // 3600).zfill(2)
    s_2 = str(((chunk_idx + 1) * chunk_duration) % 60).zfill(2)
    m_2 = str(((chunk_idx + 1) * chunk_duration) // 60).zfill(2)
    h_2 = str(((chunk_idx + 1) * chunk_duration) // 3600).zfill(2)

    cmd = (
        f"{ffmpeg_settings['ffmpeg_path']} "
        f"-i {ori_video_uri} "
        f"-threads {ffmpeg_settings['thread']} "
        f"-ss {h_1}:{m_1}:{s_1} "
        f"-to {h_2}:{m_2}:{s_2} "
        f'-vf "scale={dst_video_w}x{dst_video_h}'
        f',setdar={dst_video_w}/{dst_video_h}" '
        f"-q:v 2 -f image2 {result_frame_path} "
        f"-loglevel {ffmpeg_settings['loglevel']}"
    )

    os.system(cmd)


def transcode_ori_video(
    ori_video_uri,
    src_proj,
    dst_proj,
    src_resolution,
    dst_resolution,
    dst_video_folder,
    chunk_info,
    ffmpeg_settings,
):
    """
    Transcoding videos with different projection formats and different resolutions

    Parameters
    ----------
    source_video_uri: str
        source video uri
    src_proj: str
        source video projection
    dst_proj: str
        destination video projection
    src_resolution: list
        source video resolution with format [height, width]
    dst_resolution: list
        destination video resolution with format [height, width]
    dst_video_folder: str
        path of the destination video
    chunk_info: dict
        chunk information
    ffmpeg_settings: dict
        ffmpeg related information, with format {ffmpeg_path, log_level, thread}

    Returns
    -------
    transcode_video_uri: str
        uri (uniform resource identifier) of the transcode video
    """

    tmp_cap = cv2.VideoCapture()
    assert tmp_cap.open(ori_video_uri), f"[error] Can't read video[{ori_video_uri}]"
    frame_count = int(tmp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tmp_cap.release()

    for frame_idx in range(frame_count):
        source_frame = extract_frame(ori_video_uri, frame_idx, ffmpeg_settings)
        pixel_coord = transform_projection(
            dst_proj, src_proj, dst_resolution, src_resolution
        )
        dstMap_u, dstMap_v = cv2.convertMaps(
            pixel_coord[0].astype(np.float32),
            pixel_coord[1].astype(np.float32),
            cv2.CV_16SC2,
        )
        transcode_frame = cv2.remap(source_frame, dstMap_u, dstMap_v, cv2.INTER_LINEAR)
        transcode_frame_uri = osp.join(dst_video_folder, f"{frame_idx}.png")
        cv2.imwrite(
            transcode_frame_uri, transcode_frame, [cv2.IMWRITE_JPEG_QUALITY, 100]
        )

    transcode_video_uri = ori_video_uri.split(".")[0] + "_bitedance_transcode.mp4"
    # Ensure the highest possible quality
    cmd = (
        f"{ffmpeg_settings['ffmpeg_path']} "
        f"-start_number 0 "
        f"-i {osp.join(dst_video_folder, '%d.png')} "
        f"-threads {ffmpeg_settings['thread']} "
        f"-c:v libx264 "
        f"-preset slow "
        f"-g 30 "
        f"-bf 0 "
        f"-qp {10} "
        f"-y {transcode_video_uri} "
        f"-loglevel {ffmpeg_settings['loglevel']}"
    )
    os.system(cmd)
    remove_temp_files(dst_video_folder)

    return transcode_video_uri


def tile_segment_info(chunk_info, user_data):
    """
    Generate the information for the current tile, with required format
    Parameters
    ----------
    chunk_info: dict
        chunk information
    user_data: dict
        user related information
    Returns
    -------
    tile_info: dict
        tile related information, with format {chunk_idx:, tile_idx:}
    segment_info: dict
        segmentation related information, with format
        {segment_out_info:{width:, height:}, start_position:{width:, height:}}
    """
    tile_idx = user_data["tile_idx"]
    tile_width_num = user_data["config_params"]["tile_width_num"]
    tile_height_num = user_data["config_params"]["tile_height_num"]
    index_width = tile_idx % tile_width_num  # determine which col
    index_height = tile_idx // tile_width_num  # determine which row
    basic_tile_width = user_data["config_params"]["tile_width"]  # basic tile - width
    basic_tile_height = user_data["config_params"]["tile_height"]  # basic tile - height
    extra_tile_width = (
        user_data["config_params"]["converted_width"]
        - basic_tile_width * tile_width_num
    )
    extra_tile_height = (
        user_data["config_params"]["converted_height"]
        - basic_tile_height * tile_height_num
    )
    tile_width = basic_tile_width
    tile_height = basic_tile_height
    user_data["segment_flag"] = True
    # check which rigion the tile is in
    print(f"=== RIGION tile: {tile_idx}, col,row: {index_width},{index_height} ===")
    if index_height in [0]:
        if index_width in [0]:
            tile_width = basic_tile_width * 12 + extra_tile_width
            tile_height = basic_tile_height
        else:
            user_data["segment_flag"] = False
    elif index_height in [15]:
        if index_width in [0]:
            tile_width = basic_tile_width * 12 + extra_tile_width
            tile_height = basic_tile_height + extra_tile_height
        else:
            user_data["segment_flag"] = False
    elif index_height in [1, 13]:
        if index_width in [0, 4]:
            tile_width = basic_tile_width * 4
            tile_height = basic_tile_height * 2
        elif index_width in [8]:
            tile_width = basic_tile_width * 4 + extra_tile_width
            tile_height = basic_tile_height * 2
        else:
            user_data["segment_flag"] = False
    elif index_height in [3, 11]:
        if index_width in [0, 2, 4, 6, 8]:
            tile_width = basic_tile_width * 2
            tile_height = basic_tile_height * 2
        elif index_width in [10]:
            tile_width = basic_tile_width * 2 + extra_tile_width
            tile_height = basic_tile_height * 2
        else:
            user_data["segment_flag"] = False
    elif index_height in [5]:
        if index_width in [0, 2, 4, 6, 8]:
            tile_width = basic_tile_width * 2
            tile_height = basic_tile_height * 6
        elif index_width in [10]:
            tile_width = basic_tile_width * 2 + extra_tile_width
            tile_height = basic_tile_height * 6
        else:
            user_data["segment_flag"] = False
    else:
        user_data["segment_flag"] = False
    segment_info = {
        "segment_out_info": {"width": tile_width, "height": tile_height},
        "start_position": {
            "width": index_width * user_data["config_params"]["tile_width"],
            "height": index_height * user_data["config_params"]["tile_height"],
        },
    }
    tile_info = {
        "chunk_idx": user_data["chunk_idx"],
        "tile_idx": user_data["relative_tile_idx"],
        "chunk_duration": chunk_info["chunk_duration"],
    }
    return tile_info, segment_info
