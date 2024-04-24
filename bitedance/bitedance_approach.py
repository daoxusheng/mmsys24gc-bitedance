import os

import cv2
import numpy as np
from e3po.approaches.bitedance.bitedance_utils import *
from e3po.utils import get_logger
from e3po.utils.decision_utilities import generate_dl_list
from e3po.utils.projection_utilities import (
    _3d_polar_coord_to_pixel_coord,
    fov_to_3d_polar_coord,
    pixel_coord_to_relative_tile_coord,
    pixel_coord_to_tile,
)


def video_analysis(user_data, video_info):
    """
    This API allows users to analyze the full 360 video (if necessary) before the pre-processing starts.
    Parameters
    ----------
    user_data: is initially set to an empy object and users can change it to any structure they need.
    video_info: is a dictionary containing the required video information.

    Returns
    -------
    user_data:
        user should return the modified (or unmodified) user_data as the return value.
        Failing to do so will result in the loss of the information stored in the user_data object.
    """

    user_data = user_data or {}
    user_data["video_analysis"] = []

    return user_data


def preprocess_video(
    source_video_uri, dst_video_folder, chunk_info, user_data, video_info
):
    """
    Self defined preprocessing strategy

    Parameters
    ----------
    source_video_uri: str
        the video uri (uniform resource identifier) of source video
    dst_video_folder: str
        the folder to store processed video
    chunk_info: dict
        chunk information
    user_data: dict
        store user-related parameters along with their required content
    video_info: dict
        store video information

    Returns
    -------
    user_video_spec: dict
        a dictionary storing user specific information for the preprocessed video
    user_data: dict
        updated user_data
    """
    ori_video_uri = video_info["uri"]

    if user_data is None or "video_info" not in user_data:
        user_data = init_user(user_data, video_info)

    config_params = user_data["config_params"]
    video_info = user_data["video_info"]

    # update related information
    if user_data["chunk_idx"] == -1:
        user_data["chunk_idx"] = chunk_info["chunk_idx"]
        user_data["tile_idx"] = 0
        user_data["transcode_video_uri"] = "***"
        user_data["relative_tile_idx"] = 0
    else:
        if user_data["chunk_idx"] != chunk_info["chunk_idx"]:
            user_data["chunk_idx"] = chunk_info["chunk_idx"]
            user_data["tile_idx"] = 0
            user_data["transcode_video_uri"] = "***"
            user_data["relative_tile_idx"] = 0

    # transcoding
    src_projection = video_info["projection"]  # original video projection
    dst_projection = config_params["converted_projection_mode"]
    if src_projection != dst_projection:  # 需要转码
        transcode_video_uri = (
            video_info["uri"].split(".")[0] + "_bitedance_transcode.mp4"
        )
        if (
            user_data["chunk_idx"] == 0 and user_data["tile_idx"] == 0
        ) or not os.path.exists(transcode_video_uri):
            src_resolution = [video_info["height"], video_info["width"]]
            dst_resolution = [
                config_params["converted_height"],
                config_params["converted_width"],
            ]
            transcode_ori_video(
                ori_video_uri,
                src_projection,
                dst_projection,
                src_resolution,
                dst_resolution,
                dst_video_folder,
                chunk_info,
                config_params["ffmpeg_settings"],
            )
    else:
        transcode_video_uri = ori_video_uri

    # segmentation
    user_data["segment_flag"] = False

    while not user_data["segment_flag"] and (
        user_data["tile_idx"] < config_params["total_tile_num"]
    ):
        tile_info, segment_info = tile_segment_info(chunk_info, user_data)
        user_data["tile_idx"] += 1
        print("====== PREPROC #1 ======")
        print(f"{user_data['tile_idx']}/{config_params['total_tile_num']}")
        print(chunk_info)
        print(tile_info)
        print(segment_info)
        print(user_data["segment_flag"])

    if user_data["segment_flag"]:
        segment_video(
            config_params["ffmpeg_settings"],
            transcode_video_uri,
            dst_video_folder,
            segment_info,
            tile_info,
        )
        user_data["relative_tile_idx"] += 1
        user_video_spec = {"segment_info": segment_info, "tile_info": tile_info}

    # resize, background stream
    elif (
        user_data["tile_idx"] == config_params["total_tile_num"]
        and config_params["background_flag"]
    ):
        bg_video_uri = transcode_video_uri
        resize_video(
            config_params["ffmpeg_settings"],
            bg_video_uri,
            dst_video_folder,
            config_params["background_info"],
            tile_info,
        )

        user_data["tile_idx"] += 1
        user_video_spec = {
            "segment_info": config_params["background_info"],
            "tile_info": {"chunk_idx": chunk_info["chunk_idx"], "tile_idx": -1},
        }
    else:
        user_video_spec = None

    user_data["transcode_video_uri"] = "***"
    return user_video_spec, user_data


def download_decision(
    network_stats, motion_history, video_size, curr_ts, user_data, video_info
):
    """
    Self defined download strategy

    Parameters
    ----------
    network_stats: list
        a list represents historical network status
    motion_history: list
        a list represents historical motion information
    video_size: dict
        video size of preprocessed video
    curr_ts: int
        current system timestamp
    user_data: dict
        user related parameters and information
    video_info: dict
        video information for decision module

    Returns
    -------
    dl_list: list
        the list of tiles which are decided to be downloaded
    user_data: dict
        updated user_date
    """

    if user_data is None or "video_info" not in user_data:
        user_data = init_user(user_data, video_info)

    config_params = user_data["config_params"]
    video_info = user_data["video_info"]

    if curr_ts == 0:  # initialize the related parameters
        user_data["next_download_idx"] = 0
        user_data["latest_decision"] = []
    dl_list = []
    chunk_idx = user_data["next_download_idx"]
    latest_decision = user_data["latest_decision"]

    if (
        user_data["next_download_idx"]
        >= video_info["duration"] / video_info["chunk_duration"]
    ):
        return dl_list, user_data

    if user_data["latest_decision"] == []:
        _pp_time = user_data["pd_time"]
        print(f'chunk_idx: {chunk_idx}, pd_time: {user_data["pd_time"]}')
    else:
        _pp_time = user_data["pp_time"]
    predicted_record = predict_motion_tile(
        motion_history,
        config_params["motion_history_size"],
        config_params["motion_prediction_size"],
        _pp_time,
    )  # motion prediction
    tile_record = tile_decision(
        predicted_record, video_size, video_info["range_fov"], chunk_idx, user_data
    )  # tile decision
    dl_list = generate_dl_list(chunk_idx, tile_record, latest_decision, dl_list)
    if user_data["latest_decision"] == []:
        dl_list = modify_dl_list(dl_list, video_size, network_stats, user_data)
    user_data = update_decision_info(
        user_data, tile_record, curr_ts
    )  # update decision information

    if user_data["download_stage"] > 0:
        user_data = update_pd_time(user_data, TIME_LIST[0])
        predicted_record = predict_motion_tile(
            motion_history,
            config_params["motion_history_size"],
            config_params["motion_prediction_size"],
            user_data["pd_time"],
        )
        tile_record = tile_decision(
            predicted_record, video_size, video_info["range_fov"], chunk_idx, user_data
        )
        nc_dl_list = generate_nc_dl_list(chunk_idx, tile_record, user_data)
        while (
            user_data["pd_time"] < TIME_LIST[-1]
            and get_total_delay(nc_dl_list, video_size, network_stats)
            > (user_data["max_delay"] - TOLERANCE) * 1000
        ):
            user_data = update_pd_time(user_data, round(user_data["pd_time"] + 0.01, 2))
            predicted_record = predict_motion_tile(
                motion_history,
                config_params["motion_history_size"],
                config_params["motion_prediction_size"],
                user_data["pd_time"],
            )
            tile_record = tile_decision(
                predicted_record,
                video_size,
                video_info["range_fov"],
                chunk_idx,
                user_data,
            )
            nc_dl_list = generate_nc_dl_list(chunk_idx, tile_record, user_data)

    return dl_list, user_data


def generate_display_result(
    curr_display_frames,
    current_display_chunks,
    curr_fov,
    dst_video_frame_uri,
    frame_idx,
    video_size,
    user_data,
    video_info,
):
    """
    Generate fov images corresponding to different approaches

    Parameters
    ----------
    curr_display_frames: list
        current available video tile frames
    current_display_chunks: list
        current available video chunks
    curr_fov: dict
        current fov information, with format {"curr_motion", "range_fov", "fov_resolution"}
    dst_video_frame_uri: str
        the uri of generated fov frame
    frame_idx: int
        frame index of current display frame
    video_size: dict
        video size of preprocessed video
    user_data: dict
        user related parameters and information
    video_info: dict
        video information for evaluation

    Returns
    -------
    user_data: dict
        updated user_data
    """

    get_logger().debug(f"[evaluation] start get display img {frame_idx}")

    if user_data is None or "video_info" not in user_data:
        user_data = init_user(user_data, video_info)

    video_info = user_data["video_info"]
    config_params = user_data["config_params"]

    chunk_idx = int(
        frame_idx
        * (1000 / video_info["video_fps"])
        // (video_info["chunk_duration"] * 1000)
    )  # frame idx starts from 0
    if chunk_idx <= len(current_display_chunks) - 1:
        tile_list = current_display_chunks[chunk_idx]["tile_list"]
    else:
        tile_list = current_display_chunks[-1]["tile_list"]

    avail_tile_list = []
    for i in range(len(tile_list)):
        tile_id = tile_list[i]["tile_id"]
        tile_idx = video_size[tile_id]["user_video_spec"]["tile_info"]["tile_idx"]
        avail_tile_list.append(tile_idx)

    # calculating fov_uv parameters
    fov_ypr = [
        float(curr_fov["curr_motion"]["yaw"]),
        float(curr_fov["curr_motion"]["pitch"]),
        0,
    ]
    _3d_polar_coord = fov_to_3d_polar_coord(
        fov_ypr, curr_fov["range_fov"], curr_fov["fov_resolution"]
    )
    pixel_coord = _3d_polar_coord_to_pixel_coord(
        _3d_polar_coord,
        config_params["projection_mode"],
        [config_params["converted_height"], config_params["converted_width"]],
    )

    coord_tile_list = pixel_coord_to_tile(
        pixel_coord, config_params["total_tile_num"], video_size, chunk_idx
    )
    relative_tile_coord = pixel_coord_to_relative_tile_coord(
        pixel_coord, coord_tile_list, video_size, chunk_idx
    )
    unavail_pixel_coord = ~np.isin(coord_tile_list, avail_tile_list)
    coord_tile_list[unavail_pixel_coord] = -1

    display_img = np.full(
        (coord_tile_list.shape[0], coord_tile_list.shape[1], 3),
        [128, 128, 128],
        dtype=np.float32,
    )  # create an empty matrix for the final image
    for i, tile_idx in enumerate(avail_tile_list):
        hit_coord_mask = coord_tile_list == tile_idx
        if not np.any(hit_coord_mask):  # if no pixels belong to the current frame, skip
            continue

        if tile_idx != -1:
            dstMap_u, dstMap_v = cv2.convertMaps(
                relative_tile_coord[0].astype(np.float32),
                relative_tile_coord[1].astype(np.float32),
                cv2.CV_16SC2,
            )
        else:
            out_pixel_coord = _3d_polar_coord_to_pixel_coord(
                _3d_polar_coord,
                config_params["background_info"]["background_projection_mode"],
                [config_params["background_height"], config_params["background_width"]],
            )
            dstMap_u, dstMap_v = cv2.convertMaps(
                out_pixel_coord[0].astype(np.float32),
                out_pixel_coord[1].astype(np.float32),
                cv2.CV_16SC2,
            )
        remapped_frame = cv2.remap(
            curr_display_frames[i], dstMap_u, dstMap_v, cv2.INTER_LINEAR
        )
        display_img[hit_coord_mask] = remapped_frame[hit_coord_mask]

    cv2.imwrite(dst_video_frame_uri, display_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

    get_logger().debug(f"[evaluation] end get display img {frame_idx}")

    return user_data
