import os
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import pytest
import torch

import avreader


@pytest.mark.parametrize(
    "video, frame_rate, frame_size, filters, data_format",
    [
        pytest.param(
            torch.tensor(
                [
                    [[[255]], [[0]], [[0]]],
                    [[[0]], [[255]], [[0]]],
                    [[[0]], [[0]], [[255]]],
                ]
            ).expand(3, 3, 100, 200),
            1,
            None,
            None,
            "channels_first",
            id="video",
        ),
        pytest.param(
            torch.tensor([[[[0]]], [[[64]]], [[[128]]], [[[192]]], [[[255]]]]).expand(
                5, 1, 100, 200
            ),
            1,
            None,
            None,
            "channels_first",
            id="video",
        ),
        pytest.param(
            torch.tensor(
                [
                    [[[0]], [[0]], [[255]]],
                    [[[0]], [[255]], [[0]]],
                    [[[255]], [[0]], [[0]]],
                ]
            ).expand(3, 3, 100, 200),
            2,
            (240, 352),
            None,
            "channels_first",
            id="video",
        ),
    ],
    #
)
def test_dump_video(
    self, tmp_path, video, frame_rate, frame_size, filters, data_format,
):
    fpath = tmp_path / "video.mp4"

    # save video
    avreader.dump_video(
        video,
        fpath,
        frame_rate,
        frame_size=frame_size,
        filters=filters,
        data_format=data_format,
    )

    if data_format == "channels_first":
        seq_len, channels, height, width = video.shape
    else:
        seq_len, height, width, channels = video.shape
    video_info = avreader.get_file_info(fpath, stream="video")

    assert frame_rate == video_info["frame_rate"]
    assert seq_len == (video_info["duration"] // video_info["frame_rate"])
    assert width == video_info["width"]
    assert height == video_info["height"]
