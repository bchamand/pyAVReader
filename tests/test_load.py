import os
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import pytest
import torch

import ffmpeg.load


dataset = [
    {
        "url": "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
        "audio": {"duration": 596.474195, "channels": 2, "sample_rate": 44100},
        "video": {
            "duration": 596.458333,
            "width": 1280,
            "height": 720,
            "frame_rate": 24,
        },
    }
]


@pytest.mark.parametrize(
    (
        "fpath, offset, duration, frame_rate, frame_size"
        ", grayscale, filters, data_format, dtype, video_info"
    ),
    [
        pytest.param(
            data["url"],
            *opt,
            data["video"],
            id="{}[{}]".format(data["url"].rpartition("/")[2], index)
        )
        for data in dataset
        for index, opt in enumerate(
            [
                (0.0, 1.0, None, None, False, None, "channels_first", torch.float),
                (
                    "1.0",
                    "1.0",
                    None,
                    None,
                    False,
                    None,
                    "channels_first",
                    torch.float,
                ),
                (-1.0, 1.0, None, None, False, None, "channels_first", torch.float),
                (0.0, 1.0, 12, None, False, None, "channels_first", torch.float),
                (0.0, 1.0, None, 360, False, None, "channels_first", torch.float),
                (
                    0.0,
                    1.0,
                    None,
                    (352, 240),
                    False,
                    None,
                    "channels_first",
                    torch.float,
                ),
                (
                    0.0,
                    1.0,
                    None,
                    (-1, 240),
                    False,
                    None,
                    "channels_first",
                    torch.float,
                ),
                (
                    0.0,
                    1.0,
                    None,
                    (352, -1),
                    False,
                    None,
                    "channels_first",
                    torch.float,
                ),
                (0.0, 1.0, None, None, True, None, "channels_first", torch.float),
                (0.0, 1.0, None, None, False, None, "channels_last", torch.float),
                (0.0, 1.0, None, None, False, None, "channels_first", torch.uint8),
                (0.0, 1.0, None, None, False, None, "channels_first", torch.int),
            ]
        )
    ],
)
def test_load_video(
    self,
    fpath,
    offset,
    duration,
    frame_rate,
    frame_size,
    grayscale,
    filters,
    data_format,
    dtype,
    video_info,
):
    # load video
    frames, final_frame_rate = ffmpeg.load_video(
        fpath,
        offset=offset,
        duration=duration,
        frame_rate=frame_rate,
        frame_size=frame_size,
        grayscale=grayscale,
        filters=filters,
        data_format=data_format,
        dtype=dtype,
    )

    # compute expected values
    if frame_size is None:
        frame_size = (video_info["width"], video_info["height"])
    elif isinstance(frame_size, (int, float)):
        ratio = video_info["width"] / video_info["height"]
        frame_size = (int(frame_size * ratio), frame_size)
    elif isinstance(frame_size, Sequence):
        if frame_size[0] == -1:
            ratio = video_info["width"] / video_info["height"]
            frame_size = (int(frame_size[1] * ratio), frame_size[1])
        elif frame_size[1] == -1:
            ratio = video_info["height"] / video_info["width"]
            frame_size = (frame_size[0], int(frame_size[0] * ratio))
    channels = 1 if grayscale else 3
    expected_duration = (
        float(duration) if duration is not None else video_info["duration"]
    )
    expected_frame_rate = (
        frame_rate if frame_rate is not None else video_info["frame_rate"]
    )
    seq_len = expected_duration * expected_frame_rate

    if data_format == "channels_first":
        expected_shape = (seq_len, channels, *frame_size[::-1])
    else:
        expected_shape = (seq_len, *frame_size[::-1], channels)

    # test values
    assert final_frame_rate == expected_frame_rate
    assert frames.shape == expected_shape
    assert frames.dtype == dtype
    assert frames.min() >= 0
    assert frames.max() <= 1.0 if dtype.is_floating_point else 255





@pytest.mark.parametrize(
    (
        "fpath, offset, duration, sample_rate"
        ", mono, filters, data_format, dtype, audio_info"
    ),
    [
        pytest.param(
            data["url"],
            *opt,
            data["audio"],
            id="{}[{}]".format(data["url"].rpartition("/")[2], index)
        )
        for data in dataset
        for index, opt in enumerate(
            [
                (0.0, 1.0, None, True, None, "channels_first", torch.float),
                ("1.0", "1.0", None, True, None, "channels_first", torch.float,),
                (-1.0, 1.0, None, True, None, "channels_first", torch.float),
                (0.0, 1.0, 16000, True, None, "channels_first", torch.float),
                (0.0, 1.0, None, False, None, "channels_first", torch.float),
                (0.0, 1.0, None, True, None, "channels_last", torch.float),
                (0.0, 1.0, None, True, None, "channels_first", torch.int),
            ]
        )
    ],
)
def test_load_audio(
    self,
    fpath,
    offset,
    duration,
    sample_rate,
    mono,
    filters,
    data_format,
    dtype,
    audio_info,
):
    # load video
    audio, final_sample_rate = ffmpeg.load_audio(
        fpath,
        offset=offset,
        duration=duration,
        sample_rate=sample_rate,
        mono=mono,
        filters=filters,
        data_format=data_format,
        dtype=dtype,
    )

    # compute expected values
    channels = 1 if mono else audio_info["channels"]
    expected_duration = (
        float(duration) if duration is not None else audio_info["duration"]
    )
    expected_sample_rate = (
        sample_rate if sample_rate is not None else audio_info["sample_rate"]
    )
    seq_len = expected_duration * expected_sample_rate

    if data_format == "channels_first":
        expected_shape = (channels, seq_len)
    else:
        expected_shape = (seq_len, channels)

    # test values
    assert final_sample_rate == expected_sample_rate
    assert audio.shape == expected_shape
    assert audio.dtype == dtype
    assert audio.float().mean() != 0
    assert audio.min() >= -1 if dtype.is_floating_point else -32768  # short min
    assert audio.max() <= 1 if dtype.is_floating_point else 32767  # short max
