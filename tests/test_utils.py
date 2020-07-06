import os
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import pytest
import torch

import ffmpeg.utils


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
    "time_duration, expected",
    [
        pytest.param("32.2", (+1, 0, 0, 32.2), id="32.2"),
        pytest.param("-180.45", None, id="-180.45"),
        pytest.param("-9:56", (-1, 0, 9, 56), id="-9:56"),
        pytest.param("-79:30.31", None, id="-79:30.31"),
        pytest.param("+00:07:56.2", (+1, 0, 7, 56.2), id="+00:07:56.2"),
        pytest.param("123:29:56.3", (+1, 123, 29, 56.3), id="123:29:56.3"),
    ],
)
def test_check_time_duration(
    time_duration, expected,
):
    # convert the tuple into the same format as the output of the
    # _check_time_duration function (dict)
    if expected is not None:
        expected = {
            key: expected[index]
            for index, key in enumerate(("sign", "hours", "minutes", "seconds"))
        }
    assert ffmpeg.utils._check_time_duration(time_duration) == expected


@pytest.mark.parametrize(
    "time_duration, expected", [pytest.param(24.2, TypeError, id="TypeError")]
)
def test_check_time_duration_exception(time_duration, expected):
    with pytest.raises(expected):
        ffmpeg.utils._check_time_duration(time_duration)


@pytest.mark.parametrize(
    "hhmmss, expected",
    [
        pytest.param("32.2", 32.2, id="32.2"),
        pytest.param("-9:56", -596.0, id="-9.56"),
        pytest.param("+00:07:56.2", 476.2, id="+00:07:56.2"),
        pytest.param("123:29:56.3", 444596.3, id="123:29:56.3"),
    ],
)
def test_hhmmss2sec(hhmmss, expected):
    assert ffmpeg.utils._hhmmss2sec(hhmmss) == expected


@pytest.mark.parametrize(
    "hhmmss, expected", [pytest.param("-180.45", ValueError, id="ValueError")]
)
def test_hhmmss2sec_exception( hhmmss, expected):
    with pytest.raises(expected):
        ffmpeg.utils._hhmmss2sec("-180.45")


@pytest.mark.parametrize(
    "url, audio_info, video_info",
    [(data["url"], data["audio"], data["video"]) for data in dataset],
    ids=[data["url"].rpartition("/")[2] for data in dataset],
)
def test_get_file_info(url, audio_info, video_info):
    assert ffmpeg.utils.get_file_info(url, stream="audio") == audio_info
    assert ffmpeg.utils.get_file_info(url, stream="video") == video_info
    assert ffmpeg.utils.get_file_info(url, stream="audio+video") == {
        "audio": audio_info,
        "video": video_info,
    }


@pytest.mark.parametrize(
    "url, stream, expected",
    [
        pytest.param(data["url"], "av", ValueError, id="StreamValueError")
        for data in dataset
    ],
)
def test_get_file_info_exception(url, stream, expected):
    with pytest.raises(expected):
        assert ffmpeg.utils.get_file_info(url, stream=stream)


@pytest.mark.parametrize(
    "original, final, expected",
    [((480, 360), None, (480, 360)), ((1920, 1080), 720, (1280, 720))],
)
def test_get_frame_size(original, final, expected):
    assert ffmpeg.utils._get_frame_size(original, final) == expected
