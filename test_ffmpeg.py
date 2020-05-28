import os
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import pytest
import torch

import ffmpeg


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


class TestVideo(object):
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

    def test_load_video_exception(self):
        # if dtype is {torch.bool, torch.int8}:
        #    exception = TypeError()
        # with pytest.raises(exception):
        pass

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
                torch.tensor(
                    [[[[0]]], [[[64]]], [[[128]]], [[[192]]], [[[255]]]]
                ).expand(5, 1, 100, 200),
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
        ffmpeg.dump_video(
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
        video_info = ffmpeg.get_file_info(fpath, stream="video")

        assert frame_rate == video_info["frame_rate"]
        assert seq_len == (video_info["duration"] // video_info["frame_rate"])
        assert width == video_info["width"]
        assert height == video_info["height"]

    def test_dump_video_exception(self):
        pass


class TestAudio(object):
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

    def test_load_audio_exception(self):
        pass

    def test_dump_audio(self):
        pass

    def test_dump_audio_exception(self):
        pass


class TestAudioVideo(object):
    def test_load(self):
        pass

    def test_dump(self):
        pass


class TestUtils(object):
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
        self, time_duration, expected,
    ):
        # convert the tuple into the same format as the output of the
        # _check_time_duration function (dict)
        if expected is not None:
            expected = {
                key: expected[index]
                for index, key in enumerate(("sign", "hours", "minutes", "seconds"))
            }
        assert ffmpeg._check_time_duration(time_duration) == expected

    @pytest.mark.parametrize(
        "time_duration, expected", [pytest.param(24.2, TypeError, id="TypeError")]
    )
    def test_check_time_duration_exception(self, time_duration, expected):
        with pytest.raises(expected):
            ffmpeg._check_time_duration(time_duration)

    @pytest.mark.parametrize(
        "hhmmss, expected",
        [
            pytest.param("32.2", 32.2, id="32.2"),
            pytest.param("-9:56", -596.0, id="-9.56"),
            pytest.param("+00:07:56.2", 476.2, id="+00:07:56.2"),
            pytest.param("123:29:56.3", 444596.3, id="123:29:56.3"),
        ],
    )
    def test_hhmmss2sec(self, hhmmss, expected):
        assert ffmpeg._hhmmss2sec(hhmmss) == expected

    @pytest.mark.parametrize(
        "hhmmss, expected", [pytest.param("-180.45", ValueError, id="ValueError")]
    )
    def test_hhmmss2sec_exception(self, hhmmss, expected):
        with pytest.raises(expected):
            ffmpeg._hhmmss2sec("-180.45")

    @pytest.mark.parametrize(
        "url, audio_info, video_info",
        [(data["url"], data["audio"], data["video"]) for data in dataset],
        ids=[data["url"].rpartition("/")[2] for data in dataset],
    )
    def test_get_file_info(self, url, audio_info, video_info):
        assert ffmpeg.get_file_info(url, stream="audio") == audio_info
        assert ffmpeg.get_file_info(url, stream="video") == video_info
        assert ffmpeg.get_file_info(url, stream="audio+video") == {
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
    def test_get_file_info_exception(self, url, stream, expected):
        with pytest.raises(expected):
            assert ffmpeg.get_file_info(url, stream=stream)

    @pytest.mark.parametrize(
        "original, final, expected",
        [((480, 360), None, (480, 360)), ((1920, 1080), 720, (1280, 720))],
    )
    def test_get_frame_size(self, original, final, expected):
        assert ffmpeg._get_frame_size(original, final) == expected
