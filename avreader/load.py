import subprocess
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor

import avreader.path
from avreader.utils import _get_frame_size, _hhmmss2sec, get_file_info


def load_audio(
    file: Union[bytes, str],
    offset: Union[float, str] = 0.0,
    duration: Union[float, str, None] = None,
    sample_rate: Optional[int] = None,
    mono: bool = True,
    filters: Optional[str] = None,
    data_format: str = "channels_first",
    dtype: torch.dtype = torch.float,
) -> Tuple[Tensor, int]:
    r"""Return data and the sample rate from an audio file.

    Parameters
    ----------
    file : Union[bytes, str]
        Path to the input file.
    offset : Union[float, str], optional (default=0.0)
        Start reading after this time. Offset must be a time duration
        specification, see
        https://www.ffmpeg.org/ffmpeg-utils.html#time-duration-syntax.
    duration : Union[float, str, None], optional (default=None)
        Only load up to this much audio. Duration must be a time duration
        specification, see
        https://www.ffmpeg.org/ffmpeg-utils.html#time-duration-syntax.
    sample_rate : Optional[float], optional (default=None)
        Target sampling rate. If None, sample_rate is the native sampling rate.
    mono : bool, optional (default=True)
        Converting signal to mono.
    filters : Optional[str], optional (default=None)
        Add a FFmpeg filtergraph, see https://ffmpeg.org/ffmpeg-filters.html.
    data_format : str, optional (default="channels_first")
        The ordering of the dimensions of the output `audio`.
        If "channels_last", data_format corresponds to output tensor with shape
        (seq_len, channels) while "channels_first" corresponds to output tensor
        with shape (channels, seq_len).
    dtype : torch.dtype, optional (default=torch.float)
        Desired output data-type for the tensor, e.g, torch.int16.

    Returns
    -------
    audio: torch.Tensor
        Data read from audio file.
    sample_rate: int
        Sample rate (in samples/sec) of audio file.

    Raises
    ------
    ValueError
        [description]
    subprocess.CalledProcessError
        [description]
    """
    # retrieve information about the video (duration, sample rate,
    # number of channels)
    info = get_file_info(file, "audio")

    # check the parameters
    offset = _hhmmss2sec(offset) if isinstance(offset, str) else offset
    if offset > info["duration"]:
        raise ValueError(
            "The offset value is greater than the duration of the video:"
            f" {offset} > {info['duration']:.4}"
        )

    duration = (
        _hhmmss2sec(duration)
        if isinstance(duration, str)
        else duration or info["duration"]
    )

    # check the data format
    if not mono and data_format not in {"channels_last", "channels_first"}:
        raise ValueError(f"Unknow data_format: {data_format}")

    if dtype in {torch.bool, torch.uint8, torch.int8}:
        raise TypeError(f"Got inappropriate dtype arg: {dtype}")

    # pre-process some options of the FFmpeg command
    offset_cmd = (
        f"-ss {offset}" if offset >= 0.0 else f"-sseof {offset}"
    )  # seek the input to position
    duration_cmd = (
        f"-t {duration}" if duration else ""
    )  # limit the duration of data read
    mono_cmd = "-ac 1" if mono else ""  # convert to mono

    # create the output filter command
    filter_opt = []
    # resample audio the output if requested
    if sample_rate is not None:
        filter_opt.append(f"aresample={sample_rate}")
    # add other user-defined FFmpeg filters
    if filters is not None:
        filter_opt.append(filters.split(","))
    # create the filter command
    filter_cmd = "-filter:a {}".format(",".join(filter_opt)) if filter_opt else ""

    # create the ffmpeg command
    input_url = file if isinstance(file, str) else "pipe:0"
    command = (
        f"{avreader.path.FFMPEG_BIN} -loglevel fatal"
        f" {offset_cmd} {duration_cmd} -i {input_url}"
        f" -vn -f s16le -codec:a pcm_s16le {mono_cmd} {filter_cmd} pipe:1"
    )

    # run the command and check if the execution did not generate an error
    ffmpeg = subprocess.run(
        command.split(),
        input=None if isinstance(file, str) else file,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if ffmpeg.returncode != 0:
        raise subprocess.CalledProcessError(
            ffmpeg.returncode,
            " ".join(command.split()),
            output=ffmpeg.stdout,
            stderr=ffmpeg.stderr,
        )

    # convert the buffer to tensor
    audio = torch.ShortTensor(torch.ShortStorage.from_buffer(ffmpeg.stdout, "native"))
    # reshape in (seq_len, channels)
    sample_rate = sample_rate or info["sample_rate"]
    duration = duration or info["duration"]
    channels = 1 if mono else info["channels"]
    audio.resize_(int(duration * sample_rate), channels)

    # permute the last dimension with the first one if 'channels_first'
    if data_format == "channels_first":
        audio.transpose_(0, 1)

    # change the type of the tensor
    audio = audio.to(dtype)
    if dtype.is_floating_point:
        # rescale between -1 and 1
        audio.add_(32768).div_((2 ** 16 - 1) / 2).add_(-1)

    return audio, sample_rate


def load_video(
    file: Union[bytes, str],
    offset: Union[float, str] = 0.0,
    duration: Union[float, str, None] = None,
    frame_rate: Optional[int] = None,
    frame_size: Union[int, Tuple[int, int], None] = None,
    grayscale: bool = False,
    filters: Optional[str] = None,
    data_format: str = "channels_first",
    dtype: torch.dtype = torch.float,
) -> Tuple[Tensor, int]:
    r"""Return data and the frame rate from a video file.

    Return a torch.Tensor (C, H, W) in the range [0.0, 1.0] if the dtype is a
    floating point. In the other cases, tensors are returned without scaling.

    Parameters
    ----------
    file : Union[bytes, str]
        Path to the input file.
    offset : Union[float, str], optional (default=0.0)
        Start reading after this tile. Offset must be a time duration
        specification, see https://www.ffmpeg.org/ffmpeg-utils.html#time-duration-syntax.
    duration : Union[float, str, None], optional (default=None)
        Only load up to this much audio. Duration must be a time duration
        specification, see https://www.ffmpeg.org/ffmpeg-utils.html#time-duration-syntax.
    frame_rate : Optional[float], optional (default=None)
        Target frame rate. If None, frame_rate is the native frame rate.
    frame_size : Union[int, Tuple[int, int], None], optional (default=None)
        Target frame size (width, height). If None, frame_size is the native
        frame size. The value can be an `int` giving the height of the frame,
        the height will be automatically calculated by respecting the aspect
        ratio. With the same effect, it is possible to define only one
        component, either height or width, and set the other component to -1.
    grayscale : bool, optional (default=False)
        Converting video to grayscale.
    filters : str, optional (default=None)
        Add a FFmpeg filtergraph, see https://ffmpeg.org/ffmpeg-filters.html.
    data_format : str, optional (default="channels_first")
        The ordering of the dimensions of the output tensor `video`.
        If "channels_last", data_format corresponds to output with shape
        (seq_len, height, width, channels) while "channels_first" corresponds
        to inputs with shape (seq_len, channels, height, width).
    dtype : torch.dtype, optional (default=torch.float)
        Desired output data-type for the tensor, e.g, torch.int16.
        Can be all torch types except torch.bool and torch.int8.

    Returns
    -------
    video : torch.Tensor
        Tensor of the form (seq_len, channels, height, width) with seq_len
        representing the selected number of frames of the video.
    frame_rate : int
        The frame rate corresponding to the video.

    Raises
    ------
    TypeError
        [description]
    ValueError
        [description]
    subprocess.CalledProcessError
        If the FFmpeg command fail.
    """
    # retrieve information about the video (duration, frame rate, frame size)
    info = get_file_info(file, "video")

    # check the parameters
    offset = _hhmmss2sec(offset) if isinstance(offset, str) else offset
    if offset > info["duration"]:
        raise ValueError(
            "The offset value is greater than the duration of the video:"
            f" {offset} > {info['duration']:.4}"
        )

    duration = (
        _hhmmss2sec(duration)
        if isinstance(duration, str)
        else duration or info["duration"]
    )

    if data_format not in {"channels_last", "channels_first"}:
        raise ValueError(f"Unknow data_format: {data_format}")

    if dtype in {torch.bool, torch.int8}:
        raise TypeError(f"Got inappropriate dtype arg: {dtype}")

    # pre-process some options of the FFmpeg command
    offset_cmd = (
        f"-ss {offset}" if offset >= 0.0 else f"-sseof {offset}"
    )  # seek the input to position
    duration_cmd = (
        f"-t {duration}" if duration else ""
    )  # limit the duration of data read

    # create the output filter command
    filter_opt = []
    # change the frame rate of the output if requested
    if frame_rate is not None:
        filter_opt.append(f"fps={frame_rate}")
    # rescale the output if requested
    if frame_size is not None:
        width, height = (frame_size, -1) if isinstance(frame_size, int) else frame_size
        filter_opt.append(f"scale={width}:{height}")
    # add other user-defined FFmpeg filters
    if filters is not None:
        filter_opt.append(filters.split(","))
    # create the filter command
    filter_cmd = "-filter:v {}".format(",".join(filter_opt)) if filter_opt else ""

    # create the ffmpeg command
    input_url = file if isinstance(file, str) else "pipe:0"
    command = (
        f"{avreader.path.FFMPEG_BIN} -loglevel fatal"
        f" {offset_cmd} {duration_cmd} -i {input_url}"
        f" -an -f image2pipe -codec:v rawvideo -pix_fmt {'gray' if grayscale else 'rgb24'}"
        f" {filter_cmd} pipe:1"
    )

    # run the command and check if the execution did not generate an error
    ffmpeg = subprocess.run(
        command.split(),
        input=None if isinstance(file, str) else file,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if ffmpeg.returncode != 0:
        raise subprocess.CalledProcessError(
            ffmpeg.returncode,
            " ".join(command.split()),
            output=ffmpeg.stdout,
            stderr=ffmpeg.stderr,
        )

    # convert the buffer to tensor
    video = torch.ByteTensor(torch.ByteStorage.from_buffer(ffmpeg.stdout))
    # reshape in (seq_len, height, width, channels)
    channels = 1 if grayscale else 3
    frame_rate = frame_rate or info["frame_rate"]
    frame_size = _get_frame_size((info["width"], info["height"]), frame_size)
    video.resize_(int(duration * frame_rate), frame_size[1], frame_size[0], channels)

    # permute the last dimension with the first one if 'channels_first'
    if data_format == "channels_first":
        video = video.permute(0, 3, 1, 2)

    # change the type of the tensor
    video = video.to(dtype)
    if dtype.is_floating_point:
        video /= 255  # rescale between 0 and 1

    return video, frame_rate


def load(
    path: Union[bytes, str],
    offset: Union[float, str] = 0.0,
    duration: Union[float, str, None] = None,
    akwargs: Optional[dict] = None,
    vkwargs: Optional[dict] = None,
) -> Tuple[Tuple[Tensor, int], Tuple[Tensor, int]]:
    r"""Return audiovisual data, frame rate and sample rate.

    Parameters
    ----------
    file : Union[bytes, str]
        Path to the input file.
    offset : Union[float, str], optional (default=0.0)
        Start reading after this time. Offset must be a time duration
        specification, see
        https://www.ffmpeg.org/ffmpeg-utils.html#time-duration-syntax.
    duration : Union[float, str, None], optional (default=None)
        Only load up to this much audio. Duration must be a time duration
        specification, see
        https://www.ffmpeg.org/ffmpeg-utils.html#time-duration-syntax.
    frame_rate : Optional[float], optional (default=None)
        [description]
    frame_size : Optional[str], optional (default=None)
        [description]
    grayscale : bool, optional (default=False)
        Converting video to grayscale.
    sample_rate : Optional[float], optional (default=None)
        Target sampling rate. If None, sample_rate is the native sampling rate.
    mono : bool, optional (default=True)
        Converting signal to mono.
    data_format : str, optional (default="channels_first")
        The ordering of the dimensions in the outputs. If "channels_last",
        data_format corresponds to inputs with shape (batch, steps, channels)
        while "channels_first" corresponds to inputs with shape
        (batch, channels, steps).
    dtype : torch.dtype, optional (default=torch.float)
        Desired output data-type for the tensor, e.g, torch.int16.

    Returns
    -------
    video : Tuple[torch.Tensor, int]
        [description]
    audio : Tuple[torch.Tensor, int]
        [description]
    """

    audio, sample_rate = load_audio(path, offset=offset, duration=duration, **akwargs)
    video, frame_rate = load_video(path, offset=offset, duration=duration, **vkwargs)

    return ((audio, sample_rate), (video, frame_rate))
