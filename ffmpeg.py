import os
import re
import subprocess
from configparser import ConfigParser
from typing import Dict, Optional, Sequence, Tuple, Union

import torch

__all__ = [
    "get_file_info",
    "load",
    "load_audio",
    "load_video",
    "dump",
    "dump_audio",
    "dump_video",
]


_config = ConfigParser()
_config.read(os.path.join(os.path.dirname(__file__), "config.ini"))
FFMPEG_BIN = _config["ffmpeg"]["ffmpeg"]
FFPROBE_BIN = _config["ffmpeg"]["ffprobe"]


def _check_time_duration(time_duration: str) -> bool:
    r"""Check the time duration.

    Time duration must follow the specification given in the FFmpeg
    documentation, see https://www.ffmpeg.org/ffmpeg-utils.html#time-duration-syntax.

    Parameters
    ----------
    time_duration : str
        Time duration to be checked.

    Returns
    -------
    out : Optional[dict]
        Returns a dict if the time duration notation is correct containing the
        information of the sign, hours, minutes and seconds, None otherwise.
        The dict is structured as follows:
            {
                'sign': int,  # Optional[str],
                'hours': int,  # Optional[str],
                'minutes': int,  # Optional[str],
                'seconds': float,
            }
        #If the value of the key was not found during the match, the value
        #corresponding is None.

    Raises
    ------
    ValueError
        [description]
    """
    if not isinstance(time_duration, str):
        raise TypeError("expected string or bytes-like object")

    match = re.match(
        (
            r"^(?P<sign>[+-])?"
            r"(?:(?:(?P<hours>\d+):)?(?:(?P<minutes>[0-5]?\d):))?"
            r"(?P<seconds>[0-5]?\d(?:.\d+)?)$"
        ),
        time_duration,
    )
    out = match.groupdict() if match is not None else None

    # convert formatted string information to real numbers
    if out is not None:
        out["sign"] = -1 if out["sign"] is not None and out["sign"] == "-" else +1
        out["hours"] = int(out["hours"]) if out["hours"] is not None else 0
        out["minutes"] = int(out["minutes"]) if out["minutes"] is not None else 0
        out["seconds"] = float(out["seconds"]) if out["seconds"] is not None else 0.0

    return out


def _hhmmss2sec(time_duration: str) -> float:
    r"""Convert time duration from the form [-][[HH:]MM:]SS[.m] to [-]S[.m].

    Convert a string time duration in the form of a float time duration in seconds!!!!!

    Parameters
    ----------
    time_duration : str
        Time duration in the sexagesimal form [-][[HH:]MM:]SS[.m].
        HH expresses the number of hours, MM the number of minutes for a
        maximum of 2 digits, and SS the number of seconds for a maximum of 2
        digits. The m at the end expresses decimal value for SS.
        The optional ‘-’ indicates negative duration.

    Returns
    -------
    out : float
        Time duration in the form [-]S[.m] where S expresses the number of
        seconds, with the optional decimal part m. The optional ‘-’ indicates
        negative duration.

    Raises
    ------
    ValueError
        [description]
    """
    # check and extract time information
    info = _check_time_duration(time_duration)
    if info is None:
        raise ValueError(f"fzefez fzfz {time_duration}")

    # extract the notions of time from the dict
    sign = info["sign"]
    hours = info["hours"]
    minutes = info["minutes"]
    seconds = info["seconds"]

    # compute the number of seconds
    return sign * (hours * 3600 + minutes * 60 + seconds)


def get_file_info(
    fpath: str, stream: str = "audio+video"
) -> Dict[str, Union[float, Dict[str, float]]]:
    r"""Extract some information about the audiovisual file.

    Parameters
    ----------
    fpath : str
        Path to the input file.
    stream : str, optional (default="audio+video")
        The stream on which we want to retrieve the information. The value can
        be 'audio', 'video', 'audio+video'.

    Returns
    -------
    file_info : dict
        Extracted information from the audiovisual file structured as follows:
            {
                'audio': {
                    'duration': float,
                    'channels': int,
                    'sample_rate': int,
                },
                'video': {
                    'duration': float,
                    'width': int,
                    'height': int,
                    'frame_rate': float,
                },
            }
        If the stream parameter is different from 'audio+video', the
        sub-dictionary of the corresponding stream is returned.

    Raises
    ------
    ValueError
        [description]
    subprocess.CalledProcessError
        If the FFprobe command fail.
    """
    #
    if stream not in {"audio", "video", "audio+video"}:
        raise ValueError(
            f"unknow stream: {stream} (value can be 'audio', 'video' or 'audio+video')"
        )

    # create the ffprobe command
    command = (
        f"{FFPROBE_BIN} -loglevel fatal"
        " -print_format compact=print_section=0 -show_entries"
        " stream=codec_type,duration,sample_rate,r_frame_rate,channels,width,height"
        f" {fpath}"
    )

    # run the command and check if the execution did not generate an error
    ffprobe = subprocess.run(
        command.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    if ffprobe.returncode != 0:
        raise subprocess.CalledProcessError(
            f"FFprobe command exit with status {ffprobe.returncode}."
        )

    # initialisation of the dictionary containing the information for each
    # stream concerning
    file_info = {}
    file_info["audio"] = {} if "audio" in stream.split("+") else None
    file_info["video"] = {} if "video" in stream.split("+") else None

    # read each line containing information about each of the streams
    # containing in the file (information of the form
    # key1=value1|key2=value2|key3=value3|...)
    for out in ffprobe.stdout.splitlines():
        out = {
            key: value for key, value in [data.split("=") for data in out.split("|")]
        }
        codec_type = out.get("codec_type", None)
        if codec_type in stream.split("+"):
            if codec_type == "audio":
                file_info[codec_type]["duration"] = float(out["duration"])
                file_info[codec_type]["channels"] = int(out["channels"])
                file_info[codec_type]["sample_rate"] = int(out["sample_rate"])
            elif codec_type == "video":
                file_info[codec_type]["duration"] = float(out["duration"])
                file_info[codec_type]["height"] = int(out["height"])
                file_info[codec_type]["width"] = int(out["width"])
                r_frame_rate = tuple(map(float, out["r_frame_rate"].split("/")))
                file_info[codec_type]["frame_rate"] = r_frame_rate[0] / r_frame_rate[1]

    return file_info if stream == "audio+video" else file_info[stream]


def _get_frame_size(
    original_frame_size: Tuple[int, int],
    final_frame_size: Union[int, Tuple[int, int], None],
) -> Tuple[int, int]:
    r"""Return the new frame size while keeping ratio of the original frame.

    Parameters
    ----------
    original_frame_size: Tuple[int, int]
    final_frame_size: Union[int, Tuple[int, int], None]

    Raises
    ------
    TypeError
        [description]
    """
    if not (
        isinstance(original_frame_size, Sequence) and len(original_frame_size) == 2
    ):
        raise TypeError(
            f"got inappropriate frame_size arg - original_frame_size: {original_frame_size}"
        )
    if not (
        final_frame_size is None
        or isinstance(final_frame_size, int)
        or (isinstance(final_frame_size, Sequence) and len(final_frame_size) == 2)
    ):
        raise TypeError(
            f"got inappropriate frame_size arg - final_frame_size: {final_frame_size}"
        )

    width, height = original_frame_size

    if isinstance(final_frame_size, int):
        frame_size = (int(final_frame_size * width / height), final_frame_size)
    elif isinstance(final_frame_size, Sequence):
        if final_frame_size[0] == -1:
            frame_size = (
                int(final_frame_size[1] * width / height),
                final_frame_size[1],
            )
        elif final_frame_size[1] == -1:
            frame_size = (
                final_frame_size[0],
                int(final_frame_size[0] * height / width),
            )
        else:
            frame_size = final_frame_size
    else:
        frame_size = (width, height)

    return frame_size


def load_video(
    fpath: str,
    offset: Union[float, str] = 0.0,
    duration: Union[float, str, None] = None,
    frame_rate: Optional[float] = None,
    frame_size: Union[int, Tuple[int, int], None] = None,
    grayscale: bool = False,
    filters: Optional[str] = None,
    data_format: str = "channels_first",
    dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    r"""Return data and the frame rate from a video file.

    Return a torch.Tensor (C, H, W) in the range [0.0, 1.0] if the dtype is a
    floating point. In the other cases, tensors are returned without scaling.

    Parameters
    ----------
    fpath : str
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
        ffmpeg filter command

    data_format : str, optional (default="channels_first")
        The ordering of the dimensions in the outputs. If "channels_last",
        data_format corresponds to output with shape
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
    info = get_file_info(fpath, "video")

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
    command = (
        f"{FFMPEG_BIN} -loglevel fatal"
        f" {offset_cmd} {duration_cmd} -i {fpath}"
        f" -an -f image2pipe -codec:v rawvideo -pix_fmt {'gray' if grayscale else 'rgb24'}"
        f" {filter_cmd} pipe:1"
    )

    # run the command and check if the execution did not generate an error
    ffmpeg = subprocess.run(
        command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
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


def dump_video(
    video: torch.Tensor,
    fpath: str,
    frame_rate: int,
    frame_size: Union[int, Tuple[int, int], None] = None,
    filters: Optional[str] = None,
    overwrite: bool = True,
    codec: str = "libx264",
    data_format: str = "channels_first",
) -> None:
    """Write frames on the correct filepath

    Parameters
    ----------
    frames : torch.Tensor
        [description]
    fpath : str
        [description]

    Raises
    ------
    TypeError
        Inappropriate video format.
    subprocess.CalledProcessError
        If the FFmpeg command fail.
    """
    # check the type of the tensor
    if video.dtype in {torch.bool, torch.int8}:
        raise TypeError(f"got inappropriate video format - video.dtype: {video.dtype}")
    # convert the tensor into bytes
    if video.is_floating_point():
        video = video * 255
    video = video.to(torch.uint8)
    # check the values of the tensor
    if video.max() > 255 and video.min() < 0:
        raise ValueError()

    # convert video to channels_last if needed (required for FFmpeg)
    if data_format == "channels_first":
        video = video.permute(0, 2, 3, 1)
    # extract the dimensions of the video tensor
    seq_len, height, width, channels = video.shape

    # create the output filter command
    filter_opt = []
    # rescale the output if requested
    if frame_size is not None:
        expected_width, expected_height = (
            (frame_size, -1) if isinstance(frame_size, int) else frame_size
        )
        filter_opt.append(f"scale={expected_width}:{expected_height}")
    # add other user-defined FFmpeg filters
    if filters is not None:
        filter_opt.append(filters.split(","))
    # create the filter command
    filter_cmd = "-filter:v {}".format(",".join(filter_opt)) if filter_opt else ""

    # create the ffmpeg command
    command = (
        f"{FFMPEG_BIN} -loglevel fatal {'-y' if overwrite else ''}"  # overwrite output file if it exists
        f" -f rawvideo -codec:v rawvideo -pix_fmt {'rgb24' if channels == 3 else 'gray'}"
        f" -s {width}x{height} -r {frame_rate} -i pipe:0"
        f" -an -codec:v {codec} -pix_fmt yuv420p {filter_cmd} {fpath}"
    )

    # run the command and check if the execution did not generate an error
    ffmpeg = subprocess.run(
        command.split(),
        input=video.data.numpy().tobytes(),
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


def load_audio(
    fpath: str,
    offset: Union[float, str] = 0.0,
    duration: Union[float, str, None] = None,
    sample_rate: Optional[float] = None,
    mono: bool = True,
    filters: Optional[str] = None,
    data_format: str = "channels_first",
    dtype: torch.dtype = torch.float,
):
    r"""Return data and the sample rate from an audio file.

    Parameters
    ----------
    fpath : str
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
    data_format : str, optional (default="channels_first")
        The ordering of the dimensions in the outputs. If "channels_last",
        data_format corresponds to inputs with shape (batch, steps, channels)
        while "channels_first" corresponds to inputs with shape
        (batch, channels, steps).
    dtype : torch.dtype, optional (default=torch.float)
        Desired output data-type for the tensor, e.g, torch.int16.

    Returns
    -------
    audio: torch.Tensor
        Data read from audio file.
    sample_rate: int
        Sample rate of audio file.

    Raises
    ------
    ValueError
        [description]
    subprocess.CalledProcessError
        [description]
    """
    # retrieve information about the video (duration, sample rate,
    # number of channels)
    info = get_file_info(fpath, "audio")

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
        raise ValueError("Unknow data_format:", data_format)

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
    command = (
        f"{FFMPEG_BIN} -loglevel fatal"
        f" {offset_cmd} {duration_cmd} -i {fpath}"
        f" -vn -f s16le -codec:a pcm_s16le {mono_cmd} {filter_cmd} pipe:1"
    )

    # run the command and check if the execution did not generate an error
    ffmpeg = subprocess.run(
        command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
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


def dump_audio(
    audio: torch.Tensor,
    fpath: str,
    sample_rate: int,
    filters: Optional[str] = None,
    overwrite: bool = True,
    codec: str = "pcm_s16le",
    data_format: str = "channels_first",
) -> None:
    # check the type of the tensor
    if audio.dtype in {torch.bool, torch.uint8, torch.int8}:
        raise TypeError(f"got inappropriate audio format - audio.dtype: {audio.dtype}")
    # convert the tensor into bytes
    if audio.is_floating_point():
        audio = (audio + 1) * ((2 ** 16 - 1) / 2) - 32768

    audio = audio.to(torch.int16)
    # check the values of the tensor
    if audio.max() > 32767 and audio.min() < -32768:
        raise ValueError()

    # convert video to channels_last if needed (required for FFmpeg)
    if data_format == "channels_first":
        audio.transpose_(0, 1)

    # create the filter command
    filter_cmd = f"-filter:a {filters}" if filters is not None else ""

    # create the ffmpeg command
    command = (
        f"{FFMPEG_BIN} -loglevel fatal {'-y' if overwrite else ''}"  # overwrite output file if it exists
        f" -f s16le -codec:a pcm_s16le -r {sample_rate} -i pipe:0"
        f" -vn -codec:a {codec} {filter_cmd} {fpath}"
    )

    # run the command and check if the execution did not generate an error
    ffmpeg = subprocess.run(
        command.split(),
        input=audio.data.numpy().tobytes(),
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


def load(
    fpath: str,
    offset: Union[float, str] = 0.0,
    duration: Union[float, str, None] = None,
    frame_rate: Optional[float] = None,
    frame_size: Optional[str] = None,
    grayscale: bool = False,
    sample_rate: Optional[float] = None,
    mono: bool = True,
    data_format: str = "channels_first",
    dtype: torch.dtype = torch.float,
) -> Tuple[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int]]:
    r"""Return audiovisual data, frame rate and sample rate.

    Parameters
    ----------
    fpath : str
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
    video, frame_rate = load_video(
        fpath,
        offset=offset,
        duration=duration,
        frame_rate=frame_rate,
        frame_size=frame_size,
        grayscale=grayscale,
        data_format=data_format,
        dtype=dtype,
    )
    audio, sample_rate = load_audio(
        fpath,
        offset=offset,
        duration=duration,
        sample_rate=sample_rate,
        mono=mono,
        data_format=data_format,
        dtype=dtype,
    )
    return (video, frame_rate), (audio, sample_rate)


def dump(
    audio_data: Optional[torch.Tensor], video_data: Optional[torch.Tensor], fpath: str
) -> None:
    if video_data is not None:
        dump_video()
    if audio_data is not None:
        pass
