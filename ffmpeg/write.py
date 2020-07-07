import subprocess
from typing import Optional, Tuple, Union

import torch

import ffmpeg.path


def write_audio(
    fpath: str,
    audio: torch.Tensor,
    sample_rate: int,
    mono: bool = True,
    filters: Optional[str] = None,
    overwrite: bool = True,
    codec: str = "pcm_s16le",
    data_format: str = "channels_first",
) -> None:
    """Write a torch tensor as a WAV file.

    Parameters
    ----------
    fpath : str
        Path to the output file.
    audio : torch.Tensor
        A torch tensor containing the audio data.
    sample_rate : int
        The audio input sample rate (in samples/sec).
    filters : Optional[str], optional (default=None)
        Add a FFmpeg filtergraph, see https://ffmpeg.org/ffmpeg-filters.html.
    overwrite : bool, optional (default=True)
        Overwrite output file if it exists.
    codec : str, optional (default="pcm_s16le")
        Audio codec to be used to encode the data, see the FFmpeg documentation
        (https://ffmpeg.org/ffmpeg-codecs.html) for the list of compatible
        codecs.
    data_format : str, optional (default="channels_first")
        The ordering of the dimensions of the input `audio`.
        If "channels_last", data_format corresponds to input tensor with shape
        (seq_len, channels) while "channels_first" corresponds to input tensor
        with shape (channels, seq_len).

    Raises
    ------
    TypeError
        [description]
    ValueError
        [description]
    subprocess.CalledProcessError
        [description]
    """
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

    # convert audio to channels_last if needed (required for FFmpeg)
    if data_format == "channels_first":
        audio.transpose_(0, 1)

    # pre-process some options of the FFmpeg command
    mono_cmd = "-ac 1" if mono else ""  # convert to mono

    # create the filter command
    filter_cmd = f"-filter:a {filters}" if filters is not None else ""

    # create the ffmpeg command
    command = (
        f"{ffmpeg.path.FFMPEG_BIN} -loglevel fatal {'-y' if overwrite else ''}"
        f" -f s16le -codec:a pcm_s16le -r {sample_rate} -i pipe:0"
        f" -vn -codec:a {codec} {mono_cmd} {filter_cmd} {fpath}"
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


def write_video(
    fpath: str,
    video: torch.Tensor,
    frame_rate: int,
    frame_size: Union[int, Tuple[int, int], None] = None,
    filters: Optional[str] = None,
    overwrite: bool = True,
    codec: str = "libx264",
    data_format: str = "channels_first",
) -> None:
    """Write a torch tensor as a MP4 file.

    Parameters
    ----------
    fpath : str
        Path to the output file.
    video : torch.Tensor
        A torch tensor containing the video data
    frame_rate : int
        The video input frame rate (in frames/sec).
    frame_size : Union[int, Tuple[int, int], None], optional (default=None)
        Target frame size (width, height). If None, frame_size is the native
        frame size given by the size of the input tensor `video`. The value can
        be an `int` giving the height of the frame, the height will be
        automatically calculated by respecting the aspect ratio. With the same
        effect, it is possible to define only one component, either height or
        width, and set the other component to -1.
    filters : Optional[str], optional (default=None)
        Add a FFmpeg filtergraph, see https://ffmpeg.org/ffmpeg-filters.html.
    overwrite : bool, optional (default=True)
        Overwrite output file if it exists.
    codec : str, optional (default="libx264")
        Video codec to be used to encode the data, see the FFmpeg documentation
        (https://ffmpeg.org/ffmpeg-codecs.html) for the list of compatible
        codecs.
    data_format : str, optional (default="channels_first")
        The ordering of the dimensions of the input tensor `video`.
        If "channels_last", data_format corresponds to output with shape
        (seq_len, height, width, channels) while "channels_first" corresponds
        to inputs with shape (seq_len, channels, height, width).

    Raises
    ------
    TypeError
        [description]
    ValueError
        [description]
    subprocess.CalledProcessError
        [description]
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
        f"{ffmpeg.path.FFMPEG_BIN} -loglevel fatal {'-y' if overwrite else ''}"
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


def write(
    audio_data: Optional[torch.Tensor], video_data: Optional[torch.Tensor], fpath: str
) -> None:
    if video_data is not None:
        write_video()
    if audio_data is not None:
        pass
