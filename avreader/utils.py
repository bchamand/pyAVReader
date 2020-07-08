import re
import subprocess
from typing import Dict, Optional, Sequence, Tuple, Union

import avreader.path


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

    Convert a string time duration in the form of a float time duration in seconds.

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
        Extracted information from the audiovisual file structured as follows::

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
        f"{avreader.path.FFPROBE_BIN} -loglevel fatal"
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
