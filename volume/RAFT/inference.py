import os
import sys

os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "8000"
sys.path.insert(0, os.path.join(os.path.abspath(os.getcwd()), "core"))
print(os.path.join(os.path.abspath(os.getcwd()), "core"))

from argparse import ArgumentParser
from inferenceUtils import OpticalFlowProcessor


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model", help="restore checkpoint", default="models/raft-sintel.pth"
    )
    parser.add_argument("--iters", type=int, default=12)

    parser.add_argument("--video", type=str, default="./videos/input_video.mp4")
    parser.add_argument("--save", action="store_true", help="save demo frames")
    parser.add_argument(
        "--small", action="store_true", help="use small model", default=False
    )
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )

    args = parser.parse_args()
    of = OpticalFlowProcessor(args)
    of.run_inference("./videos/input_video.mp4")


if __name__ == "__main__":
    main()
