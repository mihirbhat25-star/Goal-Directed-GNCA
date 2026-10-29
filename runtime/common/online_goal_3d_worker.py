"""Private subprocess entry point for one cached or compiled online-goal 3D chunk."""

import argparse
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"

from runtime.common.subprocess_utils import install_filtered_stderr

install_filtered_stderr()

import joblib

from runtime.common.online_goal_3d import _worker


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--args_file", required=True)
    parser.add_argument("--chunk_index", required=True, type=int)
    parser.add_argument("--chunk_mix_file", required=True)
    parser.add_argument("--state_dir", required=True)
    parser.add_argument("--result_path", required=True)
    parser.add_argument("--run_tag", required=True)
    cli = parser.parse_args()
    args = joblib.load(cli.args_file)
    with open(cli.chunk_mix_file, encoding="utf-8") as handle:
        chunk_mix = {
            int(octant): int(count)
            for octant, count in json.load(handle).items()
        }
    _worker(
        args,
        cli.chunk_index,
        chunk_mix,
        cli.state_dir,
        cli.result_path,
        cli.run_tag,
    )


if __name__ == "__main__":
    main()
