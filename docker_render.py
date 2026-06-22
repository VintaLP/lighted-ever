#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(
        description="Run a Docker container for training with dataset & output mounted."
    )
    # Named argument -s / --scene => dataset directory
    parser.add_argument(
        "-s", "--scene",
        required=True,
        help="Path to the dataset directory (e.g. /data/nerf_datasets/zipnerf_ud/london)."
    )
    # Named argument -m / --model_path => output directory
    parser.add_argument(
        "-m", "--model_path",
        default="./model_output",
        help="Path to the model/output directory."
    )
    # Optional port/ip
    parser.add_argument(
        "--port",
        default="6009",
        help="Port to map inside the container. Defaults to 6009."
    )
    parser.add_argument(
        "--ip",
        default="127.0.0.1",
        help="IP to bind the port to. Defaults to 127.0.0.1."
    )

    parser.add_argument(
        "--mode",
        default="lighted",
        choices=["lighted", "no_lighting", "normals"],
        help="Rendering mode: lighted, nolight, or normals."
    )

    # Use parse_known_args to capture any extra flags (unknown) 
    # that we want to forward to train.py:
    known_args, unknown_args = parser.parse_known_args()

    # Now build the docker command:
    docker_cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", "/tmp/NVIDIA:/tmp/NVIDIA",
        # "--user", "$(id -u):$(id -g)",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility",
        # Mount the scene/dataset directory and model_path
        "-v", f"{known_args.scene}:/data/dataset",
        "-v", f"{known_args.model_path}:/data/output",
        # This requires the script to be executed in the repository root
        "-v", ".:/ever_training",
        "--net=host",
        "ever",
        "bash", "-c",
        (
            "source activate ever && "
            # "$@" references extra arguments from the final "_" placeholder
            f"python render.py -s /data/dataset -m /data/output --mode {known_args.mode} \"$@\""
        ),
        "_"  # Placeholder for extra arguments
    ]

    # Append the unknown_args so train.py sees them
    docker_cmd += unknown_args
    
    print("Running:", " ".join(docker_cmd))  # For debugging
    subprocess.run(docker_cmd, check=True)

if __name__ == "__main__":
    main()

