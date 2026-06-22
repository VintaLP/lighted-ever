import subprocess


def main():
    docker_cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", "/tmp/NVIDIA:/tmp/NVIDIA",
        # "--user", "$(id -u):$(id -g)",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility",
        # This requires the script to be executed in the repository root
        "-v", ".:/ever_training",
        "ever",
        "bash", "-c",
        (
            "source activate ever && "
            "rm -rf ever/build && " # always rebuild
            "mkdir ever/build && cd ever/build && "
            'cmake -DOptiX_INSTALL_DIR=$OptiX_INSTALL_DIR -DCMAKE_CUDA_ARCHITECTURES="50;60;61;70;75;80;86" .. && '
            "make -j8"
        )
    ]
    
    print("Running:", " ".join(docker_cmd))  # For debugging
    subprocess.run(docker_cmd, check=True)

if __name__ == "__main__":
    main()
