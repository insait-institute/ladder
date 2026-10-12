import os
import socket
import subprocess
from typing import List

import psutil


def is_test_env() -> bool:
    return 'TEST_UNDECLARED_OUTPUTS_DIR' in os.environ


def is_running_in_docker() -> bool:
    def docker_in_cgroup():
        with open('/proc/self/cgroup') as f:
            return any('docker' in line for line in f)

    return os.path.exists('/.dockerenv') or (os.path.isfile('/proc/self/cgroup') and docker_in_cgroup())


def _get_ladder_repo_root() -> str:
    """Absolute path to the root of this repository (ladder)"""
    # NOTE: Use os.path.realpath - this also resolves bazel symlinks and directs to the real root path
    return os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/../../..")


# -----------------
# Job metadata
# -----------------

EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME', None)

# -----------------
# SYSTEM
# -----------------

USERNAME = os.getenv('LADDER_USERNAME', '') or os.getenv('USER', '')
HOSTNAME: str = socket.gethostname().split('.')[0]

NODE_VISIBLE_CPUS_COUNT: int = len(os.sched_getaffinity(os.getpid()))  # CPUs visible to this process
NODE_TOTAL_CPUS_COUNT: int = psutil.cpu_count()  # Total number of CPUs on this node

# -----------------
# Repo
# -----------------

# Absolute path to the root of this repository (ladder)
LADDER_ROOT = REPO_ROOT = _get_ladder_repo_root()

# Set environment variable for the repo root
os.environ['LADDER_ROOT'] = LADDER_ROOT

# TODO: Can also try /var/lock or ~/. Can make a function that tests permissions
LADDER_LOCKFILES_DIR = os.getenv('LADDER_LOCKFILES_DIR', '/tmp')

# -----------------
# CUDA
# -----------------


def cuda_visible_devices_ids() -> List[str]:
    """
    Return a list of strings of the values in CUDA_VISIBLE_DEVICES. Note these aren't guaranteed
    to be numbers, e.g. slurm sets gpu ids to hashes, e.g. 'GPU-00000000-0000-0000-0000-000000000000'
    """
    return [gpu_id for gpu_id in os.getenv("CUDA_VISIBLE_DEVICES", "").split(',') if gpu_id != '']


def cuda_visible_devices_indices() -> List[int]:
    """
    Return a list of ints corresponding to the values in CUDA_VISIBLE_DEVICES.
    - If running in SLURM, it would return a list of consecutive numbers and of size equal to the number of
      visible devices. The numbers would correspond to the values you see in `nvidia-smi`. However, the
      actual value of CUDA_VISIBLE_DEVICES will contain ids like 'GPU-00000000-0000-0000-0000-000000000000'
    - If not in SLURM, will contain the actual values in CUDA_VISIBLE_DEVICES env variable, which
      will be a subset of the numbers in `nvidia-smi`, but not guaranteed to be ordered or consecutive
    """
    gpu_ids = cuda_visible_devices_ids()

    if any(not gpu_id.isdigit() for gpu_id in gpu_ids):
        return [i for i in range(len(gpu_ids))]
    return [int(gpu_id) for gpu_id in gpu_ids]


# -----------------
# HUGGING FACE
# -----------------
HF_TOKEN = os.getenv('HF_TOKEN', '')

# -----------------
# Torch
# -----------------
# `torchrun` would define these env variables before launching the script
# https://pytorch.org/docs/stable/elastic/run.html#environment-variables
IS_RUNNING_IN_TORCHRUN = 'TORCHELASTIC_RUN_ID' in os.environ

# -----------------
# Docker
# -----------------
DOCKER_TRAIN_IMAGE = os.getenv('DOCKER_TRAIN_IMAGE', 'no-docker-image-detected')

# -----------------
# Cloud
# -----------------
IS_RUNNING_IN_SLURM = "SLURM_JOB_ID" in os.environ
# TODO: These are just examples. Proper detection needs to be done
IS_RUNNING_IN_CLOUD = os.getenv('IS_RUNNING_IN_CLOUD', '') == 'true'
IS_RUNNING_IN_AWS = os.getenv('IS_RUNNING_IN_AWS', '') == 'true'
IS_RUNNING_IN_AZURE = os.getenv('IS_RUNNING_IN_AZURE', '') == 'true'
IS_RUNNING_IN_GCP = os.getenv('IS_RUNNING_IN_GCP', '') == 'true'

SLURM_NODELIST: List[str] = []
SLURM_JOB_ID: str = ''


def get_slurm_hostnames() -> List[str]:
    assert IS_RUNNING_IN_SLURM
    assert "SLURM_JOB_NODELIST" in os.environ
    nodelist = os.environ["SLURM_JOB_NODELIST"]
    if '[' in nodelist or ']' in nodelist or ',' in nodelist:
        result = subprocess.run(
            ['scontrol', 'show', 'hostnames', nodelist],
            capture_output=True,
            text=True,
            check=True,
        )
        hostnames = result.stdout.strip().split("\n")
    else:
        hostnames = [nodelist]
    return hostnames


if IS_RUNNING_IN_SLURM:
    TRAINING_PLATFORM = "slurm"
    NODE_COUNT = int(os.getenv("SLURM_JOB_NUM_NODES", '-1'))
    SLURM_NODELIST = get_slurm_hostnames()
    SLURM_JOB_ID = os.getenv("SLURM_JOB_ID", "")
elif IS_RUNNING_IN_AWS:
    TRAINING_PLATFORM = "SageMaker"
    NODE_COUNT = int(os.getenv('NODE_COUNT', '-1'))
elif IS_RUNNING_IN_AZURE:
    TRAINING_PLATFORM = "AzureML"
    NODE_COUNT = int(os.getenv('NODE_COUNT', '-1'))
elif IS_RUNNING_IN_GCP:
    TRAINING_PLATFORM = "VertexAI"
    NODE_COUNT = int(os.getenv('NODE_COUNT', '-1'))
else:
    TRAINING_PLATFORM = "Local Machine"
    NODE_COUNT = 1

IS_MULTINODE = NODE_COUNT > 1
