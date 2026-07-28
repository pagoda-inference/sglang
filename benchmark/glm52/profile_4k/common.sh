#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH=/tmp/sglang/python
export SGLANG_ALLOW_CROSS_MACHINE_CP=1
export NCCL_IB_HCA=mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_8,mlx5_9,mlx5_10,mlx5_11
export NCCL_IB_GID_INDEX=2
export NCCL_IB_ROCE_VERSION_NUM=1
export MC_GID_INDEX=4
export MC_TCP_ENABLE_CONNECTION_POOL=true

MODEL=/models/public/modellist/452/V1
