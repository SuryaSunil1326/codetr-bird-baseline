# pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel, pinned by digest
FROM pytorch/pytorch@sha256:9bfcfa72b6b244c1fbfa24864eec97fb29cfafc065999e9a9ba913fa1e690a02

# GPU architectures the mmcv-full CUDA ops are compiled for
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="/opt/conda/.." \
    FORCE_CUDA="1" \
    PYTHONPATH=/workspace

# this base is Ubuntu 18.04 and its NVIDIA apt key is expired; refresh it before apt-get update
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 || true \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# hash-locked pure-python deps; mmcv imports addict at load time so this goes before mmcv
COPY requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock.txt

# mmcv-full 1.5.0 built from source at the v1.5.0 commit (no wheel index dependency).
# --no-deps: its unpinned opencv-python requirement would replace the locked version
RUN git init /opt/mmcv && cd /opt/mmcv \
    && git remote add origin https://github.com/open-mmlab/mmcv.git \
    && git fetch --depth 1 origin 235c0253ab8806a2a2ee6954b4258a95358497ac \
    && git checkout FETCH_HEAD \
    && MMCV_WITH_OPS=1 pip install --no-cache-dir --no-deps -e . \
    && python -c "from mmcv.ops import MultiScaleDeformableAttention; print('mmcv ops import OK')"

# --no-deps on all four: each declares an unbounded torch requirement that would upgrade the base image's torch
RUN pip install --no-cache-dir --no-deps \
        fairscale==0.4.13 \
        fvcore==0.1.5.post20221221 \
        iopath==0.1.10 \
        timm==1.0.22

# the repo is mounted here at run time, see README
WORKDIR /workspace
CMD ["bash"]
