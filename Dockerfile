# syntax=docker/dockerfile:1
# hadolint global ignore=DL3007

FROM archlinux:latest AS cuda

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN --mount=type=cache,target=/var/cache/pacman/pkg,sharing=locked \
    pacman -Syu --noconfirm --needed cuda cudnn gcc-libs

ENV PATH="/opt/cuda/bin:${PATH}"


FROM cuda AS build

RUN --mount=type=cache,target=/var/cache/pacman/pkg,sharing=locked \
    pacman -S --noconfirm --needed \
        base-devel \
        cmake \
        gcc15 \
        git \
        ninja

WORKDIR /workspace
COPY --link . .
COPY --link assets/ /opt/genesia/assets/

RUN cmake -S . -B cmake-build-release -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
        -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15 \
        -DGENESIA_BUILD_UI=OFF \
        -DGENESIA_ASSET_DIRECTORY=/opt/genesia/assets \
    && cmake --build cmake-build-release --target genesia --parallel


FROM cuda AS runtime

RUN groupadd --gid 10001 genesia \
    && useradd --uid 10001 --gid 10001 --home-dir /workspace --shell /usr/bin/nologin genesia \
    && install --directory --owner=10001 --group=10001 \
        /opt/genesia/bin \
        /workspace \
        /workspace/data \
        /workspace/data/raw \
        /workspace/.genesia \
        /workspace/models \
        /workspace/cmake-build-release/genesia-cache

COPY --from=build --link /workspace/cmake-build-release/genesia /opt/genesia/bin/genesia
COPY --from=build --link --chown=10001:10001 /opt/genesia/assets /opt/genesia/assets
COPY --from=build --link /workspace/LICENSE /opt/genesia/LICENSE

USER 10001:10001
ENV HOME=/workspace \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /workspace

ENTRYPOINT ["/opt/genesia/bin/genesia"]
