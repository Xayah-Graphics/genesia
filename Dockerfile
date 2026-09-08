# syntax=docker/dockerfile:1
# hadolint global ignore=DL3007

FROM archlinux:latest AS build

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN --mount=type=cache,target=/var/cache/pacman/pkg,sharing=locked \
    pacman -Syu --noconfirm --needed \
        base-devel \
        cmake \
        cuda \
        cudnn \
        git \
        ninja

ENV PATH="/opt/cuda/bin:${PATH}"

WORKDIR /src
COPY --link . .

RUN cmake -S . -B cmake-build-release -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
        -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15 \
        -DGENESIA_BUILD_UI=OFF \
        -DGENESIA_TAG_CATALOG=/opt/genesia/assets/tags/danbooru.csv \
    && cmake --build cmake-build-release --target genesia --parallel


FROM archlinux:latest AS runtime

RUN --mount=type=cache,target=/var/cache/pacman/pkg,sharing=locked \
    pacman -Syu --noconfirm --needed cuda cudnn gcc-libs \
    && groupadd --gid 10001 genesia \
    && useradd --uid 10001 --gid 10001 --home-dir /workspace --shell /usr/bin/nologin genesia \
    && install --directory --owner=10001 --group=10001 /opt/genesia/bin /workspace

COPY --from=build --link /src/cmake-build-release/genesia /opt/genesia/bin/genesia
COPY --from=build --link /src/assets/tags/danbooru.csv /opt/genesia/assets/tags/danbooru.csv
COPY --from=build --link /src/LICENSE /opt/genesia/LICENSE

USER 10001:10001
ENV HOME=/workspace \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /workspace

ENTRYPOINT ["/opt/genesia/bin/genesia"]
