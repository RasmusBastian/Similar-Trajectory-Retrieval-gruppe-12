# Builds FMM, the map matching library used by legacy/telchar, from source.
# legacy/telchar ships prebuilt binaries, but they need the exact Arch Linux libraries they were built
# against (GDAL .so.38, Boost 1.89, Python 3.13). Ubuntu 22.04 matches FMM's build instructions
# (its CMake setup needs Python's distutils, which was removed in Python 3.12).
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates swig \
        libboost-dev libboost-serialization-dev libgdal-dev gdal-bin libbz2-dev libexpat1-dev \
        python3-dev python3-pip python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# Latest FMM commit (April 2024)
ARG FMM_COMMIT=19ef34e1f57ff2f2484231aa0d01dfffea986ec1
RUN git clone https://github.com/cyang-kth/fmm.git /opt/fmm \
    && cd /opt/fmm && git checkout "$FMM_COMMIT" \
    && mkdir build && cd build && cmake .. && make -j"$(nproc)" && make install && ldconfig

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY match_beijing.py .
ENTRYPOINT ["python", "match_beijing.py"]
