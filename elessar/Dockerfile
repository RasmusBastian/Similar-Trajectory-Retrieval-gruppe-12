FROM ubuntu:24.04

# Cross compile to server first
# zig build -Dtarget=x86_64-linux-gnu -Doptimize=ReleaseFast

WORKDIR /app
COPY zig-out/bin/elessar /app/elessar
COPY data /app/data

#Remove this line when Mads has moved onnx to data folder
COPY pytorch-lstm /app/pytorch-lstm

COPY onnxruntime-linux-x64/lib/*.so /usr/local/lib/
COPY .zig-cache/o/*/libfacil.io.so /usr/local/lib/

RUN ldconfig

EXPOSE 3030
CMD ["./elessar"]

# After this
# docker buildx build --platform linux/amd64 -t sebylizz/p5-elessar --push .
