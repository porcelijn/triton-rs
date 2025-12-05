FROM rust:slim AS builder

WORKDIR /home/juvoly

COPY Cargo.toml                 .
COPY Cargo.lock                 .
COPY triton-rs/                 triton-rs/
COPY triton-sys/                triton-sys/
COPY example-backend/Cargo.toml example-backend/
COPY example-backend/src/       example-backend/src/

RUN cargo test --release
RUN cargo build --release --lib

################################################################################
# Test image (sanity check)
FROM nvcr.io/nvidia/tritonserver:25.08-py3 AS tester

WORKDIR /home/juvoly

COPY example-backend/tritonserver/model-repository/ /model-repository/
COPY --from=builder /home/juvoly/target/release/libtriton_example.so /model-repository/example-test/1/
COPY integration-test.sh .

# Start tritonserver and run in background for 2 seconds, capturing exit status
RUN sh integration-test.sh
