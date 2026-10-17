# cognee-rs-delta deploy image — the standalone cognee-http-server binary on a
# Debian Trixie base. Built by .github/workflows/delta-image.yml, which produces
# dist/cognee-http-server first (same cargo invocation as http-parity, minus the
# harness-only dev-mock feature).
#
# Base choice (ubuntu-24.04 runner, glibc 2.39 binaries): the node fleet is
# RHEL 9.8 (host glibc 2.34). glibc comes from the IMAGE, not the host, so a
# Trixie base (glibc 2.41, libstdc++ from lbug's bundled-C++ era) runs on the
# fleet kernel while an Alinux/RHEL9 base would crash on GLIBC_2.3x symbol
# misses. Older-kernel + newer-glibc-userspace is the supported direction.
#
# No OpenSSL to satisfy: reqwest resolves to rustls. ort links statically
# (ort-sys static lib, per the http-parity cache notes). pgvector/ladybug are
# default features; lancedb/onnx stay compiled-in but unused by this
# deployment's env shape (pgvector + ladybug + openai embedder).
FROM debian:trixie-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Corporate TLS interception: cluster-egress to LLM/embedding endpoints may
# chain through the NetApp SSFMCCA CA (same bundle the Python delta image
# bakes). Harmless where egress is direct; load-bearing where it isn't.
COPY deploy/delta/netapp-ssfmcca.pem /usr/local/share/ca-certificates/netapp-ssfmcca.crt
RUN update-ca-certificates

COPY dist/cognee-http-server /usr/local/bin/cognee-http-server

RUN useradd -r -u 10001 -d /data cognee && mkdir -p /data && chown cognee /data
USER 10001

# Writable roots: ladybug graph files, session store, sqlite fallbacks if any.
ENV HTTP_API_HOST=0.0.0.0 \
    HTTP_API_PORT=8000 \
    DATA_ROOT_DIRECTORY=/data \
    SYSTEM_ROOT_DIRECTORY=/data/system \
    COGNEE_SESSION_DIR=/data/sessions

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/cognee-http-server"]
