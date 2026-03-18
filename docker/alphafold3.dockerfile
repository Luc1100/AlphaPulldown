FROM nvidia/cuda:12.6.3-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------
# System deps
# ---------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        gcc \
        g++ \
        make \
        wget \
        ca-certificates \
        git \
        patch \
        zlib1g-dev \
        zstd \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------
# uv (matching upstream AF3 Dockerfile)
# ---------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:0.9.24 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv venv --seed $UV_PROJECT_ENVIRONMENT
ENV PATH="/hmmer/bin:${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"

# ---------------------------------------------------------------------
# HMMER (with seq_limit patch)
# ---------------------------------------------------------------------
RUN mkdir /hmmer_build /hmmer && \
    wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz -O /hmmer_build/hmmer-3.4.tar.gz && \
    echo "ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3  /hmmer_build/hmmer-3.4.tar.gz" | sha256sum -c - && \
    tar -xzf /hmmer_build/hmmer-3.4.tar.gz -C /hmmer_build

RUN wget -O /hmmer_build/jackhmmer_seq_limit.patch \
    https://raw.githubusercontent.com/google-deepmind/alphafold3/main/docker/jackhmmer_seq_limit.patch

RUN cd /hmmer_build && \
    patch -p0 < jackhmmer_seq_limit.patch && \
    cd /hmmer_build/hmmer-3.4 && \
    ./configure --prefix=/hmmer && \
    make -j$(nproc) && \
    make install && \
    cd easel && make install && \
    rm -rf /hmmer_build

# ---------------------------------------------------------------------
# Clone AlphaPulldown with submodules
# ---------------------------------------------------------------------
RUN git clone --recurse-submodules https://github.com/Luc1100/AlphaPulldown.git /app/AlphaPulldown

# ---------------------------------------------------------------------
# Install AlphaFold3 via uv (matching upstream approach)
# ---------------------------------------------------------------------
WORKDIR /app/AlphaPulldown/alphafold3

RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy uv sync --all-groups --no-editable

# Build chemical components database
RUN uv run build_data

# ---------------------------------------------------------------------
# Install AlphaPulldown
# --no-deps avoids re-resolving/clobbering the AF3 that uv installed.
# Install AlphaPulldown's own dependencies first, then the package.
# ---------------------------------------------------------------------
WORKDIR /app/AlphaPulldown
RUN pip install --no-cache-dir --no-deps . && \
    pip install --no-cache-dir \
        "absl-py>=0.13.0" \
        "alphapulldown-input-parser>=0.3.0" \
        "dm-tree>=0.1.6" \
        "h5py>=3.1.0" \
        "matplotlib>=3.3.3" \
        "ml-collections>=0.1.0" \
        "pandas>=1.5.3" \
        "tensorflow-cpu>=2.16.1" \
        "importlib-resources>=6.1.0" \
        "importlib-metadata>=4.8.2,<5.0.0" \
        "biopython>=1.82" \
        "nbformat>=5.9.2" \
        "py3Dmol==2.0.4" \
        "tqdm>=4.66.1" \
        "appdirs>=1.4.4" \
        "ml-dtypes" \
        "chex>=0.1.86" \
        "immutabledict>=2.0.0" \
        "typing-extensions>=4.14.0" \
        "openmm>=8.0" \
        "pdbfixer" \
        "requests"

# ---------------------------------------------------------------------
# Runtime env
# ---------------------------------------------------------------------
ENV XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
ENV XLA_PYTHON_CLIENT_PREALLOCATE=true
ENV XLA_CLIENT_MEM_FRACTION=0.95
ENV TF_FORCE_UNIFIED_MEMORY='1'
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ---------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------
RUN python3 -c "\
from alphafold3.constants import chemical_components; \
from alphapulldown.folding_backend.folding_backend import FoldingBackend; \
print('AF3 + AlphaPulldown import OK, CCD present')\
"
