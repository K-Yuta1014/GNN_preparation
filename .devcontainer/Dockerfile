FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    build-essential ca-certificates wget \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml .

RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict && \
    conda env create -f environment.yml && \
    conda clean -afy

# 以降は「必ず」gnn_env の中で実行する（ログも出す）
RUN conda run -n gnn_env --no-capture-output python -m pip install --no-cache-dir -U pip

# torch/torchvision を gnn_env に固定（cu121）
RUN conda run -n gnn_env --no-capture-output python -m pip install --no-cache-dir \
    torch==2.2.1+cu121 torchvision==0.17.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# PyG の依存 wheel を先に明示インストール（事故率を下げる）
RUN conda run -n gnn_env --no-capture-output python -m pip install --no-cache-dir \
    torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.2.1+cu121.html

# 最後に torch_geometric 本体
RUN conda run -n gnn_env --no-capture-output python -m pip install --no-cache-dir \
    torch_geometric

# PATH 設定
ENV PATH=/opt/conda/envs/gnn_env/bin:$PATH

# 任意：bash 起動時に自動で activate
RUN echo "conda activate gnn_env" >> /root/.bashrc
