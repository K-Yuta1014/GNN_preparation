FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# システムパッケージのインストール
RUN apt-get update && apt-get install -y \
    build-essential ca-certificates wget git \
    && rm -rf /var/lib/apt/lists/*

# environment.yml のコピー
COPY environment.yml .

RUN conda update -n base -c defaults conda -y
RUN conda config --set solver libmamba
RUN conda env create -f environment.yml
RUN conda clean -afy

# 以降の pip インストール（--no-cache-dir を徹底してメモリと容量を節約）
ENV CONDA_DEFAULT_ENV=gnn_env
ENV PATH=/opt/conda/envs/gnn_env/bin:$PATH

RUN python -m pip install --no-cache-dir -U pip

# PyG 関連のインストール（一気に書くことで依存解決を1回にする）
RUN python -m pip install --no-cache-dir \
    torch==2.2.1+cu121 torchvision==0.17.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN python -m pip install --no-cache-dir \
    torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.2.1+cu121.html

RUN python -m pip install --no-cache-dir torch_geometric

# シェルの設定
RUN echo "conda activate gnn_env" >> /root/.bashrc
