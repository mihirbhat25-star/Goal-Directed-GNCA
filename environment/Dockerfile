FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3
ENV TF_USE_LEGACY_KERAS=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1
COPY environment/requirements-ngc.txt /tmp/gnca-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/gnca-requirements.txt
WORKDIR /workspace/gnca
CMD ["bash"]
