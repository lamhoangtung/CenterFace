FROM venalone/tensorrt7:cuda10.2

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

COPY . /workspace/
WORKDIR /workspace
ENV LD_LIBRARY_PATH=/usr/local/tensorrt/lib:/usr/local/tensorrt/targets/x86_64-linux-gnu/lib:/usr/local/cuda-10.2/lib64/:/usr/local/cuda-10.2/extras/CUPTI/lib64/:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
CMD /bin/bash -c "/workspace/init_engine.sh" && /bin/bash
