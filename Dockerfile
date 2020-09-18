FROM venalone/tensorrt7:cuda10.2

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

COPY . /workspace/
WORKDIR /workspace
