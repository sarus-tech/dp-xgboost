# Sarus XGBoost Docker Image 

# Install python 3。6.
FROM python:3.6

RUN apt-get update

# Install build tools 
RUN apt-get install -y git make g++ cmake libomp-dev

RUN mkdir -p /opt/xgb 

# Download the rabit tracker and xgboost code.

COPY sarus-docker/requirements.txt /opt/xgb/

# Install requirements

RUN pip install -r /opt/xgb/requirements.txt

# Build Sarus XGBoost 

RUN mkdir /opt/xgb/sarus-xgboost 
COPY ./ /opt/xgb/sarus-xgboost

RUN cd /opt/xgb/sarus-xgboost && \
    rm -rf build/ && \ 
    mkdir build/ &&\ 
    cd build && \ 
    cmake ../ && \
    make -j4

# Install the python package 
RUN cd /opt/xgb/sarus-xgboost/python-package/ && \ 
    python -m pip install -e .

# Copy and run example classification

COPY sarus-docker/example_classification.py /opt/xgb/
ENTRYPOINT ["python", "/opt/xgb/example_classification.py"]