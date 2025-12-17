FROM nvcr.io/nvidia/pytorch:24.12-py3

COPY ./requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip && pip install -r requirements.txt

WORKDIR /source