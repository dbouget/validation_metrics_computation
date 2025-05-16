FROM python:3.9-slim

MAINTAINER David Bouget <david.bouget@sintef.no>

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get -y install sudo
RUN apt-get update && apt-get install -y git

WORKDIR /workspace

RUN git clone https://github.com/dbouget/validation_metrics_computation.git
RUN pip3 install --upgrade pip
RUN pip3 install -e validation_metrics_computation/

RUN mkdir /workspace/resources

# CMD ["/bin/bash"]
ENTRYPOINT ["python3", "/workspace/validation_metrics_computation/main.py"]
