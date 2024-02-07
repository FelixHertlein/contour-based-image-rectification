FROM ubuntu:22.04

RUN apt -f install
RUN apt update && apt -y dist-upgrade 
RUN apt install -y python3-pip git python-is-python3 ffmpeg libsm6 libxext6

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /opt/app/requirements.txt
RUN pip3 install -r /opt/app/requirements.txt
RUN rm /opt/app/requirements.txt

RUN git config --global --add safe.directory /workspaces/contour-based-image-rectification
RUN git config --global core.autocrlf true
