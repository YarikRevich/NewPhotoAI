FROM python:3.6

WORKDIR /NewPhotoAI

COPY . .

RUN apt-get update -y && \
    apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install -r requirements.txt


ENTRYPOINT . ./credentials.sh && python3 main.py --prod



