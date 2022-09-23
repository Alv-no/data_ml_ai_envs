FROM python:3.10.6-bullseye

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Oslo
RUN apt-get update && apt upgrade -y

RUN pip install --upgrade pip
RUN useradd -m -s /bin/bash docker_user
USER docker_user
ENV PATH "$PATH:/home/docker_user/.local/bin"
COPY ./requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/home/docker_user/base_analytics/tests:/home/docker_user/base_analytics:"
WORKDIR /home/docker_user/base_analytics
ENTRYPOINT ["bash"]