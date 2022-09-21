FROM  python:3.11.0rc1-bullseye

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Oslo

RUN pip install --upgrade pip
RUN useradd -m -s /bin/bash docker_user
USER docker_user
ENV PATH "$PATH:/home/docker_user/.local/bin"
COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/home/docker_user/python_environment/tests:/home/docker_user/python_environment:"

WORKDIR /home/docker_user/python_environment
ENTRYPOINT ["bash"]
