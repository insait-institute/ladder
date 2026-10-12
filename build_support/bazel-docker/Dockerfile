ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE} as base

ARG BAZEL_VERSION=8.3.1

RUN apt update \
  && apt install -y \
    make \
    git \
    vim \
    less \
  # Install bazel dependencies
    apt-transport-https \
    curl \
    gnupg \
  # Install bazel
  && curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg \
  && mv bazel-archive-keyring.gpg /usr/share/keyrings \
  && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list \
  && apt update \
  && apt install bazel-${BAZEL_VERSION} -y\
  # Add bazel to $PATH
  && ln -s /usr/bin/bazel-${BAZEL_VERSION} /usr/local/bin/bazel \
  # Clean up
  && apt autoremove \
  && apt clean \
  && rm -rf /var/lib/apt/lists/*


ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ARG HOME=/home/${USERNAME}

# Create a non-root user
# NOTE: None of the following is necessary on ubuntu:24.04
RUN groupadd --gid $USER_GID $USERNAME && \
  useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME && \
  echo "ubuntu ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers && \
  chmod 0440 /etc/sudoers && \
  chmod g+w /etc/passwd

USER ${USERNAME}

ARG DOCKER_BAZEL_CACHE=/tmp/bazel-docker-cache

# Symlink bazel cache
RUN mkdir -p $HOME/.cache && ln -s ${DOCKER_BAZEL_CACHE} $HOME/.cache/bazel

WORKDIR $HOME
