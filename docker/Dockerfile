FROM python:3.12.0-slim-bullseye

# Install build-essential (including gcc)
RUN apt-get update && \
    apt-get install -y git && \
    apt-get install -y build-essential

# Copy over the code
COPY . /sciterra

# Install sciterra
WORKDIR /sciterra
RUN pip install -e .

# Download a trained model
RUN python -m spacy download en_core_web_sm