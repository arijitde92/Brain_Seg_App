# syntax=docker/dockerfile:1
# Use an official Python runtime as a parent image
FROM python:3.10.14-slim

# Set environment variables to prevent Python from buffering outputs
ENV PYTHONUNBUFFERED=1

# Install dependencies
RUN apt-get update && apt-get install -y wget bzip2 && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/miniconda && \
    rm ~/miniconda.sh

# RUN conda create -n hipp python=3.10 numpy pandas matplotlib tqdm scikit-learn scikit-image nibabel itk simpleitk flask monai tensorboard plotly pytorch torchvision torchaudio pytorch-cuda=12.1 ignite ants -c pytorch -c nvidia -c simpleitk -c conda-forge -c anaconda -c aramislab && conda activate hipp

# Add Miniconda to PATH
ENV PATH="/opt/miniconda/bin:$PATH"

# Copy the environment file
COPY hipp.yml .

# Create the environment and install the packages in one step
RUN conda env create -f hipp.yml && conda clean -afy

# Activate the environment and install additional Python packages in one step
RUN /bin/bash -c "source activate hipp && pip install antspyx hsf[cpu] onnxruntime==1.18.1"

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 85 available to the world outside this container
EXPOSE 85

# Ensure the environment is activated when the container starts, and run the application
CMD ["/bin/bash", "-c", "source activate hipp && python app/app.py"]
