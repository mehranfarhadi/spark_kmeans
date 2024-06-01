ARG IMAGE_VARIANT=slim-buster
ARG OPENJDK_VERSION=8
ARG PYTHON_VERSION=3.9.8

FROM python:${PYTHON_VERSION}-${IMAGE_VARIANT} AS py3
FROM openjdk:${OPENJDK_VERSION}-${IMAGE_VARIANT}

COPY --from=py3 / /

# Set environment variable for the PySpark version
ARG PYSPARK_VERSION=3.2.0

# Install PySpark and numpy
RUN pip --no-cache-dir install pyspark==${PYSPARK_VERSION} numpy

# Copy the Python script into the Docker image
COPY kmeans_clustering.py /opt/kmeans_clustering.py

# Copy the CSV data file into the Docker image
COPY data.csv /opt/data.csv

# Set the working directory
WORKDIR /opt

# Execute the Python script
ENTRYPOINT ["python", "kmeans_clustering.py"]
