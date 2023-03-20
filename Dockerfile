FROM continuumio/miniconda3:latest

SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install nano

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
RUN echo "conda activate graph_env" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# entypoint.sh is used so as to load with configured conda environment!
COPY entrypoint.sh ./
#give permission duh so linux obvious..
RUN chmod +x ./entrypoint.sh
RUN ./entrypoint.sh
#remove environment.yml file
RUN rm environment.yml entrypoint.sh
