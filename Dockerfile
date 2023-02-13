# For more information, please refer to https://aka.ms/vscode-docker-python
FROM ubuntu:latest

RUN apt update

RUN apt install python3 -y 

FROM python:3
    RUN pip install protobuf
    RUN pip install streamlit
    RUN pip install scikit-learn

WORKDIR /usr/app/src

# Install pip requirements
COPY prediction.py ./
COPY model_normalization_constants.pkl ./
COPY NN_model.sav ./
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["prediction.py"]

