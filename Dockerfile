FROM pytorch/pytorch:latest

ADD ./ /workspace

RUN pip install -r requirements.txt
