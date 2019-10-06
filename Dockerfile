FROM pytorch/pytorch

ADD ./ /workspace

RUN pip install -r requirements.txt
