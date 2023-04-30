FROM amazonlinux:latest
COPY --from=openjdk:8-jre-slim /usr/local/openjdk-8 /usr/local/openjdk-8
ENV JAVA_HOME /usr/local/openjdk-8
RUN update-alternatives --install /usr/bin/java java /usr/local/openjdk-8/bin/java 1
WORKDIR /app

RUN yum update


RUN yum install python -y
RUN python -m ensurepip --upgrade
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY application ./application

WORKDIR /app/application
RUN pip3 install -r requirements.txt

RUN yum install vim -y

RUN ls -la

EXPOSE 80

ENV NAME World

CMD ["python3", "main.py"]
