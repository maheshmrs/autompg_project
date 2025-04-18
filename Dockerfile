# pull python base image
FROM python:3.10-slim

# copy application files
ADD /autompg_api /autompg_api/

# specify working directory
WORKDIR /autompg_api

# update pip
RUN pip install --upgrade pip
RUN pwd
# install dependencies
RUN pip install -r autompg_api/requirements.txt

# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "app/main.py"]