FROM python:3.6-slim

# install relevant packages
RUN mkdir /app
WORKDIR /app
COPY ./requirements.txt ./reqs.txt
RUN pip install -r reqs.txt

# execute data
RUN mkdir /app/app
RUN mkdir /app/data
RUN mkdir /app/models

# copy data and generate models
COPY ./data /app/data/
RUN python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
COPY ./models /app/models/
RUN python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

# copy the final app data
COPY ./app /app/app/

# ports
EXPOSE 3001

# run command
WORKDIR /app/app
CMD ["python", "run.py"]
