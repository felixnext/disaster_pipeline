# Disaster Response Pipeline Project

Natural Language Processing Pipeline that analyzes text messages and classifies them for various disaster responses.

> This data is based on Udacity course for Data Science NanoDegree

## Getting Started

If you just want to get the system up and running, you can use the provided docker file to build the system (this requires docker to be installed on your system):

```bash
$ docker build -t felixnext/disaster_pipeline .
$ docker run -d -p 8000:3001 --name disaster_pipe felixnext/disaster_pipeline
```

The service should now run on your system and be reachable through: `https://localhost:8000/`.

In order to manually setup the system, you have to follow some distinct steps:

1. Clean the relevant data and train classifier models:
  ```bash
  $ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
  $ python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
  ```
2. Go in the `app` folder and run `python run.py`
3. Service should now be reachable under `https://localhost:3001/`

## Algorithm Design

The system is divided into two parts: An ETL pipeline to load and clean the data and the machine learning pipeline to build the model. They are defined in `data/process_data.py` and `models/train_classifier.py` respectively. The ETL pipeline is pretty straight forward. However, the machine learning pipeline allows to provide different configurations (defined in the `config.py`). The pipeline has three distinct steps:

1. Processing: Loads the data created by the ETL pipeline and splits it into distinct features to be used by the classifiers
2. Classifier: Defines the actual classifiers used by the pipeline and which features they use
3. Training: Defines additional parameters required for the training of the classifier

I have run experiments against using different configurations. The results can be found in the `experiments.csv` file.

### Experiment Design

The design of the experiments follows a sequential approach. First of all I used a `MultiOutputClassifier` and `LogisticRegression` as a baseline for the classifiers. They are working on basic simple lemmatized words embedded through TF-IDF.

From there on I experimented with the data in relation to target variables to build hypothesis about the usefulness of certain features (you can see the process in the ML Notebook).
I then tested these hypothesis against the baseline model.

The current best approach is:

TODO

## External Libraries

The code depends on various external libraries:

* Python Tool Stack (`pandas`, `numpy`, `scikit-learn`)
* TensorFlow
* Flask
* SQLAlchemy

### Experimental

* GloVe - Uses a pretrained twitter-model from [here](https://github.com/stanfordnlp/GloVe)

> Glove Training: https://github.com/JonathanRaiman/glove

## License

Published under MIT License
