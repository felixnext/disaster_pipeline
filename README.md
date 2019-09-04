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

### External Libraries

The code depends on various external libraries:

* Python Tool Stack (`pandas`, `numpy`, `scikit-learn`)
* TensorFlow
* Flask
* SQLAlchemy

## License

Published under MIT License
