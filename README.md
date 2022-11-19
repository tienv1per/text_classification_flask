Simple Text Classification applied Machine Learning with Flask</br>

## Installation
### Clone this repo
```console
git clone https://github.com/tienv1per/text_classification_flask.git
```
### Create an environment
Whatever you prefer (e.g. `conda` or `venv`)
```console
mkdir myproject
$ cd myproject
$ python3 -m venv venv
```

### Activate
```console
. venv/bin/activate
```

### Install requirements dependencies
 ```console
pip install -r requirements.txt
 ```
 
 Download the training dataset [here](https://www.kaggle.com/datasets/kazanova/sentiment140)</br>
 
 Download the pretrained model [here](https://drive.google.com/drive/u/4/folders/1vmLJYoMUP4BQxnxjSw6lCt9yNgyEd238)</br> 
 Or you can train from scratch with
 ```console
 python3 text_classification.py
 ```
 
## Usage
Run
```console
python3 app.py
```

Send a post request:
```console
curl -X POST -H "Content-Type: application/json" -d '{"text": "Whaterver you want to predict"}' 0.0.0.0:5000/predict
```

Or you can use Postman:</br>
method = "POST"</br>
url: http://localhost:5000/predict</br>
Headers: Content-Type: application/json</br>
Body: raw: {
    "text": "Whaterver you want to predict"
}
</br>
then it will return a response json form

