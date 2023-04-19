import random

from flask import Flask, request

from dataset import *
from model import *
from autocorrect import Speller

from flask_cors import CORS

from faker import Faker
import random
from waitress import serve

from flask import Flask, render_template, request
app = Flask(__name__)
CORS(app)

### START PRE PROCESS STEPS ###
dataset = dataset()
dataset.process_data()
model = model()
spell = Speller(lang='en')

model.tain_model(dataset.data, 64, 64)


### END PRE PROCESS STEPS ###

@app.route("/")
def home():
    return render_template("index.html")



@app.route('/ask', methods=['GET', 'POST'])
def index():
    question = ""
    request_data = request.get_json()

    if request_data:
        if 'question' in request_data:
            question = request_data['question']
    if model.model_trained:

        question = spell(question)
        if "" == question:
            return random.choice(dataset.responses['BLANK'])
        intent_current = model.predict(question)
        if "INVALID" != intent_current:
            answer = random.choice(dataset.responses[intent_current])
            if not (answer and answer.strip()):
                answer = "I have not Trained on " + intent_current
        else:
            answer = "SORRY! I cant help you with that, I have not trained on that"

        return str(answer)
    else:
        return "MODEL IS NOT TRAINED, CONTACT TEAM!!"




@app.route('/train')
def train():
    epochs = int(request.args.get('epochs', default=64))
    batch_size = request.args.get('batch', default=64)
    model.tain_model(dataset.data, epochs, batch_size)
    return "MODEL TRAINED, USE /ask ENDPOINT"


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    serve(app, host='0.0.0.0', port=port)
