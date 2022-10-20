import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = " "
print("Let's chat! (type 'quit' to exit)")
def chat(sentence):
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f" {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")     
app = Flask(__name__)
cors = CORS(app)

@app.route("/", methods=["GET","POST"])
def working():
    print(request)
    return "Working"
@app.route("/sendChat", methods=["POST"])
def hello_world():
    try:
        sentence=request.json
        print('text: ', sentence)
        # print(chat(text))
        result = chat(sentence['sentence'])
        res = {}
        res['result'] = result
        return res
    except(error):
        return error
@app.route("/texttoquestion", methods=["GET", "POST"])
def generate():
    if request.method == "POST":
        text = request.form.get('key')
        print(chat(sentence))
    return chat(sentence)  
