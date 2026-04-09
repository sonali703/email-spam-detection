from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    data = vectorizer.transform([message])
    result = model.predict(data)

    if result[0] == 1:
        output = "Spam Message ❌"
    else:
        output = "Not Spam ✅"

    return render_template('index.html', prediction=output)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
