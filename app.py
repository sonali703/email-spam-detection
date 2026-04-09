from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    vec = vectorizer.transform([message])
    
    result = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    if result == 1:
        output = f"Spam ❌ ({prob[1]*100:.2f}%)"
    else:
        output = f"Not Spam ✅ ({prob[0]*100:.2f}%)"

    return render_template('index.html', prediction=output)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
