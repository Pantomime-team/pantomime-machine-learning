from flask import Flask, request, jsonify
from video_handler import video_process

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
        prediction = video_process(uploaded_file.filename)
        return jsonify(prediction)


if __name__ == '__main__':
    app.run()
