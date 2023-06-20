from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
        os.system(f"python video_handler.py --input_video {uploaded_file.filename}")

        with open("subtitles.txt", "r", encoding='utf-8') as f:
            prediction = f.read()

        os.remove(uploaded_file.filename)
        os.remove("subtitles.txt")
        
        return jsonify(prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
