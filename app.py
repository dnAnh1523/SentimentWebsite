from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import underthesea
import numpy as np
import re
from gensim.models.keyedvectors import KeyedVectors
from tensorflow.keras.models import load_model
import os

# Set paths for model and HTML files directly in the same directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models.h5')
WORD_MODEL_PATH = os.path.join(BASE_DIR, 'word.model')
HTML_FILE = os.path.join(BASE_DIR, 'index.html')
CSS_FILE = os.path.join(BASE_DIR, 'style.css')

# Tải mô hình đã huấn luyện
model = load_model(MODEL_PATH)
model_embedding = KeyedVectors.load(WORD_MODEL_PATH)

max_seq = 200
embedding_size = model_embedding.vector_size

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            # Serve the index.html file
            try:
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                with open(HTML_FILE, 'rb') as f:
                    self.wfile.write(f.read())
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'File not found.')
        elif self.path == '/style.css':
            # Serve the CSS file
            try:
                self.send_response(200)
                self.send_header('Content-type', 'text/css')
                self.end_headers()
                with open(CSS_FILE, 'rb') as f:
                    self.wfile.write(f.read())
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'File not found.')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            comment = data.get('comment')

            if comment:
                # Tiến hành xử lý và dự đoán
                processed_text = pre_process(comment)
                embedded_comment = np.expand_dims(comment_embedding(processed_text), axis=0)
                embedded_comment = np.expand_dims(embedded_comment, axis=3)

                result = model.predict(embedded_comment)
                label = np.argmax(result)

                label_map = {0: 'neutral', 1: 'positive', 2: 'negative'}
                response = {
                    'comment': comment,
                    'prediction': label_map[label]
                }

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'Invalid comment input.')

def pre_process(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', 'link_spam', text)
    text = re.sub(r'(.)\1+', r'\1', text)
    abbreviation_dict = {
        'k': 'không',
        'ko': 'không',
        'bt': 'bình thường',
    }
    words = text.split()
    text = ' '.join([abbreviation_dict.get(word, word) for word in words])
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if len(word) > 1 and not word.isdigit()])
    text = underthesea.word_tokenize(text, format="text")
    return text

def comment_embedding(comment):
    matrix = np.zeros((max_seq, embedding_size))
    words = comment.split()
    for i in range(min(max_seq, len(words))):
        word = words[i]
        if word in model_embedding.key_to_index:
            matrix[i] = model_embedding[word]
    return matrix

def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler):
    server_address = ('', 8000)
    httpd = server_class(server_address, handler_class)
    print('Server đang chạy tại http://localhost:8000')
    httpd.serve_forever()

if __name__ == "__main__":
    run()