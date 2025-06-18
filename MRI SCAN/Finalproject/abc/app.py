import os
import base64
import io
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import openai
from dotenv import load_dotenv
import jwt
import datetime
from fpdf import FPDF
from functools import wraps
import time
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'default_secret')
JWT_SECRET = os.getenv('JWT_SECRET', 'jwt_secret')

openai.api_key = OPENAI_API_KEY

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
CORS(app)

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Cache for OpenAI responses
response_cache = {}
CACHE_EXPIRY = 3600  # 1 hour in seconds

def cache_response(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        cache_key = str(args) + str(kwargs)
        current_time = time.time()
        
        # Check cache
        if cache_key in response_cache:
            timestamp, response = response_cache[cache_key]
            if current_time - timestamp < CACHE_EXPIRY:
                return response
        
        # Get fresh response
        response = func(*args, **kwargs)
        response_cache[cache_key] = (current_time, response)
        return response
    return wrapper

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

@app.route('/llm', methods=['POST'])
@limiter.limit("10 per minute")
@cache_response
def llm():
    data = request.json
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,  # Limit response length for cost efficiency
            temperature=0.7  # Balance between creativity and consistency
        )
        return jsonify({'response': response['choices'][0]['message']['content']}), 200
    except openai.error.RateLimitError:
        return jsonify({'error': 'OpenAI API rate limit exceeded. Please try again later.'}), 429
    except openai.error.APIError as e:
        return jsonify({'error': f'OpenAI API error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
@limiter.limit("5 per minute")
@cache_response
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        image_bytes = file.read()
        if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
            return jsonify({'error': 'File size too large. Maximum size is 10MB.'}), 400
            
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        prompt = (
            "You are a medical AI assistant. Given the following MRI scan image (base64-encoded), "
            "analyze it and answer: Does this scan show a brain tumor? Respond with a clear yes/no and a brief explanation. "
            "Base64 image: " + image_b64
        )
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.3  # Lower temperature for more consistent medical analysis
        )
        return jsonify({'result': response['choices'][0]['message']['content']}), 200
    except openai.error.RateLimitError:
        return jsonify({'error': 'OpenAI API rate limit exceeded. Please try again later.'}), 429
    except openai.error.APIError as e:
        return jsonify({'error': f'OpenAI API error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/generate_pdf', methods=['POST'])
@limiter.limit("20 per minute")
def generate_pdf():
    data = request.json
    report_text = data.get('report', 'No report provided.')
    
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, report_text)
        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        pdf_output.seek(0)
        return send_file(pdf_output, as_attachment=True, download_name='report.pdf', mimetype='application/pdf')
    except Exception as e:
        return jsonify({'error': f'PDF generation failed: {str(e)}'}), 500

@app.route('/create_jwt', methods=['POST'])
@limiter.limit("30 per minute")
def create_jwt():
    data = request.json
    username = data.get('username', 'user')
    payload = {
        'username': username,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }
    try:
        token = jwt.encode(payload, JWT_SECRET, algorithm='HS256')
        return jsonify({'token': token})
    except Exception as e:
        return jsonify({'error': f'JWT creation failed: {str(e)}'}), 500

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
    
@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/index.html')
def index_html():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # TODO: Add upload handling logic here
    return 'Upload endpoint hit'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 
