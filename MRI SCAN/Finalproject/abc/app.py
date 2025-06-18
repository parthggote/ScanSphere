import os
import base64
import io
import time
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template, url_for, redirect
from flask_cors import CORS
import openai
from dotenv import load_dotenv
import jwt
import datetime
from fpdf import FPDF
from functools import wraps
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename

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
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
CORS(app)

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Cache for OpenAI responses
response_cache = {}
CACHE_EXPIRY = 3600  # 1 hour in seconds

def allowed_file(filename):
    """Check if the file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(file_path):
    """Mock function to simulate image prediction - replace with actual model inference"""
    # This is a placeholder - replace with actual model inference
    return {
        'class': 'Normal',
        'class_id': 0,
        'confidence': 87.3,
        'probabilities': [0.127, 0.873]
    }

def generate_diagnostic_report(prediction):
    """Generate a diagnostic report based on the prediction"""
    if prediction['class_id'] == 0:
        return f"""
        <div class='diagnostic-report'>
            <h3>Diagnostic Report</h3>
            <p>Based on the analysis of the MRI scan, no significant abnormalities were detected.</p>
            <p>Confidence: {prediction['confidence']}%</p>
            <p>Recommendation: Regular follow-up as per standard protocol.</p>
        </div>
        """
    else:
        return f"""
        <div class='diagnostic-report'>
            <h3>Diagnostic Report</h3>
            <p>Based on the analysis of the MRI scan, potential abnormalities were detected.</p>
            <p>Confidence: {prediction['confidence']}%</p>
            <p>Recommendation: Please consult with a medical professional for further evaluation.</p>
        </div>
        """

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
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a PNG, JPG, JPEG, or GIF file.'}), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Start timing
        start_time = time.time()
        
        # Get prediction
        prediction = predict_image(file_path)
        
        # Generate diagnostic report
        diagnostic_report = generate_diagnostic_report(prediction)
        
        # Calculate processing time
        processing_time = f"{time.time() - start_time:.2f}s"
        
        # Prepare scan data for JavaScript
        scan_data = {
            'filename': filename,
            'prediction': prediction['class'],
            'analysis': diagnostic_report,
            'confidence': prediction['confidence'],
            'probabilities': prediction['probabilities']
        }
        
        # Render result template with proper static file URL
        return render_template('result.html',
                             image_file=url_for('static', filename=os.path.join('uploads', filename)),
                             is_base64=False,
                             prediction=prediction,
                             diagnostic_report=diagnostic_report,
                             processing_time=processing_time,
                             scan_data=scan_data)
                             
    except Exception as e:
        # Log the error for debugging
        app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

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

@app.route('/ask_ai', methods=['POST'])
@limiter.limit("10 per minute")
def ask_ai():
    try:
        data = request.json
        if not data or not all(key in data for key in ['question', 'prediction', 'analysis']):
            return jsonify({'error': 'Missing required fields'}), 400

        # Construct the prompt
        prompt = f"Question: {data['question']}\nPrediction: {data['prediction']}\nAnalysis: {data['analysis']}"
        
        # Make the API call to OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that explains brain MRI predictions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        # Extract and return the response
        ai_response = response['choices'][0]['message']['content']
        return jsonify({'response': ai_response}), 200
        
    except openai.error.RateLimitError:
        app.logger.error("OpenAI API rate limit exceeded")
        return jsonify({'error': 'OpenAI API rate limit exceeded. Please try again later.'}), 429
    except openai.error.APIError as e:
        app.logger.error(f"OpenAI API error: {str(e)}")
        return jsonify({'error': f'OpenAI API error: {str(e)}'}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error in ask_ai: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 
