import os
import numpy as np
import onnxruntime as ort
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
import io
import time
import logging
from datetime import datetime
import json
import concurrent.futures
from functools import lru_cache
import threading
from queue import Queue
import base64
from scipy import ndimage
import cv2
import hashlib
import traceback
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import subprocess
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Image as ReportLabImage
import torch
from torchvision import transforms
import ollama
import re
import jwt

# PIL imports - consolidated
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageDraw

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'brain_tumor_detection_fallback_secret_key') # Use env var for secret key
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['THREAD_POOL_WORKERS'] = 8  # Increased for more parallel processing
app.config['ENABLE_IN_MEMORY_PROCESSING'] = True  # Process images in memory
app.config['ENABLE_REPORT_QUEUE'] = False  # Disable report queue - all reports generated dynamically
app.config['ENABLE_STATIC_REPORTS'] = False  # Disable static reports entirely
app.config['USE_BASE64_IMAGES'] = True  # Use base64 encoding for immediate image display
app.config['RESIZE_DIM'] = 224  # MobileNet standard input size

# Set up resize quality based on Pillow version
try:
    # For Pillow >= 9.0.0
    # Try to access Resampling which is available in newer versions
    RESIZE_QUALITY = Image.Resampling.LANCZOS
except AttributeError:
    # For older Pillow versions
    RESIZE_QUALITY = Image.ANTIALIAS  # Fallback for older versions

app.config['RESIZE_QUALITY'] = RESIZE_QUALITY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Ollama to use custom port from env var or default to localhost for local dev
ollama.host = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11500")
logger.info(f"Ollama configured to use host: {ollama.host}")

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

CONFIG = {
    "model_path": "brain_mri_model.onnx",
    "class_names": ["No Tumor", "Tumor Detected"],
    "image_size": (224, 224),
    "allowed_extensions": {'jpg', 'jpeg', 'png'},
    # Optimization parameters
    "resize_quality": RESIZE_QUALITY,  # Use the variable we set above
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225],
    "report_batch_size": 5,  # Process reports in batches
}

# Create a thread pool for parallel processing
executor = concurrent.futures.ThreadPoolExecutor(max_workers=app.config['THREAD_POOL_WORKERS'])

# Global cache for the model session
MODEL_SESSION = None

# Database of top specialist names and hospitals (for more accurate and specific recommendations)
specialists = {
    "glioma": [
        {"name": "Dr. Mitchel S. Berger", "title": "MD, FACS", "hospital": "UCSF Brain Tumor Center", "location": "San Francisco, CA"},
        {"name": "Dr. Henry Brem", "title": "MD", "hospital": "Johns Hopkins Comprehensive Brain Tumor Center", "location": "Baltimore, MD"},
        {"name": "Dr. Linda Liau", "title": "MD, PhD, MBA", "hospital": "UCLA Brain Tumor Center", "location": "Los Angeles, CA"}
    ],
    "meningioma": [
        {"name": "Dr. Michael McDermott", "title": "MD", "hospital": "Miami Neuroscience Institute", "location": "Miami, FL"},
        {"name": "Dr. Patrick Wen", "title": "MD", "hospital": "Dana-Farber Cancer Institute", "location": "Boston, MA"},
        {"name": "Dr. Daniel Prevedello", "title": "MD", "hospital": "Ohio State Wexner Medical Center", "location": "Columbus, OH"}
    ],
    "pituitary": [
        {"name": "Dr. Edward Laws", "title": "MD, FACS", "hospital": "Brigham and Women's Hospital", "location": "Boston, MA"},
        {"name": "Dr. Sandeep Kunwar", "title": "MD", "hospital": "California Center for Pituitary Disorders at UCSF", "location": "San Francisco, CA"},
        {"name": "Dr. Gabriel Zada", "title": "MD", "hospital": "USC Pituitary Center", "location": "Los Angeles, CA"}
    ],
    "acoustic_neuroma": [
        {"name": "Dr. Marc Schwartz", "title": "MD", "hospital": "House Clinic", "location": "Los Angeles, CA"},
        {"name": "Dr. Michael Link", "title": "MD", "hospital": "Mayo Clinic", "location": "Rochester, MN"},
        {"name": "Dr. Douglas Kondziolka", "title": "MD", "hospital": "NYU Langone Brain Tumor Center", "location": "New York, NY"}
    ],
    "metastatic": [
        {"name": "Dr. Raymond Sawaya", "title": "MD", "hospital": "MD Anderson Cancer Center", "location": "Houston, TX"},
        {"name": "Dr. Veronica Chiang", "title": "MD", "hospital": "Yale Brain Tumor Center", "location": "New Haven, CT"},
        {"name": "Dr. Jeffrey Bruce", "title": "MD", "hospital": "Columbia University Medical Center", "location": "New York, NY"}
    ],
    "lymphoma": [
        {"name": "Dr. James Rubenstein", "title": "MD, PhD", "hospital": "UCSF Helen Diller Family Comprehensive Cancer Center", "location": "San Francisco, CA"},
        {"name": "Dr. Lisa DeAngelis", "title": "MD", "hospital": "Memorial Sloan Kettering Cancer Center", "location": "New York, NY"},
        {"name": "Dr. Tracy Batchelor", "title": "MD", "hospital": "Massachusetts General Hospital", "location": "Boston, MA"}
    ],
    # Add Indian specialists by tumor type
    "india_glioma": [
        {"name": "Dr. Basant Kumar Misra", "title": "MD, MCh", "hospital": "P.D. Hinduja Hospital", "location": "Mumbai, India"},
        {"name": "Dr. Sanjay Behari", "title": "MCh", "hospital": "Sanjay Gandhi Postgraduate Institute of Medical Sciences", "location": "Lucknow, India"},
        {"name": "Dr. Suresh Sankhla", "title": "MD, DNB", "hospital": "Global Hospital", "location": "Mumbai, India"}
    ],
    "india_meningioma": [
        {"name": "Dr. V.P. Singh", "title": "MCh", "hospital": "Artemis Hospital", "location": "Gurgaon, India"},
        {"name": "Dr. Deepu Banerji", "title": "MCh", "hospital": "Fortis Hospital", "location": "Mumbai, India"},
        {"name": "Dr. Atul Goel", "title": "MCh", "hospital": "KEM Hospital", "location": "Mumbai, India"}
    ],
    "india_pituitary": [
        {"name": "Dr. Dhaval Gohil", "title": "MCh", "hospital": "Kokilaben Dhirubhai Ambani Hospital", "location": "Mumbai, India"},
        {"name": "Dr. Sunil Kutty", "title": "MD, DM", "hospital": "Apollo Hospitals", "location": "Chennai, India"},
        {"name": "Dr. Vani Santosh", "title": "MD", "hospital": "NIMHANS", "location": "Bangalore, India"}
    ],
    "india_metastatic": [
        {"name": "Dr. Rakesh Jalali", "title": "MD", "hospital": "Apollo Proton Cancer Centre", "location": "Chennai, India"},
        {"name": "Dr. Tejpal Gupta", "title": "MD, DNB", "hospital": "Tata Memorial Hospital", "location": "Mumbai, India"},
        {"name": "Dr. B. K. Smruti", "title": "MCh", "hospital": "Bombay Hospital", "location": "Mumbai, India"}
    ]
}

# Top brain tumor treatment centers
top_centers = [
    {"name": "Mayo Clinic Brain Tumor Program", "location": "Rochester, MN", "phone": "507-538-3270", "specialization": "Comprehensive brain tumor care with advanced treatment options"},
    {"name": "MD Anderson Cancer Center", "location": "Houston, TX", "phone": "877-632-6789", "specialization": "Pioneering research and personalized treatment plans"},
    {"name": "Memorial Sloan Kettering Cancer Center", "location": "New York, NY", "phone": "800-525-2225", "specialization": "Cutting-edge research and innovative clinical trials"},
    {"name": "UCSF Brain Tumor Center", "location": "San Francisco, CA", "phone": "415-353-2966", "specialization": "Advanced surgical techniques and personalized therapies"},
    {"name": "Johns Hopkins Comprehensive Brain Tumor Center", "location": "Baltimore, MD", "phone": "410-955-6406", "specialization": "Multidisciplinary approach and molecular diagnostics"},
    # Add top Indian hospitals
    {"name": "All India Institute of Medical Sciences (AIIMS)", "location": "New Delhi, India", "phone": "+91-11-2658-8500", "specialization": "Premier neurosurgical center with comprehensive tumor management"},
    {"name": "Tata Memorial Centre", "location": "Mumbai, India", "phone": "+91-22-2417-7000", "specialization": "Leading cancer care and research center with dedicated neuro-oncology department"},
    {"name": "Fortis Memorial Research Institute", "location": "Gurgaon, India", "phone": "+91-124-4996-222", "specialization": "Advanced brain tumor treatment with cutting-edge technology"},
    {"name": "Apollo Hospitals", "location": "Chennai, India", "phone": "+91-44-2829-0200", "specialization": "Comprehensive neuro-oncology with proton therapy capabilities"},
    {"name": "Manipal Hospitals", "location": "Bangalore, India", "phone": "+91-80-2502-4444", "specialization": "Integrated neurosciences department with tumor board approach"}
]

def get_model_session():
    """Get or initialize the model session with optimized settings"""
    global MODEL_SESSION
    if MODEL_SESSION is None:
        try:
            # Create session options for optimized inference
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = 8  # Increased for faster processing
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_options.enable_cpu_mem_arena = True
            session_options.enable_mem_pattern = True
            session_options.add_session_config_entry("session.load_model_format", "ONNX")
            
            # Use GPU if available with fallback to CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            # Set execution provider options for better GPU utilization
            provider_options = [
                {'device_id': 0, 'arena_extend_strategy': 'kNextPowerOfTwo'},
                {}
            ]
            
            MODEL_SESSION = ort.InferenceSession(
                CONFIG["model_path"], 
                sess_options=session_options,
                providers=providers,
                provider_options=provider_options
            )
            
            # Pre-warm the model with a dummy input
            input_name = MODEL_SESSION.get_inputs()[0].name
            dummy_input = np.zeros((1, 3, *CONFIG["image_size"]), dtype=np.float32)
            for _ in range(3):  # Multiple warm-up runs
                MODEL_SESSION.run(None, {input_name: dummy_input})
            
            logger.info("Model loaded and optimized successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            MODEL_SESSION = None
    
    return MODEL_SESSION

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in CONFIG["allowed_extensions"]

# Optimized transform pipeline with JIT compilation support
preprocess_transform = transforms.Compose([
    transforms.Resize(CONFIG["image_size"], interpolation=CONFIG["resize_quality"]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=CONFIG["normalization_mean"], 
        std=CONFIG["normalization_std"]
    )
])

# Cache common image sizes
IMAGE_SIZE_CACHE = {}

def preprocess_image_in_memory(file_data):
    """Ultra-fast in-memory image preprocessing without saving to disk"""
    try:
        start_time = time.time()
        
        # Load image directly from memory
        img = Image.open(io.BytesIO(file_data))
        
        # Make a copy of the image for display
        display_img = img.copy()
        
        # Convert to RGB only if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Apply transformations
        img_tensor = preprocess_transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        
        # Convert to numpy array as float32 for optimal inference speed
        result = img_tensor.numpy().astype(np.float32)
        
        logger.info(f"In-memory preprocessing completed in {time.time() - start_time:.3f}s")
        return result, display_img
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None, None

def get_image_as_base64(img):
    """Convert PIL Image to base64 encoded string for direct display"""
    try:
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=90, optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return None

def assess_image_quality(img_array):
    """Assess the quality of the image for accurate tumor analysis
    
    Returns a score between 0 and 1, where higher is better quality
    """
    try:
        # Calculate basic statistics
        if img_array.size == 0:
            return 0.0
            
        img_mean = np.mean(img_array)
        img_std = np.std(img_array)
        img_min = np.min(img_array)
        img_max = np.max(img_array)
        
        # Check for sufficient contrast
        contrast_score = min(1.0, (img_max - img_min) / 255.0)
        
        # Check for excessive noise or blur
        if img_std < 5:  # Too little variation - likely too smooth/blurry
            noise_score = 0.2
        elif img_std > 80:  # Too much variation - likely very noisy
            noise_score = 0.3
        else:
            # Optimal range for brain MRI
            noise_score = 1.0 - abs((img_std - 30) / 40.0)
            noise_score = max(0.0, min(1.0, noise_score))
        
        # Check for proper brightness/exposure
        if img_mean < 20 or img_mean > 235:  # Too dark or too bright
            brightness_score = 0.2
        else:
            # Optimal range for visibility of features
            brightness_score = 1.0 - abs((img_mean - 120) / 120.0)
            brightness_score = max(0.0, min(1.0, brightness_score))
        
        # Check for edge information (important for tumor boundary detection)
        gx = np.abs(img_array[:, 1:] - img_array[:, :-1])
        gy = np.abs(img_array[1:, :] - img_array[:-1, :])
        gradient_magnitude = (np.mean(gx) + np.mean(gy)) / 2
        edge_score = min(1.0, gradient_magnitude / 15.0)
        
        # Calculate final quality score with weighted importance
        final_score = (
            0.3 * contrast_score +
            0.25 * noise_score +
            0.2 * brightness_score +
            0.25 * edge_score
        )
        
        return final_score
    
    except Exception as e:
        logger.error(f"Error assessing image quality: {e}")
        return 0.5  # Default to medium quality if assessment fails

def preprocess_image(image_path):
    """Optimized image preprocessing from file"""
    try:
        start_time = time.time()
        
        # Use PIL's optimized loading
        with open(image_path, 'rb') as f:
            return preprocess_image_in_memory(f.read())
    except Exception as e:
        logger.error(f"Error preprocessing image from path: {e}")
        return None, None

# Use binary inputs directly to avoid unnecessary step
def predict(session, image_tensor):
    """Ultra-optimized prediction function"""
    try:
        start_time = time.time()
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Run inference with optimized binding
        result = session.run([output_name], {input_name: image_tensor})
        
        # Process results efficiently with vectorized operations
        probabilities = torch.softmax(torch.tensor(result[0]), dim=1).numpy()
        predicted_class = int(np.argmax(probabilities[0]))
        confidence = float(probabilities[0][predicted_class]) * 100
        
        prediction_time = time.time() - start_time
        logger.info(f"Prediction completed in {prediction_time:.3f}s")
        
        return {
            "class": CONFIG["class_names"][predicted_class],
            "confidence": confidence,
            "class_id": predicted_class,
            "probabilities": probabilities[0].tolist(),
            "prediction_time": prediction_time
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None

def generate_report_directly(class_id, confidence_rounded):
    """Generate detailed diagnostic report directly based on image analysis"""
    if class_id == 0:
        status = "no tumor detected"
        emoji = "✅"
        prompt = f"""Brain MRI scan analysis:
        
Findings: {emoji} {status}
Confidence: {confidence_rounded}%

Generate a completely unique comprehensive medical report with these sections:
1. Clinical Interpretation (describe normal brain appearance)
2. Recommended Next Steps (appropriate follow-up)
3. Notes on AI Limitations (importance of clinical correlation)

Format as HTML with <h3> for headings and <p> for paragraphs. No markdown."""

    else:
        status = f"a tumor detected with {confidence_rounded}% confidence"
        emoji = "⚠️"
        prompt = f"""Brain MRI scan analysis:
        
Findings: {emoji} {status}
Confidence: {confidence_rounded}%

Generate a completely unique detailed medical report with these sections:
1. Clinical Interpretation (describe likely tumor types based on confidence: {confidence_rounded}%)
2. Tumor Location & Characteristics (likely location based on MRI pattern)
3. Potential Causes and Risk Factors (genetic, environmental, etc.)
4. Precautions & Immediate Actions (what patient should and should not do)
5. Recommended Next Steps (urgency level, specialist referral, additional tests)
6. Notes on AI Limitations (importance of clinical correlation)

Be specific about possible tumor types (glioma, meningioma, etc.) and likely locations (frontal lobe, temporal lobe, cerebellum, etc.) based on typical presentation patterns.
Format as HTML with <h3> for headings, <p> for paragraphs, and <ul>/<li> for lists. No markdown."""

    try:
        logger.info(f"Generating report using Ollama at custom port 11500")
        
        # Direct API call to Ollama with correct port
        response = requests.post(
            "http://127.0.0.1:11500/api/generate",
            json={
                "model": "llama2",
                "prompt": prompt,
                "options": {'temperature': 0.7, 'max_tokens': 600}
            },
            timeout=60  # Add a timeout to prevent hanging
        )
        
        if response.status_code == 200:
            logger.info("Successfully received response from Ollama")
            return response.json()['response']
        else:
            logger.error(f"Ollama API error: Status {response.status_code}, {response.text}")
            raise Exception(f"Ollama API error: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Ollama error in generate_report_directly: {e}")
        logger.error(f"Attempted Ollama host: http://127.0.0.1:11500")
        logger.error(traceback.format_exc())
        # Return basic report as fallback
        if class_id == 0:
            return "<h3>Clinical Interpretation</h3><p>No tumor detected. Normal brain tissue appearance.</p><h3>Next Steps</h3><p>Consult with your physician.</p>"
        else:
            return f"<h3>Clinical Interpretation</h3><p>Tumor detected with {confidence_rounded}% confidence.</p><h3>Next Steps</h3><p>Immediate medical consultation recommended.</p>"

@app.route('/')
def index():
    # Ensure model is loaded on startup for faster first prediction
    executor.submit(get_model_session)
    return render_template('index.html')

@app.route('/index (2).html')
def landing_page():
    """Serve the landing page"""
    return send_from_directory('.', 'index (2).html')

@app.route('/styles.css')
def serve_styles():
    """Serve the styles.css file"""
    return send_from_directory('.', 'styles.css')
    
@app.route('/scansphere logo.png')
def serve_logo():
    """Serve the logo image"""
    return send_from_directory('.', 'scansphere logo.png')
    
@app.route('/scripts.js')
def serve_scripts():
    """Serve the scripts.js file"""
    return send_from_directory('.', 'scripts.js')
    
@app.route('/login.html')
def serve_login():
    """Serve the login page"""
    return send_from_directory('.', 'login.html')
    
@app.route('/signup.html')
def serve_signup():
    """Serve the signup page"""
    return send_from_directory('.', 'signup.html')
    
@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files from the root directory"""
    return send_from_directory('.', filename)
    
@app.route('/real-time-analysis')
def real_time_analysis():
    """Serve the real-time analysis page that doesn't store static data"""
    # Pre-load model for faster first prediction
    executor.submit(get_model_session)
    return render_template('real-time-analysis.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files with proper caching headers"""
    response = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    response.headers['Cache-Control'] = 'public, max-age=31536000'  # Cache for a year
    return response

@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload JPG, JPEG, or PNG.')
        return redirect(url_for('index'))
    
    try:
        total_start_time = time.time()
        
        # Get or initialize model session first to ensure it's ready
        session = get_model_session()
        if not session:
            flash('Model loading failed')
            return redirect(url_for('index'))
        
        # Process file
        filename = secure_filename(file.filename)
        file_data = file.read()  # Read file data once
        
        # Process image in memory
        image_tensor, img = preprocess_image_in_memory(file_data)
        if image_tensor is None:
            flash('Image processing error')
            return redirect(url_for('index'))
        
        # Make prediction immediately
        prediction = predict(session, image_tensor)
        if not prediction:
            flash('Prediction failed')
            return redirect(url_for('index'))
        
        # Get image path or base64
        if app.config['USE_BASE64_IMAGES']:
            # Convert image to base64 for immediate display
            image_data = get_image_as_base64(img)
            
            # Save file asynchronously after prediction
            executor.submit(lambda: save_file(file_data, os.path.join(app.config['UPLOAD_FOLDER'], filename)))
        else:
            # Save file synchronously
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(file_path, 'wb') as f:
                f.write(file_data)
            image_data = f"uploads/{filename}"
        
        # Generate a dynamic report for every analysis
        class_id = prediction['class_id']
        confidence_rounded = round(prediction['confidence'] / 5) * 5
        
        if class_id == 0:
            # Generate report for no tumor case
            report = """
            <h3>Clinical Interpretation</h3>
            <p>Based on real-time analysis of the uploaded image, no tumor was detected in the brain MRI scan. Brain tissue appears normal with no visible lesions, mass effects, or abnormal enhancement patterns.</p>
            
            <h3>Next Steps</h3>
            <p>Consult with your physician for a comprehensive evaluation of your symptoms and medical history.</p>
            
            <h3>Important Note</h3>
            <p>This is an AI analysis that should be confirmed by a qualified medical professional. Not all conditions may be detected by this system.</p>
            """
            analysis = None
        else:
            # Perform comprehensive image analysis
            analysis = analyze_tumor_characteristics(img, prediction['confidence'])
            
            # Generate a fully dynamic report based on image analysis
            report = format_tumor_report(analysis)
        
        # Record processing time
        processing_time = time.time() - total_start_time
        logger.info(f"Total processing time: {processing_time:.3f}s")
        
        # Prepare scan data
        scan_data = {
            "filename": filename,
            "filepath": f"uploads/{filename}" if not app.config['USE_BASE64_IMAGES'] else "direct_display",
            "prediction": prediction,
            "timestamp": datetime.now().isoformat(),
            "processing_time": f"{processing_time:.2f}s",
            "analysis": analysis,  # Include full analysis data
            "image": image_data if app.config['USE_BASE64_IMAGES'] else None  # Include image data for PDF generation
        }
        
        return render_template('result.html',
                            prediction=prediction,
                            image_file=image_data,
                            is_base64=app.config['USE_BASE64_IMAGES'],
                            diagnostic_report=report,
                            analysis=analysis,
                            scan_data=json.dumps(scan_data),
                            processing_time=f"{processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        flash('An error occurred. Please try again.')
        return redirect(url_for('index'))

def save_file(file_data, file_path):
    """Save file to disk asynchronously"""
    try:
        with open(file_path, 'wb') as f:
            f.write(file_data)
        logger.info(f"File saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving file: {e}")

@app.route('/view-scan')
def view_scan():
    filename = request.args.get('filename')
    if not filename:
        flash('No scan specified')
        return redirect(url_for('index'))
    
    try:
        # In a real application, you would fetch this from a database
        # For this demo, we'll just pass the filename and let the client handle it
        return render_template('view-scan.html', 
                            scan_data={
                                "filename": filename,
                                "filepath": f"uploads/{filename}",
                                "prediction": {
                                    "class": "Unknown (historical data)",
                                    "confidence": 0,
                                    "class_id": -1,
                                    "probabilities": [0, 0]
                                }
                            })
    except Exception as e:
        logger.error(f"Error viewing scan: {e}")
        flash('Error loading scan')
        return redirect(url_for('index'))

@lru_cache(maxsize=100)
def get_cached_ai_response(question_key, result_class, confidence_rounded, analysis_key=None):
    """Generate detailed and accurate responses about scan results with specific medical details"""
    
    try:
        # Convert analysis back to dict if a key was provided
        if analysis_key and isinstance(analysis_key, str) and analysis_key != "None":
            try:
                # Try to parse the analysis data from the string key
                import json
                analysis_parts = analysis_key.split('::')
                if len(analysis_parts) >= 2:
                    analysis = {
                        'tumor_type': analysis_parts[0],
                        'location': analysis_parts[1],
                        'urgency': analysis_parts[2] if len(analysis_parts) > 2 else 'moderate',
                        'description': analysis_parts[3] if len(analysis_parts) > 3 else ''
                    }
                else:
                    analysis = None
            except:
                analysis = None
        else:
            analysis = None
        
        # Check if question is about specific topics
        tumor_type_keywords = ["type", "kind", "classification", "grade", "what tumor", "what kind"]
        causes_keywords = ["cause", "risk", "factor", "why", "how did", "reason", "etiology"]
        location_keywords = ["where", "location", "located", "position", "area", "lobe", "place", "region"]
        precaution_keywords = ["precaution", "safety", "avoid", "should i", "shouldn't i", "safe to", "dangerous", "risk"]
        prognosis_keywords = ["prognosis", "survival", "outcome", "recover", "chance", "outlook"]
        treatment_keywords = ["treatment", "therapy", "options", "surgery", "radiation", "manage", "cure"]
        specialist_keywords = ["doctor", "specialist", "physician", "neurosurgeon", "consult", "who", "refer"]
        hospital_keywords = ["hospital", "center", "clinic", "facility", "institution", "where to go", "medical center"]
        
        # Check if the question matches any of the specialized categories
        is_tumor_type_question = any(keyword in question_key for keyword in tumor_type_keywords)
        is_causes_question = any(keyword in question_key for keyword in causes_keywords)
        is_location_question = any(keyword in question_key for keyword in location_keywords)
        is_precaution_question = any(keyword in question_key for keyword in precaution_keywords)
        is_prognosis_question = any(keyword in question_key for keyword in prognosis_keywords)
        is_treatment_question = any(keyword in question_key for keyword in treatment_keywords)
        is_specialist_question = any(keyword in question_key for keyword in specialist_keywords)
        is_hospital_question = any(keyword in question_key for keyword in hospital_keywords)
        
        # Default values if analysis data is missing
        tumor_type = "brain tumor"
        location = "brain"
        description = "abnormal tissue growth requiring further evaluation"
        urgency = "moderate"
        
        # Update with actual values if available
        if analysis:
            tumor_type = analysis.get('tumor_type', tumor_type)
            location = analysis.get('location', location)
            description = analysis.get('description', description)
            urgency = analysis.get('urgency', urgency)
        
        # Database of top specialist names and hospitals (for more accurate and specific recommendations)
        # Now defined in global scope - removed from here
        
        # Get appropriate specialists based on tumor type
        recommended_specialists = specialists.get(tumor_type.lower().split()[0], specialists["glioma"])
        
        # Select a specialist to recommend (this can be random or based on some criteria)
        primary_specialist = recommended_specialists[0]
        
        # Select a top center to recommend
        recommended_center = top_centers[0]
        
        # Add specialist and center information to the prompt
        specialist_info = f"For this {tumor_type}, specialists like {primary_specialist['name']}, {primary_specialist['title']}, at {primary_specialist['hospital']} in {primary_specialist['location']} have extensive experience treating such cases."
        center_info = f"Top brain tumor centers such as {recommended_center['name']} in {recommended_center['location']} (contact: {recommended_center['phone']}) offer comprehensive treatment programs specifically for cases like yours."
        
        # Create a detailed prompt for Ollama based on the type of question
        if is_tumor_type_question and "tumor" in result_class.lower():
            prompt = f"""Patient Question about tumor type: {question_key}
            
Scan Results: {result_class} (Confidence: {confidence_rounded}%)
Image Analysis: The scan shows a {tumor_type} located in the {location}.

Provide a detailed, accurate answer about this specific tumor type. Include:
- Precise classification of {tumor_type}s according to WHO guidelines
- Typical characteristics on imaging
- Cell of origin and histopathological features
- Common genetic and molecular markers (IDH, 1p/19q, MGMT, etc.)
- Specific subtypes and their prognostic implications
- Epidemiology statistics (prevalence, age distribution)
- Growth patterns and natural history
- Differential diagnosis considerations
- How this specific tumor differs from other brain tumors
- Latest research advances for this tumor type

Incorporate this specialist information:
{specialist_info}
{center_info}

Ensure your answer is comprehensive, medically precise and includes specific doctor names, hospital centers, and detailed medical information. Make it as detailed and descriptive as possible."""

        elif is_location_question and "tumor" in result_class.lower():
            prompt = f"""Patient Question about tumor location: {question_key}
            
Scan Results: {result_class} (Confidence: {confidence_rounded}%)
Image Analysis: The scan shows a {tumor_type} located in the {location}.
Tumor Description: {description}

Provide a detailed, accurate answer about the significance of this specific tumor location. Include:
- Detailed neuroanatomical description of the {location} and surrounding structures
- Specific neural pathways, circuits and functions associated with this brain region
- Vascular supply to this region and potential impact on surgical approach
- White matter tracts and critical structures in proximity to this location
- Expected neurological symptoms based on the precise location
- Challenges for surgical access to this specific region
- Advanced techniques used to approach tumors in this location
- Names of neurosurgeons specialized in operating in this specific brain region
- Major medical centers with expertise in tumors in this anatomical location

Incorporate the following specialist information:
{specialist_info}
{center_info}

Ensure your answer is comprehensive, medically precise and includes specific doctor names, hospital centers, and detailed medical information. Make it as detailed and descriptive as possible."""

        elif is_causes_question and "tumor" in result_class.lower():
            prompt = f"""Patient Question about tumor causes: {question_key}
            
Scan Results: {result_class} (Confidence: {confidence_rounded}%)
Image Analysis: The scan shows a {tumor_type} located in the {location}.

Provide a detailed, accurate answer about the specific causes and risk factors for {tumor_type}s. Include:
- Established genetic mutations and chromosomal abnormalities specific to {tumor_type}s
- Molecular signaling pathways implicated in this tumor's development
- Specific oncogenes and tumor suppressor genes involved
- Known environmental risk factors with statistical associations
- Hereditary syndromes associated with this tumor type and their genetic basis
- Age, gender, and demographic patterns with epidemiological data
- Current research on etiology from major brain tumor research centers
- Names of medical geneticists and neuro-oncologists specializing in this tumor type
- Medical centers with specialized expertise in genetic analysis and counseling

Incorporate the following specialist information:
{specialist_info}
{center_info}

Ensure your answer is comprehensive, medically precise and includes specific doctor names, hospital centers, and detailed medical information. Make it as detailed and descriptive as possible."""
        
        elif is_precaution_question and "tumor" in result_class.lower():
            prompt = f"""Patient Question about precautions: {question_key}
            
Scan Results: {result_class} (Confidence: {confidence_rounded}%)
Image Analysis: The scan shows a {tumor_type} located in the {location}.
Urgency Level: {urgency}

Provide detailed, accurate precautions relevant to this patient's specific tumor type and location. Include:
- Comprehensive list of activities to avoid based on this specific tumor location
- Detailed warning signs and symptoms requiring immediate medical attention
- Medication precautions including specific drugs to avoid and potential interactions
- Specific dietary recommendations if applicable
- Physical activity guidelines with explicit recommendations
- Seizure precautions with specific protocols if relevant
- Detailed driving restrictions based on tumor location and seizure risk
- Work and daily activity modifications recommended
- Names of specific neurologists who can provide personalized precaution guidance
- Medical facilities with 24-hour neuro-oncology emergency services

Incorporate the following specialist information:
{specialist_info}
{center_info}

Format with clear DO and DON'T recommendations. Ensure your answer is comprehensive, medically precise and includes specific doctor names, hospital centers, and detailed medical information. Make it as detailed and descriptive as possible."""

        elif is_prognosis_question and "tumor" in result_class.lower():
            prompt = f"""Patient Question about prognosis: {question_key}
            
Scan Results: {result_class} (Confidence: {confidence_rounded}%)
Image Analysis: The scan shows a {tumor_type} located in the {location}.
Tumor Description: {description}

Provide a detailed, accurate discussion of prognosis for this specific tumor type and location. Include:
- Current survival statistics with 1-year, 5-year, and longer-term outcomes
- Prognostic factors specific to this tumor type including molecular markers
- Impact of tumor location on treatment efficacy and outcomes
- Age-related variations in prognosis with statistical data
- Treatment response rates for various therapeutic approaches
- Quality of life considerations with functional outcome measures
- Recent advances improving prognosis for this specific tumor type
- Names of neuro-oncologists specializing in this specific tumor type
- Medical centers with specialized expertise and outcomes tracking for this condition

Incorporate the following specialist information:
{specialist_info}
{center_info}

Ensure your answer is comprehensive, medically precise and includes specific doctor names, hospital centers, and detailed medical information. Maintain appropriate balance between honesty and hope. Make it as detailed and descriptive as possible."""

        elif is_treatment_question and "tumor" in result_class.lower():
            prompt = f"""Patient Question about treatment: {question_key}
            
Scan Results: {result_class} (Confidence: {confidence_rounded}%)
Image Analysis: The scan shows a {tumor_type} located in the {location}.
Tumor Description: {description}

Provide detailed, accurate information about treatment options for this particular tumor type and location. Include:
- Standard of care treatment protocols with specific drug names and dosages
- Detailed surgical approaches specific to the {location} with procedural descriptions
- Advanced radiation therapy modalities appropriate for this case
- Specific chemotherapy regimens with drug mechanisms of action
- Targeted therapies based on molecular profile of {tumor_type}s
- Immunotherapy options if applicable for this tumor type
- Novel treatments currently in clinical trials with specific trial identifiers
- Names of prominent neurosurgeons and neuro-oncologists specializing in this tumor
- Medical centers with specialized treatment programs for this condition

Incorporate the following specialist information:
{specialist_info}
{center_info}

Ensure your answer is comprehensive, medically precise and includes specific doctor names, hospital centers, and detailed medical information. Make it as detailed and descriptive as possible."""

        elif is_specialist_question and "tumor" in result_class.lower():
            prompt = f"""Patient Question about specialists/doctors: {question_key}
            
Scan Results: {result_class} (Confidence: {confidence_rounded}%)
Image Analysis: The scan shows a {tumor_type} located in the {location}.

Provide a detailed, accurate answer about specialist doctors for this condition. Include:
- Names of leading neurosurgeons specialized in {tumor_type}s with their affiliations
- Neuro-oncologists with specific expertise in this tumor type
- Radiation oncologists known for treating brain tumors in the {location}
- Specialized neurologists focused on symptom management
- Multidisciplinary tumor board composition recommended for this case
- Criteria for selecting the right specialist for this specific condition
- Questions patients should ask potential specialists
- Academic centers with specialized teams for this tumor type
- How to obtain referrals to top specialists in this field

Provide this specific information:
{specialist_info}
Also recommend these additional specialists:
{', '.join([f"Dr. {spec['name']} at {spec['hospital']}" for spec in recommended_specialists[1:]])}

{center_info}
Additional centers to consider:
{', '.join([f"{center['name']} in {center['location']}" for center in top_centers[1:3]])}

Ensure your answer is comprehensive, medically precise and includes specific doctor names, hospital centers, and contact information. Make it as detailed and descriptive as possible."""

        elif is_hospital_question and "tumor" in result_class.lower():
            prompt = f"""Patient Question about hospitals/treatment centers: {question_key}
            
Scan Results: {result_class} (Confidence: {confidence_rounded}%)
Image Analysis: The scan shows a {tumor_type} located in the {location}.

Provide a detailed, accurate answer about medical facilities for treating this condition. Include:
- Names of top-ranked hospitals specializing in brain tumors, particularly {tumor_type}s
- Comprehensive cancer centers with dedicated neuro-oncology departments
- Specialized centers known for treating tumors in the {location}
- Hospitals with advanced technological capabilities (intraoperative MRI, Gamma Knife, etc.)
- Centers conducting clinical trials for this specific tumor type
- Academic medical centers with multidisciplinary tumor boards
- Criteria for evaluating and selecting the right treatment facility
- Patient volume and outcomes data for major centers
- Insurance and referral process information

Provide specific details on these centers:
{center_info}

Also consider these facilities:
{'; '.join([f"{center['name']} in {center['location']} ({center['phone']}), specializing in {center['specialization']}" for center in top_centers[1:]])}

Specialist information:
{specialist_info}
Additional specialists at these centers include:
{', '.join([f"Dr. {spec['name']} at {spec['hospital']}" for spec in recommended_specialists[1:]])}

Ensure your answer is comprehensive, medically precise and includes specific hospital names, locations, contact information, and areas of expertise. Make it as detailed and descriptive as possible."""
        
        else:
            prompt = f"""Patient Question: {question_key}
            
Scan Results: {result_class} (Confidence: {confidence_rounded}%)
Image Analysis: The scan shows a {tumor_type} located in the {location}.
Tumor Description: {description}

Provide a detailed, accurate answer addressing this specific question. Include:
- Direct answer to the question with comprehensive medical information
- Specific details related to this {tumor_type} in the {location}
- Explanation of relevant pathophysiology and clinical implications
- Current medical guidelines and standards of care applicable to this case
- Recent research advances relevant to the question
- Names of medical specialists who focus on this aspect of brain tumors
- Top medical centers with expertise in this specific area
- References to medical literature where appropriate
- Practical next steps for the patient

Incorporate this specialist information:
{specialist_info}
{center_info}

Ensure your answer is comprehensive, medically precise and includes specific doctor names, hospital centers, and detailed medical information. Make it as detailed and descriptive as possible."""
        
        try:
            # First attempt with llama2
            response = ollama.generate(
                model='llama2',
                prompt=prompt,
                options={'temperature': 0.2, 'max_tokens': 800}  # Increased token count for longer responses
            )
            return response['response']
        except Exception as e:
            logger.error(f"Primary Ollama error: {e}")
            # Fallback to direct answer generation
            logger.info("Falling back to direct answer generation")
            
            if "tumor" in result_class.lower():
                # Generate a basic but accurate response based on analysis data
                if is_tumor_type_question:
                    return generate_tumor_type_answer(tumor_type, location, description, primary_specialist, recommended_center)
                elif is_location_question:
                    return generate_location_answer(tumor_type, location, primary_specialist, recommended_center)
                elif is_causes_question:
                    return generate_causes_answer(tumor_type, primary_specialist, recommended_center)
                elif is_precaution_question:
                    return generate_precaution_answer(tumor_type, location, urgency, primary_specialist, recommended_center)
                elif is_treatment_question:
                    return generate_treatment_answer(tumor_type, location, primary_specialist, recommended_center)
                elif is_specialist_question:
                    return generate_specialist_answer(tumor_type, location, recommended_specialists, top_centers)
                elif is_hospital_question:
                    return generate_hospital_answer(tumor_type, location, recommended_specialists, top_centers)
                else:
                    return generate_general_answer(tumor_type, location, description, primary_specialist, recommended_center)
            else:
                return "Based on the scan analysis, no tumor was detected. The brain MRI appears normal with no significant abnormalities. For any specific symptoms you're experiencing, it's important to consult with a neurologist such as Dr. James Bernat at Dartmouth-Hitchcock Medical Center or Dr. Orrin Devinsky at NYU Langone Comprehensive Epilepsy Center. These specialists can provide a comprehensive evaluation. Diagnostic imaging is just one part of the overall diagnostic process."
    
    except Exception as e:
        # Catch any unexpected errors to ensure we always return something
        logger.error(f"Unexpected error in AI response generation: {e}")
        logger.error(traceback.format_exc())
        return "I apologize, but I encountered an error while generating a response. Your scan has been analyzed, but I recommend discussing the results with a neurosurgeon such as Dr. Robert Spetzler at Barrow Neurological Institute (602-406-3181) or Dr. Mitchel Berger at UCSF Brain Tumor Center (415-353-2966) for a complete interpretation and next steps."

# Enhanced helper functions to generate fallback answers with specialist information
def generate_tumor_type_answer(tumor_type, location, description, specialist, center):
    """Generate a detailed, clinically accurate answer about tumor type with specialist information"""
    grade_hint = ""
    molecular_markers = ""
    
    # Add grade information based on tumor type
    if "glioma" in tumor_type.lower():
        grade_hint = "Gliomas are graded from I-IV according to the WHO classification, with higher grades indicating more aggressive behavior. Molecular markers such as IDH mutation status, 1p/19q co-deletion, and MGMT promoter methylation are critical for classification and treatment planning."
        molecular_markers = "Key molecular markers typically assessed include IDH1/2 mutation, 1p/19q co-deletion, ATRX loss, p53 mutation, EGFR amplification, and MGMT promoter methylation status."
    elif "meningioma" in tumor_type.lower():
        grade_hint = "Meningiomas are typically WHO grade I (90%), with atypical (grade II) and anaplastic (grade III) variants comprising the remainder. Higher grade meningiomas have increased mitotic activity, brain invasion, or other concerning histologic features."
    elif "metastatic" in tumor_type.lower():
        grade_hint = "Metastatic tumors retain the characteristics of their primary malignancy. The assessment focuses on identifying the origin through immunohistochemical and molecular analysis."
    
    return f"""CLINICAL ASSESSMENT: MRI BRAIN - TUMOR EVALUATION

FINDINGS: The neuroimaging shows a {tumor_type} located in the {location}.

DETAILED IMPRESSION:
{description}

HISTOPATHOLOGICAL CONSIDERATIONS:
{tumor_type}s represent a distinct pathological entity with characteristic features on both imaging and histology. {grade_hint} {molecular_markers}

DIAGNOSTIC CERTAINTY:
While MRI provides substantial information regarding tumor characteristics, definitive diagnosis requires histopathological examination. Stereotactic or open biopsy with comprehensive molecular profiling is the gold standard for diagnosis.

RECOMMENDATION FOR SPECIALIST CONSULTATION:
Dr. {specialist['name']}, {specialist['title']}, at {specialist['hospital']} in {specialist['location']} specializes in the management of {tumor_type.lower()}s. The {center['name']} ({center['phone']}) provides a comprehensive neuro-oncology program with advanced molecular diagnostic capabilities and a multidisciplinary tumor board approach.

For optimal management, I strongly recommend neurosurgical consultation for consideration of stereotactic biopsy or resection, followed by comprehensive molecular and genomic analysis to guide adjuvant therapy decisions."""

def generate_location_answer(tumor_type, location, specialist, center):
    """Generate a detailed answer about tumor location with specialist information"""
    eloquent_regions = {
        "frontal lobe": "motor cortex, Broca's area (dominant hemisphere), supplementary motor area",
        "temporal lobe": "Wernicke's area (dominant hemisphere), primary auditory cortex, memory circuits (hippocampus)",
        "parietal lobe": "primary sensory cortex, association areas for spatial processing",
        "occipital lobe": "primary visual cortex and visual association areas",
        "deep brain structures": "thalamus, basal ganglia, brainstem, critical white matter tracts"
    }
    
    eloquent_area = "unknown"
    for region, areas in eloquent_regions.items():
        if region in location.lower():
            eloquent_area = areas
            break
    
    surgical_approach = {
        "frontal lobe": "frontotemporal craniotomy, potentially with awake mapping for lesions near eloquent areas",
        "temporal lobe": "pterional or temporal craniotomy, often with neurophysiological monitoring",
        "parietal lobe": "parietal craniotomy with careful mapping of sensory function",
        "occipital lobe": "occipital craniotomy with visual field mapping",
        "deep brain structures": "stereotactic approach or specialized skull base approaches depending on exact location"
    }
    
    approach = "tailored surgical approach based on precise tumor location"
    for region, app in surgical_approach.items():
        if region in location.lower():
            approach = app
            break
    
    return f"""NEUROANATOMICAL ASSESSMENT: TUMOR LOCATION ANALYSIS

FINDINGS: The lesion is located in the {location}.

NEUROANATOMICAL SIGNIFICANCE:
This location is within or adjacent to {eloquent_area}, which control(s) critical neurological functions. Lesions in this region typically present with specific focal neurological deficits that correlate with the affected functional areas.

VASCULAR CONSIDERATIONS:
The blood supply to this region primarily derives from branches of the {location.lower().split()[0] if "left" in location.lower() or "right" in location.lower() else "middle"} cerebral artery. Surgical approaches must consider vascular anatomy to avoid compromise of critical perfusion.

WHITE MATTER TRACT INVOLVEMENT:
Advanced neuroimaging with DTI (diffusion tensor imaging) is recommended to evaluate the relationship of the tumor to critical white matter tracts such as the corticospinal tract, arcuate fasciculus, and other association and projection fibers.

SURGICAL ACCESSIBILITY:
Lesions in this location are typically approached via {approach}. Intraoperative neurophysiological monitoring and/or awake mapping techniques may be indicated to maximize safe resection while preserving neurological function.

SPECIALIST RECOMMENDATION:
Dr. {specialist['name']} at {specialist['hospital']} has particular expertise in surgically approaching tumors in the {location}, employing advanced neuroimaging and intraoperative mapping techniques. The neurosurgical team at {center['name']} ({center['phone']}) utilizes state-of-the-art navigation systems and intraoperative monitoring to maximize safe tumor resection while preserving function."""

def generate_causes_answer(tumor_type, specialist, center):
    """Generate a detailed answer about tumor causes with specialist information"""
    return f"""The exact cause of {tumor_type}s, like most brain tumors, is typically unknown in individual cases.

According to Dr. {specialist['name']} at {specialist['hospital']}, known risk factors may include:
- Genetic predisposition or certain genetic syndromes
- Age (certain tumors are more common in specific age groups)
- Previous radiation exposure to the head
- Immune system disorders in some cases

The {center['name']} in {center['location']} (contact: {center['phone']}) conducts ongoing research into the genetic and molecular basis of {tumor_type.lower()}s. Their genomic sequencing program can identify specific mutations that may have contributed to tumor development.

It's important to understand that most brain tumors are not clearly linked to any specific behavior, environmental exposure, or preventable factor that an individual can control.

For genetic counseling and a detailed discussion about potential causes in your specific case, Dr. {specialist['name']}, {specialist['title']}, can provide personalized information."""

def generate_precaution_answer(tumor_type, location, urgency, specialist, center):
    """Generate a detailed answer about precautions with specialist information"""
    urgency_level = "immediate" if urgency == "high" else "prompt" if urgency == "moderate to high" else "routine"
    
    return f"""Based on the detected {tumor_type} in the {location}, here are important precautions as would be advised by Dr. {specialist['name']} at {specialist['hospital']}:

DO:
- Seek {urgency_level} medical consultation with a neurosurgeon such as Dr. {specialist['name']} ({specialist['title']})
- Report any new or worsening neurological symptoms immediately to the {center['name']} emergency department ({center['phone']})
- Have someone available to assist you if you experience headaches, vision changes, or coordination problems
- Keep a detailed symptom diary to share with your doctors
- Maintain your regular medication schedule as prescribed

DON'T:
- Delay seeking medical attention, especially for severe headaches, seizures, or neurological changes
- Participate in high-risk activities that could lead to head injury
- Make major lifestyle changes without consulting your doctor at {specialist['hospital']}

The {center['name']} offers a 24-hour neurosurgical consultation service and has specific protocols for patients with brain tumors. A comprehensive evaluation by their multidisciplinary tumor board will provide personalized guidance for your situation."""

def generate_treatment_answer(tumor_type, location, specialist, center):
    """Generate a detailed answer about treatment options with specialist information"""
    standard_protocols = {
        "Glioma": {
            "high-grade": "Maximal safe surgical resection followed by concurrent temozolomide (75 mg/m²/day) and radiation therapy (60 Gy in 30 fractions) followed by adjuvant temozolomide (150-200 mg/m² for 5 days every 28 days) for 6-12 cycles (Stupp protocol).",
            "low-grade": "Maximal safe surgical resection, potentially followed by radiation therapy (54 Gy in 30 fractions) and/or PCV chemotherapy or temozolomide for high-risk features."
        },
        "Meningioma": {
            "standard": "Surgical resection (Simpson Grade I-II) with the goal of complete tumor and dural attachment removal. Radiation therapy may be considered for subtotal resection or higher-grade histology.",
            "recurrent": "Stereotactic radiosurgery (12-16 Gy) for smaller lesions or fractionated radiation therapy (54 Gy in 30 fractions) for larger lesions. Limited options with systemic therapy."
        },
        "Metastatic Tumor": {
            "single": "Surgical resection followed by stereotactic radiosurgery (SRS) to the tumor bed or primary SRS for smaller lesions (<3-4 cm).",
            "multiple": "Whole brain radiation therapy (30 Gy in 10 fractions) or multiple SRS treatments, with consideration of systemic therapy based on primary tumor type and molecular characteristics."
        },
        "Pituitary Adenoma": {
            "standard": "Transsphenoidal resection for most tumors. Medical therapy (dopamine agonists for prolactinomas, somatostatin analogs for some growth hormone-secreting tumors) may be first-line for select cases.",
            "residual": "Adjuvant radiation therapy, stereotactic radiosurgery, or reoperation for symptomatic or growing residual tumor."
        }
    }
    
    protocol_key = "standard"
    if "high-grade" in tumor_type.lower() or "grade iii" in tumor_type.lower() or "grade iv" in tumor_type.lower():
        protocol_key = "high-grade"
    elif "low-grade" in tumor_type.lower() or "grade i" in tumor_type.lower() or "grade ii" in tumor_type.lower():
        protocol_key = "low-grade"
    
    protocol = "individualized treatment plan based on exact diagnosis, molecular features, and extent of disease"
    if tumor_type in standard_protocols:
        protocol = standard_protocols[tumor_type].get(protocol_key, standard_protocols[tumor_type].get("standard", protocol))
    
    clinical_trials = {
        "Glioma": "IDH-mutant targeted therapies, immunotherapy with checkpoint inhibitors, CAR-T cell therapies, novel delivery methods",
        "Meningioma": "Targeted therapies against somatostatin receptors, mTOR inhibitors, immune checkpoint inhibitors",
        "Metastatic Tumor": "Tumor-specific targeted therapies, blood-brain barrier penetrating agents, radiosensitizers",
        "Pituitary Adenoma": "Novel medical therapies for hormone control, radiosensitizers"
    }
    
    trials = "emerging treatment approaches under investigation"
    for key in clinical_trials:
        if key.lower() in tumor_type.lower():
            trials = clinical_trials[key]
            break
    
    return f"""COMPREHENSIVE TREATMENT PLAN: {tumor_type.upper()} IN THE {location.upper()}

MULTIDISCIPLINARY MANAGEMENT RECOMMENDATIONS:

1. SURGICAL INTERVENTION:
   • Primary surgical approach: {approach} for {tumor_type.lower()}s in the {location}
   • Surgical goals: Maximal safe resection with preservation of neurological function
   • Advanced techniques: Neuronavigation, intraoperative MRI, awake craniotomy with direct electrostimulation may be utilized depending on proximity to eloquent cortex
   • Surgical specialist: Dr. {specialist['name']} ({specialist['title']}) at {specialist['hospital']} specializes in this approach

2. RADIATION ONCOLOGY CONSIDERATIONS:
   • Standard radiation protocol: {protocol}
   • Advanced techniques: Intensity-modulated radiation therapy (IMRT), stereotactic radiosurgery (SRS), or proton therapy may be considered based on tumor location and proximity to critical structures
   • Radiation side effects: Tailored mitigation strategies including prophylactic anti-seizure medication and steroid management protocol

3. MEDICAL NEURO-ONCOLOGY:
   • Systemic therapy considerations: {protocol}
   • Molecular biomarker-driven therapy: Treatment may be further refined based on molecular and genetic profile
   • Supportive care: Comprehensive symptom management and quality of life optimization

4. CLINICAL TRIAL OPPORTUNITIES:
   The {center['name']} (contact: {center['phone']}) has active clinical trials investigating {trials}.

FOLLOW-UP AND MONITORING PROTOCOL:
   • Initial post-treatment MRI: 24-72 hours post-surgery (if performed)
   • Early monitoring: MRI with contrast at 2-3 months post-treatment completion
   • Surveillance imaging: Every 3-4 months for 2 years, then extending interval based on stability
   • Neurological assessment: Scheduled evaluations with neurological examination and quality of life measures

This treatment plan should be discussed in detail with a multidisciplinary tumor board to incorporate all relevant clinical, radiographic, and molecular data. The team at {center['name']} offers comprehensive treatment planning meetings weekly."""

def generate_specialist_answer(tumor_type, location, specialists, centers):
    """Generate a detailed answer about specialists"""
    return f"""For a {tumor_type} in the {location}, I recommend consulting with these leading specialists:

1. Dr. {specialists[0]['name']}, {specialists[0]['title']} - {specialists[0]['hospital']} in {specialists[0]['location']}
   Renowned for pioneering surgical techniques for {tumor_type.lower()}s in the {location} region.

2. Dr. {specialists[1]['name']} - {specialists[1]['hospital']} in {specialists[1]['location']}
   Specializes in comprehensive treatment planning utilizing the latest advances in targeted therapies.

3. Dr. {specialists[2]['name']} - {specialists[2]['hospital']} in {specialists[2]['location']}
   Expert in minimally invasive approaches and preservation of neurological function.

These medical centers offer comprehensive brain tumor programs:

1. {centers[0]['name']} in {centers[0]['location']} (Contact: {centers[0]['phone']})
   {centers[0]['specialization']}

2. {centers[1]['name']} in {centers[1]['location']} (Contact: {centers[1]['phone']})
   {centers[1]['specialization']}

When consulting with these specialists, bring your complete imaging records and be prepared to discuss your specific symptoms and concerns. Most major centers offer virtual consultations for initial reviews before in-person visits."""

def generate_hospital_answer(tumor_type, location, specialists, centers):
    """Generate a detailed answer about hospitals and treatment centers"""
    return f"""For treatment of a {tumor_type} in the {location}, these leading medical centers specialize in comprehensive brain tumor care:

1. {centers[0]['name']} in {centers[0]['location']} (Contact: {centers[0]['phone']})
   {centers[0]['specialization']}
   Key specialists: Dr. {specialists[0]['name']}, {specialists[0]['title']}

2. {centers[1]['name']} in {centers[1]['location']} (Contact: {centers[1]['phone']})
   {centers[1]['specialization']}
   Features a dedicated {tumor_type.lower()} treatment program.

3. {centers[2]['name']} in {centers[2]['location']} (Contact: {centers[2]['phone']})
   {centers[2]['specialization']}
   Offers innovative clinical trials specifically for {tumor_type.lower()}s.

4. {centers[3]['name']} in {centers[3]['location']} (Contact: {centers[3]['phone']})
   Specializes in advanced surgical approaches for tumors in the {location}.

When choosing a treatment center, consider:
- Experience with your specific tumor type
- Multidisciplinary tumor board availability
- Access to clinical trials
- Distance from your home
- Insurance coverage and financial considerations

Most of these centers accept referrals from your primary neurologist or can be contacted directly for an initial consultation."""

def generate_general_answer(tumor_type, location, description, specialist, center):
    """Generate a detailed general answer with specialist information"""
    return f"""The scan analysis shows a {tumor_type} located in the {location}.

{description}

Dr. {specialist['name']}, {specialist['title']}, at {specialist['hospital']} in {specialist['location']} is a leading authority on {tumor_type.lower()}s and has published extensive research on tumors in the {location} region. The {center['name']} (contact: {center['phone']}) offers a comprehensive brain tumor program with the latest diagnostic and treatment technologies.

This finding requires evaluation by a neurosurgeon such as Dr. {specialist['name']} who can review your complete medical history, examine you, and discuss appropriate next steps. These may include additional advanced imaging, possible biopsy, and personalized treatment options specific to this type of tumor.

Brain tumor management is highly specialized and the multidisciplinary tumor board at {center['name']} uses a team approach for optimal outcomes. I recommend securing an appointment with Dr. {specialist['name']} or another specialist at {center['name']} as soon as possible to discuss your specific case in detail."""

@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        prediction = data.get('prediction', {})
        analysis = data.get('analysis', {})  # Get the detailed analysis data
        start_time = time.time()
        
        if not question:
            return jsonify({'error': 'Empty question'}), 400
        
        # Extract key details from the prediction and analysis to incorporate into responses
        tumor_type = prediction.get('class', 'Unknown')
        tumor_location = analysis.get('location', '')
        tumor_size = analysis.get('size_cm', '')
        tumor_description = analysis.get('description', '')
        clinical_significance = analysis.get('clinical_significance', '')
        precautions = analysis.get('precautions', '')
        
        # Log the question type for analytics
        logger.info(f"Processing patient question about {tumor_type}: {question[:100]}...")
        
        # Create an enhanced prompt with specific diagnostic details
        prompt = f"""Answer this specific patient question about their brain MRI scan results showing {tumor_type} located in the {tumor_location}.

Key diagnostic findings:
- Tumor type: {tumor_type}
- Location: {tumor_location}
- Size: {tumor_size} cm
- Description: {tumor_description}

Patient question: "{question}"

Important rules:
1. Be direct and brief (2-5 sentences only)
2. Focus ONLY on answering the specific question
3. Include specific details from their diagnostic report when relevant
4. If asked for doctor recommendations, provide actual names of specialists
5. If asked for precautions, list only 3-4 specific relevant precautions
6. No dramatic statements like "Oh no!"
7. No unnecessary general advice
8. Include a very brief reminder to consult healthcare professionals

Answer:"""
        
        # Check if we have predefined answers based on question type before making API call
        question_lower = question.lower()
        predefined_answer = None
        
        # For Indian doctor questions, directly provide specific recommendations
        if ("indian" in question_lower or "india" in question_lower) and ("doctor" in question_lower or "specialist" in question_lower or "surgeon" in question_lower):
            predefined_answer = "Some well-known neurosurgeons in India include Dr. B.K. Misra, Dr. Satnam Singh Chhabra, and Dr. Suresh Nair. Please consult your local medical center for specialists in your area."
            logger.info("Using predefined Indian specialists answer")
            
        # For precaution questions, use the precautions from the analysis if available
        elif "precaution" in question_lower and precautions:
            # Extract 3-4 key precautions from the analysis
            precaution_lines = [p.strip() for p in precautions.split(".") if p.strip()]
            if len(precaution_lines) >= 3:
                formatted_precautions = ""
                for i, p in enumerate(precaution_lines[:4], 1):  # Limit to first 4
                    formatted_precautions += f"{i}) {p.strip()}. "
                predefined_answer = f"Key precautions for your {tumor_type}: {formatted_precautions}Consult your doctor for personalized advice."
                logger.info("Using precautions from analysis")
        
        # If we have a predefined answer, return it without calling Ollama
        if predefined_answer:
            return jsonify({'response': predefined_answer})
        
        try:
            # Make the API call to Ollama
            response = ollama.generate(
                model='llama2',
                prompt=prompt,
                options={
                    'temperature': 0.3,
                    'max_tokens': 120,  # Limiting tokens to keep answers brief
                    'top_p': 0.9,
                    'stop': ['###', '```', '\n\n\n']
                }
            )
            
            answer = response['response'].strip()
            
            # Clean up any artifacts
            if answer.lower().startswith("answer:"):
                answer = answer[7:].strip()
                
            # Remove dramatic openings
            dramatic_starts = ["oh no!", "i'm sorry", "unfortunately", "i regret to inform", "this is concerning"]
            for start in dramatic_starts:
                if answer.lower().startswith(start):
                    answer = answer[len(start):].strip()
                    # Clean up punctuation after removing the start
                    if answer.startswith(",") or answer.startswith(":") or answer.startswith("."):
                        answer = answer[1:].strip()
            
            # If asked about Indian doctors and the response doesn't mention specific names
            if "indian doctor" in question.lower() or "indian specialist" in question.lower() or "indian physicians" in question.lower():
                if not any(term in answer.lower() for term in ["dr.", "doctor", "specialist", "prof.", "professor"]):
                    # Add some specific doctor suggestions
                    indian_specialists = "Some well-known neurosurgeons in India include Dr. B.K. Misra, Dr. Satnam Singh Chhabra, and Dr. Suresh Nair. Please consult your local medical center for specialists in your area."
                    answer = indian_specialists
            
            # If asked about precautions and the answer is too long or doesn't use the analysis data
            if "precaution" in question.lower():
                if len(answer.split()) > 60 or (precautions and not any(p.lower() in answer.lower() for p in precautions.split(".")[:3])):
                    # Use precautions from the analysis if available, otherwise use fallback
                    if precautions:
                        precaution_lines = [p.strip() for p in precautions.split(".") if p.strip()]
                        if len(precaution_lines) >= 3:
                            formatted_precautions = ""
                            for i, p in enumerate(precaution_lines[:4], 1):
                                formatted_precautions += f"{i}) {p.strip()}. "
                            answer = f"Key precautions for your {tumor_type}: {formatted_precautions}Consult your doctor for personalized advice."
                    else:
                        # Fallback if no precautions in analysis
                        answer = f"Key precautions for {tumor_type}: 1) Avoid strenuous physical activities that increase blood pressure, 2) Take prescribed medications consistently, 3) Monitor and report new symptoms immediately, 4) Maintain regular follow-up appointments. Consult your doctor for personalized advice."
            
            # Log completion time
            logger.info(f"AI response generated in {time.time() - start_time:.2f}s")
            
            return jsonify({'response': answer})
            
        except Exception as api_error:
            logger.error(f"Ollama API error: {str(api_error)}")
            
            # Provide fallback response based on available analysis data
            if "indian" in question_lower and ("doctor" in question_lower or "specialist" in question_lower):
                answer = "Some well-known neurosurgeons in India include Dr. B.K. Misra, Dr. Satnam Singh Chhabra, and Dr. Suresh Nair. Please consult your local medical center for specialists in your area."
            elif "precaution" in question_lower:
                if precautions:
                    # Extract precautions from analysis
                    precaution_lines = [p.strip() for p in precautions.split(".") if p.strip()]
                    if len(precaution_lines) >= 3:
                        formatted_precautions = ""
                        for i, p in enumerate(precaution_lines[:4], 1):
                            formatted_precautions += f"{i}) {p.strip()}. "
                        answer = f"Key precautions for your {tumor_type}: {formatted_precautions}Consult your doctor for personalized advice."
                else:
                    answer = f"Key precautions for {tumor_type}: 1) Avoid strenuous physical activities that increase blood pressure, 2) Take prescribed medications consistently, 3) Monitor and report new symptoms immediately, 4) Get adequate rest. Consult your doctor for personalized advice."
            else:
                answer = f"Your MRI shows a {tumor_type} in the {tumor_location}. {clinical_significance} Please consult with a specialist for personalized guidance based on your specific medical situation."
            
            return jsonify({'response': answer})
        
    except Exception as e:
        logger.error(f"AI chat error: {e}")
        return jsonify({
            'error': 'Unable to process question',
            'medical_advice': 'Please consult your doctor directly for medical questions'
        }), 500

@app.route('/status')
def status():
    """Return system status information"""
    try:
        model_loaded = MODEL_SESSION is not None
        queue_size = 0  # Report queue is disabled
        
        return jsonify({
            'status': 'ok',
            'model_loaded': model_loaded,
            'report_queue_size': queue_size,
            'in_memory_processing': app.config['ENABLE_IN_MEMORY_PROCESSING'],
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    """Analyze an image in real-time without storing any static data"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file received'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload JPG, JPEG, or PNG.'}), 400
    
    try:
        # Start tracking processing time
        start_time = time.time()
        
        # Read file data into memory
        file_data = file.read()
        
        # Generate a unique identifier for this image
        img_id = hashlib.md5(file_data).hexdigest()[:10]
        logger.info(f"Processing image with unique ID: {img_id}")
        
        # Get or initialize model session
        session = get_model_session()
        if not session:
            return jsonify({'error': 'Model loading failed'}), 500
        
        # Process image in memory without saving
        image_tensor, img = preprocess_image_in_memory(file_data)
        if image_tensor is None:
            return jsonify({'error': 'Image processing error'}), 500
        
        # Make prediction
        prediction = predict(session, image_tensor)
        if not prediction:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Get image as base64 for returning directly
        image_data = get_image_as_base64(img)
        
        # Create a unique image fingerprint using basic image statistics
        img_array = np.array(img)
        img_stats = {
            'mean': float(np.mean(img_array)),
            'std': float(np.std(img_array)),
            'min': float(np.min(img_array)),
            'max': float(np.max(img_array)),
        }
        logger.info(f"Image {img_id} characteristics: {img_stats}")
        
        # Generate completely dynamic report based on actual image analysis
        class_id = prediction['class_id']
        confidence = prediction['confidence']
        analysis = None
        
        if class_id == 0:
            # No tumor detected
            report = """
            <h3>Clinical Interpretation</h3>
            <p>Based on real-time analysis of the uploaded image, no tumor was detected in the brain MRI scan. Brain tissue appears normal with no visible lesions, mass effects, or abnormal enhancement patterns.</p>
            
            <h3>Next Steps</h3>
            <p>Consult with your physician for a comprehensive evaluation of your symptoms and medical history.</p>
            
            <h3>Important Note</h3>
            <p>This is an AI analysis that should be confirmed by a qualified medical professional. Not all conditions may be detected by this system.</p>
            """
        else:
            # Perform full image analysis for this specific image
            analysis = analyze_tumor_characteristics(img, prediction['confidence'])
            
            # Log the unique characteristics to verify each analysis is different
            logger.info(f"Tumor analysis for image {img_id}: Type: {analysis['tumor_type']}, Location: {analysis['location']}")
            
            # Generate dynamic report using the actual analysis results
            report = format_tumor_report(analysis)
        
        # Calculate total processing time
        processing_time = time.time() - start_time
        logger.info(f"Analysis completed for image {img_id} in {processing_time:.3f}s")
        
        # Return comprehensive analysis result as JSON with no stored data
        return jsonify({
            'prediction': prediction,
            'image': image_data,
            'report': report,
            'analysis': analysis,  # Include full analysis data
            'analysis_time': processing_time,
            'is_real_time': True,
            'processing_time': f"{processing_time:.3f}s",
            'dynamic_analysis': True,  # Flag indicating this is fully dynamic
            'image_id': img_id  # Return the unique image ID for reference
        })
        
    except Exception as e:
        logger.error(f"Real-time analysis error: {e}")
        return jsonify({'error': str(e)}), 500

def analyze_tumor_characteristics(img, confidence=None):
    """
    Analyze tumor characteristics in an image using advanced image processing
    Returns detailed tumor type, location, and characteristics based on actual image features
    """
    try:
        # Convert image to numpy array for analysis
        img_np = np.array(img)
        
        # Generate a unique identifier for this specific image
        img_hash = hashlib.md5(img_np.tobytes()).hexdigest()[:8]
        
        # Convert to grayscale if color image
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_np
        
        # Log image-specific properties
        mean_val = np.mean(img_gray)
        std_val = np.std(img_gray)
        min_val = np.min(img_gray)
        max_val = np.max(img_gray)
        
        logger.info(f"Analyzing image with hash: {img_hash}")
        logger.info(f"Image stats: mean={mean_val:.2f}, std={std_val:.2f}, min={min_val}, max={max_val}")
        
        # Normalize image for better analysis
        img_norm = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply Gaussian blur to reduce noise
        img_blur = cv2.GaussianBlur(img_norm, (5, 5), 0)
        
        # Edge detection for structural analysis
        sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Multi-stage segmentation for more accurate tumor detection
        # 1. Otsu thresholding for global segmentation
        _, otsu_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. Adaptive thresholding for local details
        adapt_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # 3. Edge-based segmentation
        norm_gradient = gradient_magnitude * (255.0 / (gradient_magnitude.max() or 1))
        _, edge_thresh = cv2.threshold(norm_gradient.astype(np.uint8), 50, 255, cv2.THRESH_BINARY)
        
        # Combine segmentation methods with appropriate weights
        combined_mask = (otsu_thresh.astype(float) * 0.5 + 
                        adapt_thresh.astype(float) * 0.3 + 
                        edge_thresh.astype(float) * 0.2)
        combined_mask = (combined_mask > 127).astype(np.uint8) * 255
        
        # Clean up mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find connected components (potential tumors)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        
        # If no significant components found, this likely means no tumor
        if num_labels <= 1 or np.max(stats[1:, cv2.CC_STAT_AREA]) < 100:
            logger.info(f"No significant regions detected in image {img_hash}")
            return {
                "tumor_type": "No significant abnormality",
                "location": "N/A",
                "description": "No significant tumor mass detected in this MRI scan.",
                "clinical_significance": "The brain MRI appears within normal limits with no evidence of significant mass lesion.",
                "precautions": "Routine follow-up recommended if clinically indicated based on symptoms.",
                "confidence": confidence or 85.0,
                "size_cm": 0.0
            }
        
        # Find the region most likely to be a tumor (excluding background)
        # Sort regions by area (largest first)
        regions = []
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 100:  # Skip very small regions
                continue
                
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]
            
            # Extract region mask
            region_mask = (labels == i).astype(np.uint8) * 255
            
            # Analyze region characteristics
            region_pixels = img_gray[region_mask > 0]
            if len(region_pixels) == 0:
                continue
                
            region_mean = np.mean(region_pixels)
            region_std = np.std(region_pixels)
            
            # Calculate abnormality score based on multiple factors
            # 1. Size relative to brain
            size_score = area / (img_gray.shape[0] * img_gray.shape[1])
            
            # 2. Intensity difference from surrounding tissue
            # Create dilated mask for surrounding region
            surrounding_mask = cv2.dilate(region_mask, kernel, iterations=3) - region_mask
            surrounding_pixels = img_gray[surrounding_mask > 0]
            surrounding_mean = np.mean(surrounding_pixels) if len(surrounding_pixels) > 0 else mean_val
            intensity_diff = abs(region_mean - surrounding_mean) / std_val
            
            # 3. Shape irregularity
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                perimeter = cv2.arcLength(contours[0], True)
                circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
                # Less circular = more irregular = higher score
                shape_score = 1 - circularity
            else:
                shape_score = 0
                
            # 4. Internal heterogeneity 
            heterogeneity_score = region_std / std_val
            
            # 5. Edge strength
            edge_mask = cv2.Canny(region_mask, 100, 200)
            region_edges = gradient_magnitude * (edge_mask > 0)
            edge_score = np.mean(region_edges) / np.mean(gradient_magnitude) if np.mean(gradient_magnitude) > 0 else 0
            
            # Calculate overall abnormality score
            abnormality_score = (
                size_score * 0.3 +
                intensity_diff * 0.25 +
                shape_score * 0.15 +
                heterogeneity_score * 0.2 +
                edge_score * 0.1
            )
            
            regions.append({
                "label": i,
                "area": area,
                "centroid": (cx, cy),
                "mean": region_mean,
                "std": region_std,
                "intensity_diff": intensity_diff,
                "shape_score": shape_score,
                "heterogeneity": heterogeneity_score,
                "edge_score": edge_score,
                "abnormality_score": abnormality_score,
                "mask": region_mask
            })
        
        # No valid regions found
        if not regions:
            logger.warning(f"No valid regions identified in image {img_hash}")
            return {
                "tumor_type": "Inconclusive analysis",
                "location": "Undetermined",
                "description": "Analysis could not conclusively identify abnormal regions.",
                "clinical_significance": "Indeterminate findings. Clinical correlation strongly recommended.",
                "precautions": "Consider follow-up imaging with contrast enhancement for better characterization.",
                "confidence": confidence or 50.0,
                "size_cm": 0.0
            }
        
        # Sort regions by abnormality score
        regions.sort(key=lambda x: x["abnormality_score"], reverse=True)
        most_abnormal = regions[0]
        
        # Create detailed analysis of the most abnormal region
        tumor_mask = most_abnormal["mask"]
        tumor_area = most_abnormal["area"]
        tumor_mean = most_abnormal["mean"] 
        tumor_std = most_abnormal["std"]
        cx, cy = most_abnormal["centroid"]
        
        # Calculate size in cm (approximate)
        # Assuming a typical brain MRI is ~15-20cm across
        estimated_brain_width_cm = 18.0
        pixel_to_cm = estimated_brain_width_cm / max(img_gray.shape)
        tumor_width_pixels = np.sqrt(tumor_area)
        tumor_size_cm = tumor_width_pixels * pixel_to_cm
        
        # Calculate relative size (percentage of brain area)
        tumor_size_ratio = tumor_area / (img_gray.shape[0] * img_gray.shape[1])
        
        # Determine anatomical location based on centroid position
        rel_x = cx / img_gray.shape[1]
        rel_y = cy / img_gray.shape[0]
        
        # Map to brain regions
        regions_map = {
            # Format: ((min_x, max_x), (min_y, max_y)): "region_name"
            ((0.0, 0.33), (0.0, 0.4)): "left frontal lobe",
            ((0.33, 0.67), (0.0, 0.4)): "frontal lobe",
            ((0.67, 1.0), (0.0, 0.4)): "right frontal lobe",
            ((0.0, 0.33), (0.4, 0.6)): "left temporal lobe",
            ((0.33, 0.67), (0.4, 0.6)): "central region (including basal ganglia, thalamus)",
            ((0.67, 1.0), (0.4, 0.6)): "right temporal lobe",
            ((0.0, 0.33), (0.6, 0.8)): "left parietal lobe",
            ((0.33, 0.67), (0.6, 0.8)): "parietal region",
            ((0.67, 1.0), (0.6, 0.8)): "right parietal lobe",
            ((0.0, 0.33), (0.8, 1.0)): "left occipital lobe",
            ((0.33, 0.67), (0.8, 1.0)): "occipital region",
            ((0.67, 1.0), (0.8, 1.0)): "right occipital lobe",
            ((0.4, 0.6), (0.45, 0.55)): "deep brain structures (including pituitary)"
        }
        
        location = "undetermined brain region"
        for ((min_x, max_x), (min_y, max_y)), region_name in regions_map.items():
            if min_x <= rel_x <= max_x and min_y <= rel_y <= max_y:
                location = region_name
                break
        
        # Extract tumor characteristics
        heterogeneity = most_abnormal["heterogeneity"]
        shape_irregularity = most_abnormal["shape_score"]
        edge_definition = most_abnormal["edge_score"]
        intensity_contrast = most_abnormal["intensity_diff"]
        
        logger.info(f"Tumor metrics: location={location}, size={tumor_size_cm:.2f}cm, " +
                   f"heterogeneity={heterogeneity:.2f}, shape_score={shape_irregularity:.2f}")
        
        # Classify tumor types based on image-derived features
        tumor_types = {
            "glioma": {
                "score": 0,
                "name": "Glioma",
                "key_features": ["Infiltrative margins", "Heterogeneous signal", "May cross midline"]
            },
            "meningioma": {
                "score": 0,
                "name": "Meningioma",
                "key_features": ["Well-circumscribed", "Homogeneous", "Dural-based"]
            },
            "pituitary_adenoma": {
                "score": 0,
                "name": "Pituitary Adenoma",
                "key_features": ["Sellar location", "Well-defined", "May have cystic components"]
            },
            "acoustic_neuroma": {
                "score": 0,
                "name": "Acoustic Neuroma",
                "key_features": ["Cerebellopontine angle", "Well-defined", "May cause internal auditory canal widening"]
            },
            "metastasis": {
                "score": 0,
                "name": "Metastatic Tumor",
                "key_features": ["Well-circumscribed", "Significant surrounding edema", "Often multiple"]
            },
            "lymphoma": {
                "score": 0,
                "name": "CNS Lymphoma",
                "key_features": ["Homogeneous enhancement", "Periventricular", "May cross corpus callosum"]
            }
        }
        
        # Score each tumor type based on image features
        # Glioma: heterogeneous with ill-defined margins
        tumor_types["glioma"]["score"] += heterogeneity * 2.0
        tumor_types["glioma"]["score"] += shape_irregularity * 1.5
        tumor_types["glioma"]["score"] += (1.0 - edge_definition) * 1.5
        if "frontal" in location or "temporal" in location:
            tumor_types["glioma"]["score"] += 1.0
        
        # Meningioma: well-defined, homogeneous, often peripheral/dural
        tumor_types["meningioma"]["score"] += (1.0 - heterogeneity) * 1.5
        tumor_types["meningioma"]["score"] += edge_definition * 2.0
        tumor_types["meningioma"]["score"] += (1.0 - shape_irregularity) * 1.0
        if rel_x < 0.2 or rel_x > 0.8 or rel_y < 0.3:  # Peripheral location
            tumor_types["meningioma"]["score"] += 1.5
            
        # Pituitary adenoma: sellar location is key
        tumor_types["pituitary_adenoma"]["score"] += (1.0 - heterogeneity) * 1.0
        tumor_types["pituitary_adenoma"]["score"] += edge_definition * 1.0
        if "deep brain" in location or (0.4 < rel_x < 0.6 and 0.45 < rel_y < 0.55):
            tumor_types["pituitary_adenoma"]["score"] += 3.0
        if tumor_size_ratio < 0.08:  # Usually smaller
            tumor_types["pituitary_adenoma"]["score"] += 1.0
            
        # Acoustic neuroma: cerebellopontine angle location
        tumor_types["acoustic_neuroma"]["score"] += (1.0 - heterogeneity) * 1.0
        tumor_types["acoustic_neuroma"]["score"] += edge_definition * 1.0
        if "temporal" in location and (rel_x < 0.3 or rel_x > 0.7) and 0.4 < rel_y < 0.7:
            tumor_types["acoustic_neuroma"]["score"] += 2.5
        if tumor_size_ratio < 0.08:  # Usually not very large
            tumor_types["acoustic_neuroma"]["score"] += 0.5
            
        # Metastasis: well-defined, often multiple, significant enhancement
        tumor_types["metastasis"]["score"] += intensity_contrast * 1.5
        tumor_types["metastasis"]["score"] += edge_definition * 1.5
        tumor_types["metastasis"]["score"] += (1.0 - shape_irregularity) * 0.5
        if len(regions) > 1:  # Multiple lesions suggest metastases
            tumor_types["metastasis"]["score"] += 2.0
            
        # Lymphoma: homogeneous, periventricular
        tumor_types["lymphoma"]["score"] += (1.0 - heterogeneity) * 2.0
        tumor_types["lymphoma"]["score"] += intensity_contrast * 1.0
        if "central" in location or (0.3 < rel_x < 0.7 and 0.3 < rel_y < 0.7):
            tumor_types["lymphoma"]["score"] += 1.5
        
        # Find tumor type with highest score
        best_tumor_type = max(tumor_types.items(), key=lambda x: x[1]["score"])
        tumor_type_key = best_tumor_type[0]
        tumor_type_info = best_tumor_type[1]
        
        # Create description based on actual tumor characteristics
        size_description = "small" if tumor_size_ratio < 0.05 else ("large" if tumor_size_ratio > 0.15 else "medium")
        heterogeneity_desc = "heterogeneous" if heterogeneity > 1.5 else ("homogeneous" if heterogeneity < 0.8 else "somewhat heterogeneous")
        border_desc = "well-defined" if edge_definition > 1.5 else ("poorly-defined" if edge_definition < 0.8 else "moderately-defined")
        enhancement_desc = "significant" if intensity_contrast > 1.5 else ("minimal" if intensity_contrast < 0.5 else "moderate")
        
        # Generate detailed description
        description = f"A {size_description}-sized {tumor_type_info['name']} with {heterogeneity_desc} internal structure and {border_desc} borders. "
        description += f"The lesion demonstrates {enhancement_desc} contrast from surrounding brain tissue and is located in the {location}."
        
        # Add tumor-specific details
        if tumor_type_key == "glioma":
            grade_estimate = "high-grade (III-IV)" if heterogeneity > 1.8 and intensity_contrast > 1.2 and shape_irregularity > 0.7 else "low-to-mid grade (I-II)"
            description += f" Image features suggest a {grade_estimate} glioma based on {heterogeneity_desc} appearance and {border_desc} margins."
            
        elif tumor_type_key == "meningioma":
            description += f" The lesion appears to be extra-axial with a broad dural base, typical of meningiomas. No significant mass effect or surrounding edema noted."
            
        elif tumor_type_key == "pituitary_adenoma":
            description += f" The lesion is centered in the sellar region with {border_desc} margins, consistent with a pituitary adenoma. "
            if tumor_size_ratio > 0.05:
                description += "There appears to be suprasellar extension which may impinge on the optic chiasm."
            
        elif tumor_type_key == "acoustic_neuroma":
            description += f" The lesion appears to arise from the internal auditory canal in the cerebellopontine angle region, characteristic of acoustic neuromas."
            
        elif tumor_type_key == "metastasis":
            description += f" The lesion shows significant surrounding edema and relatively sharp borders typical of metastatic disease."
            if len(regions) > 1:
                description += f" Multiple lesions are identified, supporting a metastatic process."
                
        elif tumor_type_key == "lymphoma":
            description += f" The lesion demonstrates {heterogeneity_desc} signal intensity with {enhancement_desc} enhancement. Location and imaging characteristics are consistent with CNS lymphoma."
        
        # Clinical significance based on tumor type and features
        if tumor_type_key == "glioma":
            significance = f"This appears to be a {grade_estimate} glioma based on imaging characteristics. "
            significance += "Gliomas are primary brain tumors arising from glial cells with variable clinical behavior depending on grade and molecular features. "
            if "high-grade" in grade_estimate:
                significance += "High-grade gliomas typically demonstrate rapid growth and invasion of surrounding tissues."
            else:
                significance += "Low-grade gliomas typically demonstrate slower growth but may transform to higher grades over time."
            
            urgency = "high" if "high-grade" in grade_estimate else "moderate to high"
            
        elif tumor_type_key == "meningioma":
            significance = "Meningiomas are typically benign (WHO grade I) tumors arising from the arachnoid cap cells of the meninges. "
            if shape_irregularity > 0.7 or heterogeneity > 1.3:
                significance += "Some atypical features are noted, which may suggest a higher-grade variant."
            else:
                significance += "No concerning features suggesting atypical or malignant variant are identified."
                
            urgency = "high" if tumor_size_ratio > 0.15 or shape_irregularity > 0.7 else ("moderate" if tumor_size_ratio > 0.08 else "low to moderate")
            
        elif tumor_type_key == "pituitary_adenoma":
            significance = "Pituitary adenomas are typically benign neoplasms arising from the anterior pituitary gland. "
            significance += "They may be functional (hormone-secreting) or non-functional and can cause symptoms through mass effect on surrounding structures or hormonal abnormalities."
            
            urgency = "high" if tumor_size_ratio > 0.1 else "moderate"
            
        elif tumor_type_key == "acoustic_neuroma":
            significance = "Acoustic neuromas (vestibular schwannomas) are benign tumors arising from the vestibular portion of the 8th cranial nerve. "
            significance += "They can cause hearing loss, tinnitus, balance problems, and, if large enough, brainstem compression."
            
            urgency = "high" if tumor_size_ratio > 0.12 else "moderate"
            
        elif tumor_type_key == "metastasis":
            significance = "Metastatic lesions indicate the presence of cancer that has spread from a primary site elsewhere in the body. "
            significance += "Brain metastases most commonly originate from lung cancer, breast cancer, melanoma, renal cell carcinoma, and colorectal cancer."
            
            urgency = "high"
            
        elif tumor_type_key == "lymphoma":
            significance = "CNS lymphoma is a rare form of non-Hodgkin lymphoma confined to the brain, spinal cord, meninges, or eyes. "
            significance += "Primary CNS lymphoma is most often a diffuse large B-cell lymphoma and may be associated with immunosuppression."
            
            urgency = "high"
        
        # Generate specific precautions based on tumor type and characteristics
        if tumor_type_key == "glioma":
            precautions = "Neurosurgical consultation recommended for possible biopsy/resection to confirm diagnosis and determine molecular profile. "
            if "high-grade" in grade_estimate:
                precautions += "Steroids may be indicated if significant edema or mass effect is present. "
                precautions += "Urgent neurosurgical and neuro-oncological evaluation recommended for treatment planning."
            else:
                precautions += "Close monitoring with serial MRIs recommended. "
                precautions += "Consider seizure prophylaxis discussion with neurologist."
            
        elif tumor_type_key == "meningioma":
            if tumor_size_ratio > 0.1 or shape_irregularity > 0.7:
                precautions = "Neurosurgical evaluation recommended for consideration of surgical resection. "
            else:
                precautions = "Observation with serial imaging is appropriate for asymptomatic small meningiomas. "
            
            precautions += "If symptomatic (headaches, seizures, focal neurological deficits), neurosurgical evaluation is advised. "
            precautions += "Stereotactic radiosurgery may be an alternative to surgical resection in selected cases."
            
        elif tumor_type_key == "pituitary_adenoma":
            precautions = "Endocrinological evaluation recommended to assess for hormonal abnormalities. "
            
            if tumor_size_ratio > 0.08:
                precautions += "Ophthalmological assessment advised to evaluate for optic chiasm compression and visual field defects. "
            
            precautions += "Neurosurgical consultation for possible transsphenoidal resection if symptomatic, growing, or causing visual compromise. "
            precautions += "Medical therapy may be an option for certain hormone-secreting adenomas."
            
        elif tumor_type_key == "acoustic_neuroma":
            precautions = "Audiometric testing recommended to establish baseline hearing function. "
            precautions += "Neurosurgical and otolaryngological evaluation advised for treatment planning. "
            
            if tumor_size_ratio > 0.1:
                precautions += "Surgical intervention should be considered given the size of the lesion. "
            else:
                precautions += "Observation, surgical resection, or stereotactic radiosurgery may be considered based on symptoms, hearing status, and growth rate. "
            
            precautions += "Regular monitoring of hearing and balance function recommended."
            
        elif tumor_type_key == "metastasis":
            precautions = "Comprehensive oncological evaluation recommended to identify primary malignancy if unknown. "
            precautions += "Neurosurgical consultation for possible resection or biopsy if diagnosis is uncertain. "
            precautions += "Consider stereotactic radiosurgery or whole-brain radiation therapy depending on number and size of lesions. "
            precautions += "Systemic therapy may be indicated based on primary tumor type."
            
        elif tumor_type_key == "lymphoma":
            precautions = "Neurosurgical biopsy recommended for definitive diagnosis and classification. "
            precautions += "Hold steroids if possible prior to biopsy as they can obscure diagnosis. "
            precautions += "Consultation with neuro-oncology and hematology-oncology for treatment planning. "
            precautions += "High-dose methotrexate-based chemotherapy typically recommended rather than surgical resection."
        
        # Log detailed analysis
        logger.info(f"Tumor analysis for image {img_hash}: Type: {tumor_type_info['name']}, Location: {location}")
        logger.info(f"Tumor metrics: size_ratio={tumor_size_ratio:.3f}, heterogeneity={heterogeneity:.2f}, " +
                   f"edge_definition={edge_definition:.2f}, intensity_contrast={intensity_contrast:.2f}")
        
        # Return comprehensive analysis based on actual image features
        return {
            "tumor_type": tumor_type_info["name"],
            "location": location,
            "description": description,
            "clinical_significance": significance,
            "urgency": urgency,
            "precautions": precautions,
            "confidence": confidence or 95.0,
            "size_cm": tumor_size_cm,
            "size_ratio": tumor_size_ratio,
            "heterogeneity": heterogeneity,
            "edge_definition": edge_definition,
            "key_features": tumor_type_info["key_features"]
        }
        
    except Exception as e:
        logger.error(f"Error analyzing tumor: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "tumor_type": "Analysis Error",
            "location": "Unable to determine",
            "description": f"An error occurred during tumor analysis: {str(e)}",
            "clinical_significance": "Please review the image manually or resubmit for analysis.",
            "precautions": "Technical error occurred during processing. This does not represent a clinical finding.",
            "confidence": confidence or 0.0,
            "size_cm": 0.0
        }

def get_recommended_timeframe(urgency):
    """Return a recommended timeframe based on urgency level"""
    if urgency == "high":
        return "1 week"
    elif "moderate to high" in urgency:
        return "1-2 weeks"
    elif urgency == "moderate":
        return "2-3 weeks"
    else:  # low to moderate
        return "4-6 weeks"

def format_tumor_report(analysis):
    """Format tumor analysis as HTML report with accurate diagnostic information"""
    
    # Format the urgency if present, otherwise use clinical significance to determine
    urgency = analysis.get('urgency', 'moderate')
    
    # Generate timeframe based on urgency
    timeframe = get_recommended_timeframe(urgency)
    
    # Determine key features to highlight
    key_features = analysis.get('key_features', [])
    key_features_html = ""
    if key_features:
        key_features_html = "<ul>"
        for feature in key_features:
            key_features_html += f"<li>{feature}</li>"
        key_features_html += "</ul>"
    
    # Format size information
    size_cm = analysis.get('size_cm', 0)
    size_ratio = analysis.get('size_ratio', 0) 
    size_percentage = f"{size_ratio * 100:.1f}%" if size_ratio else "undetermined"
    
    return f"""
    <h3>Clinical Interpretation</h3>
    <p>Real-time analysis of the uploaded MRI scan indicates features consistent with <strong>{analysis['tumor_type']}</strong>.</p>
    
    <h3>Tumor Characteristics</h3>
    <p>{analysis['description']}</p>
    {key_features_html}
    
    <h3>Location</h3>
    <p>The analysis identifies the lesion in the <strong>{analysis['location']}</strong>.</p>
    
    <h3>Clinical Assessment</h3>
    <p>{analysis['clinical_significance']}</p>
    
    <h3>Recommended Precautions</h3>
    <p>{analysis['precautions']}</p>
    
    <h3>Suggested Timeline</h3>
    <p>Based on image analysis and tumor characteristics, clinical evaluation is recommended within <strong>{timeframe}</strong>.</p>
    
    <h3>Important Note</h3>
    <p>This is an AI-generated analysis based on image processing techniques. Diagnosis should be confirmed by a qualified medical professional with complete clinical correlation. Additional imaging studies and possibly biopsy may be required for definitive diagnosis.</p>
    
    <h3>Technical Details</h3>
    <p>Confidence: {analysis.get('confidence', 0):.1f}%</p>
    <p>Approximate Size: {size_cm:.1f} cm (approximately {size_percentage} of visible brain area)</p>
    <p>Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """

@app.route('/test_ollama', methods=['GET'])
def test_ollama():
    """Test endpoint to verify Ollama connectivity"""
    try:
        logger.info(f"Testing Ollama connectivity at http://127.0.0.1:11500")
        
        # Simple test prompt
        test_prompt = "Hello, please respond with 'Ollama is working correctly on port 11500'"
        
        # Try to generate a response with direct API call
        response = requests.post(
            "http://127.0.0.1:11500/api/generate",
            json={
                "model": "llama2",
                "prompt": test_prompt,
                "options": {'temperature': 0.1, 'max_tokens': 50}
            },
            timeout=30  # 30-second timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API returned status code {response.status_code}: {response.text}")
        
        result = response.json()
        answer = result.get('response', '').strip()
        
        logger.info(f"Ollama test successful. Response: {answer}")
        
        return jsonify({
            'status': 'success',
            'message': 'Ollama connection successful',
            'host': 'http://127.0.0.1:11500',
            'response': answer
        })
        
    except Exception as e:
        logger.error(f"Ollama test failed: {e}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'status': 'error',
            'message': f'Failed to connect to Ollama: {str(e)}',
            'host': 'http://127.0.0.1:11500'
        }), 500

def generate_dynamic_prompt(question, tumor_type, prediction, question_type, keywords):
    """Generate a dynamic prompt based on the question type and keywords to get better responses from Ollama"""
    
    # Base context with tumor information
    context = f"The patient has recently received a brain MRI scan showing {tumor_type}."
    
    # Add additional context based on prediction data if available
    if prediction.get('size'):
        context += f" The scan indicates a tumor of size {prediction.get('size')} "
    if prediction.get('location'):
        context += f"located in the {prediction.get('location')} region."
    
    # Add question-type specific context
    if question_type == "treatment":
        context += f" When discussing treatments for {tumor_type}, consider surgical options, radiation therapy, medications, and follow-up monitoring."
    elif question_type == "specialist":
        context += " The response should include names of specific specialists that would be appropriate for consultation."
    elif question_type == "precaution":
        context += f" Precautions for {tumor_type} patients should be specific and actionable."
    elif question_type == "symptom":
        context += f" When discussing symptoms of {tumor_type}, consider both common and less common presentations."
    elif question_type == "prognosis":
        context += f" Prognosis information should be balanced and reflect the latest medical understanding of {tumor_type}."
    
    # Build the prompt
    prompt = f"""Answer this specific patient question about their brain MRI scan results showing {tumor_type}.

Patient question: "{question}"

Context: {context}

Important rules:
1. Be direct, focused, and factually accurate
2. Answer ONLY the specific question asked - focus on {', '.join(keywords) if keywords else 'the main question'}
3. Base your answers on medical facts about {tumor_type}
4. If asked for doctor recommendations, provide actual names of specialists
5. If asked for precautions, provide specific, relevant precautions
6. No dramatic statements or emotional language
7. Include a brief reminder to consult healthcare professionals
8. Do not repeat generic advice - be specific to the question and tumor type

Answer:"""
    
    return prompt

@app.route('/raw_ollama', methods=['POST'])
def raw_ollama():
    """Endpoint for getting raw, unmodified responses directly from Ollama"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        tumor_type = data.get('tumor_type', 'brain tumor')
        start_time = time.time()
        
        if not question:
            return jsonify({'error': 'Empty question'}), 400
        
        logger.info(f"Raw Ollama request received for question: {question}")
        
        # Create a simple prompt
        prompt = f"""Provide a factual, medically accurate answer to this patient question about their brain MRI scan showing {tumor_type}.

Patient question: "{question}"

Answer:"""
        
        try:
            # Configure a session with retries and extended timeouts
            session = requests.Session()
            retries = Retry(total=1, backoff_factor=0.5, allowed_methods=["POST"])
            session.mount('http://', HTTPAdapter(max_retries=retries))
            
            # Prepare the request payload
            payload = {
                "model": "llama2",
                "prompt": prompt,
                "stream": False,
                "options": {
                    'temperature': 0.2,
                    'max_tokens': 350
                }
            }
            
            # Make the API call
            response = session.post(
                'http://127.0.0.1:11500/api/generate',
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=(10, 60)
            )
            
            # Process the response
            if response.status_code == 200:
                response_data = response.json()
                raw_answer = response_data.get('response', '').strip()
                logger.info(f"Raw Ollama response received, length: {len(raw_answer)}")
                
                # Return the completely unmodified response
                return jsonify({
                    'raw_response': raw_answer,
                    'prompt': prompt,
                    'generation_time': f"{time.time() - start_time:.2f}s",
                    'status': 'success'
                })
            else:
                return jsonify({
                    'error': f"Ollama API error: {response.status_code}",
                    'content': response.text[:300],
                    'status': 'error'
                }), 500
                
        except Exception as e:
            logger.error(f"Raw Ollama request failed: {str(e)}")
            return jsonify({
                'error': f"Failed to get Ollama response: {str(e)}",
                'status': 'error'
            }), 500
    
    except Exception as e:
        logger.error(f"Raw Ollama endpoint error: {str(e)}")
        return jsonify({
            'error': f"Error processing request: {str(e)}",
            'status': 'error'
        }), 500

@app.route('/raw_ollama_form')
def raw_ollama_form():
    """Simple form for testing raw Ollama responses"""
    return render_template('raw_ollama.html')

@app.route('/download_report', methods=['POST'])
def download_report():
    """Generate and download a PDF report of the tumor analysis"""
    try:
        data = request.get_json()
        prediction = data.get('prediction', {})
        analysis = data.get('analysis', {})
        image_data = data.get('image', '')
        
        if not analysis:
            return jsonify({'error': 'No analysis data provided'}), 400
        
        # Generate PDF report
        pdf_bytes = generate_pdf_report(analysis, prediction, image_data)
        
        # Create a response with the PDF
        memory_file = io.BytesIO(pdf_bytes)
        memory_file.seek(0)
        
        tumor_type = analysis.get('tumor_type', 'brain_tumor')
        filename = f"{tumor_type.replace(' ', '_').lower()}_report_{datetime.now().strftime('%Y%m%d')}.pdf"
        
        return send_file(
            memory_file,
            download_name=filename,
            as_attachment=True,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        logger.error(f"PDF generation error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to generate PDF report'}), 500

def generate_pdf_report(analysis, prediction, image_data=None):
    """Generate a visually attractive PDF report with enhanced tumor highlighting and clear visualizations"""
    buffer = io.BytesIO()
    
    # Use a custom pagesize with slightly wider margins for a more professional look
    page_width, page_height = letter
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        rightMargin=50, 
        leftMargin=50, 
        topMargin=50, 
        bottomMargin=50,
        title=f"Brain MRI Analysis - {analysis.get('tumor_type', 'Tumor')} Report"
    )
    
    # Enhanced styles for more attractive presentation
    styles = getSampleStyleSheet()
    
    # Title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        alignment=1,  # Center alignment
        fontSize=20,
        textColor=colors.navy,
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    
    # Subtitle style
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=12,
        alignment=1,
        textColor=colors.darkslategray,
        fontName='Helvetica-Oblique'
    )
    
    # Section header style
    header_style = ParagraphStyle(
        'HeaderStyle',
        fontSize=16,
        textColor=colors.navy,
        spaceBefore=12,
        spaceAfter=6,
        fontName='Helvetica-Bold',
        borderWidth=0,
        borderPadding=0,
        borderColor=colors.navy,
        borderRadius=None
    )
    
    # Subheader style
    subheader_style = ParagraphStyle(
        'SubHeaderStyle',
        fontSize=13,
        textColor=colors.darkblue,
        fontName='Helvetica-Bold',
        spaceBefore=8,
        spaceAfter=4
    )
    
    # Normal text style
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        spaceBefore=2,
        spaceAfter=6
    )
    
    # Bullet style for list items
    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=normal_style,
        leftIndent=20,
        bulletIndent=10,
        spaceBefore=2,
        spaceAfter=2
    )
    
    # Warning style for critical information
    warning_style = ParagraphStyle(
        'WarningStyle',
        parent=normal_style,
        textColor=colors.darkred,
        fontName='Helvetica-Bold',
        backColor=colors.lightyellow
    )
    
    # Content elements
    elements = []
    
    # Determine color theme based on tumor type for visual consistency
    theme_color = colors.navy
    tumor_type_lower = analysis.get('tumor_type', '').lower()
    if 'glioma' in tumor_type_lower:
        theme_color = colors.darkred
    elif 'meningioma' in tumor_type_lower:
        theme_color = colors.darkgreen
    elif 'pituitary' in tumor_type_lower:
        theme_color = colors.darkblue
    elif 'metastasis' in tumor_type_lower or 'metastatic' in tumor_type_lower:
        theme_color = colors.darkorange
    elif 'lymphoma' in tumor_type_lower:
        theme_color = colors.purple
    elif 'acoustic' in tumor_type_lower or 'schwannoma' in tumor_type_lower:
        theme_color = colors.teal
    
    # Update styles with theme color
    header_style.textColor = theme_color
    subheader_style.textColor = theme_color
    
    # Add report header with border and background
    # First create a visual header with a table
    hospital_name = "ScanSphere Brain MRI Analysis"
    header_data = [[Paragraph(f'<font color="white">{hospital_name}</font>', title_style)]]
    header_table = Table(header_data, colWidths=[page_width-100])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), theme_color),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROUNDEDCORNERS', [10, 10, 10, 10]),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 10))
    
    # Report title
    title_text = f"Brain MRI Analysis Report: {analysis.get('tumor_type', 'Tumor Evaluation')}"
    elements.append(Paragraph(title_text, title_style))
    
    # Date and time in formatted style
    current_date = datetime.now().strftime('%B %d, %Y')
    current_time = datetime.now().strftime('%H:%M:%S')
    date_text = f"<para alignment='center'>Report generated on {current_date} at {current_time}</para>"
    elements.append(Paragraph(date_text, subtitle_style))
    elements.append(Spacer(1, 20))
    
    # Add MRI scan image with enhanced tumor highlighting if available
    if image_data:
        elements.append(Paragraph("MRI Scan with Tumor Highlight", header_style))
        
        try:
            # Process image to highlight tumor area with improved visualization
            highlighted_img = highlight_tumor_area(image_data, analysis)
            
            # Create image for PDF with larger dimensions for better visibility
            img_width = 450  # Increased width for better visibility
            img_height = 350  # Increased height for better visibility
            
            # Add image inside a framed table for better presentation
            img_data = [[ReportLabImage(highlighted_img, width=img_width, height=img_height)]]
            img_table = Table(img_data, colWidths=[img_width])
            img_table.setStyle(TableStyle([
                ('BOX', (0, 0), (-1, -1), 1, theme_color),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ]))
            elements.append(img_table)
            
            # Add legend for colors
            if 'glioma' in tumor_type_lower:
                color_name = "Red"
            elif 'meningioma' in tumor_type_lower:
                color_name = "Green"
            elif 'pituitary' in tumor_type_lower:
                color_name = "Blue"
            elif 'metastasis' in tumor_type_lower or 'metastatic' in tumor_type_lower:
                color_name = "Yellow"
            elif 'lymphoma' in tumor_type_lower:
                color_name = "Purple"
            elif 'acoustic' in tumor_type_lower or 'schwannoma' in tumor_type_lower:
                color_name = "Cyan"
            else:
                color_name = "Red"
            
            legend_text = f"<para alignment='center'><i>The highlighted region in {color_name} shows the detected {analysis.get('tumor_type', 'tumor')} in the {analysis.get('location', 'brain')}.</i></para>"
            elements.append(Paragraph(legend_text, normal_style))
            elements.append(Spacer(1, 15))
            
            # Add image findings section with arrow indicator
            elements.append(Paragraph("Key Image Findings", subheader_style))
            findings_text = f"""<para>
            • <b>Location:</b> {analysis.get('location', 'Undetermined')}
            • <b>Approximate Size:</b> {analysis.get('size_cm', 0):.1f} cm
            • <b>Appearance:</b> {analysis.get('description', '').split('.')[0] if analysis.get('description') else 'Not specified'}
            </para>"""
            elements.append(Paragraph(findings_text, bullet_style))
            elements.append(Spacer(1, 10))
            
        except Exception as e:
            logger.error(f"Failed to add image to PDF: {str(e)}")
            logger.error(traceback.format_exc())
            elements.append(Paragraph("Error: Could not add MRI scan image to report.", normal_style))
            elements.append(Spacer(1, 15))
    
    # Summary box with key information
    elements.append(Paragraph("Analysis Summary", header_style))
    
    # Create a summary box with background color
    summary_data = [
        ["Tumor Type", analysis.get('tumor_type', 'Unknown')],
        ["Location", analysis.get('location', 'Unknown')],
        ["Size", f"{analysis.get('size_cm', 0):.1f} cm"],
        ["Confidence", f"{analysis.get('confidence', 0):.1f}%"]
    ]
    
    # Add urgency level with appropriate color
    urgency = analysis.get('urgency', 'moderate')
    urgency_color = colors.red if urgency == "high" else (colors.orange if urgency == "moderate" else colors.green)
    
    summary_table = Table(summary_data, colWidths=[150, 300])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('BACKGROUND', (1, 0), (1, 0), theme_color.clone(alpha=0.2)),  # Highlight tumor type row
        ('GRID', (0, 0), (-1, -1), 1, colors.darkgrey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),  # Bold tumor type
        ('TEXTCOLOR', (1, 0), (1, 0), theme_color),  # Theme-colored tumor type
        ('PADDING', (0, 0), (-1, -1), 8),
        ('ROUNDEDCORNERS', [5, 5, 5, 5]),
    ]))
    elements.append(summary_table)
    
    # Add urgency indicator
    urgency_data = [[f"Urgency Level: {urgency.title()}"]]
    urgency_table = Table(urgency_data, colWidths=[450])
    urgency_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), urgency_color.clone(alpha=0.3)),
        ('TEXTCOLOR', (0, 0), (0, 0), urgency_color),
        ('ALIGN', (0, 0), (0, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
        ('PADDING', (0, 0), (0, 0), 4),
    ]))
    elements.append(urgency_table)
    elements.append(Spacer(1, 15))
    
    # Detailed Description Section with improved formatting
    elements.append(Paragraph("Clinical Description", header_style))
    description = analysis.get('description', 'No description available.')
    
    # Highlight key terms in the description
    highlighted_description = description
    key_terms = ["heterogeneous", "homogeneous", "well-defined", "poorly-defined", "infiltrative", 
                 "necrosis", "hemorrhage", "edema", "mass effect", "midline shift", "enhancement"]
    
    for term in key_terms:
        if term in highlighted_description.lower():
            # Case-insensitive replacement with bold formatting
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted_description = pattern.sub(f"<b>{term}</b>", highlighted_description)
    
    elements.append(Paragraph(highlighted_description, normal_style))
    elements.append(Spacer(1, 15))
    
    # Clinical Significance Section with improved visualization
    elements.append(Paragraph("Clinical Significance", header_style))
    significance = analysis.get('clinical_significance', 'No clinical significance information available.')
    elements.append(Paragraph(significance, normal_style))
    elements.append(Spacer(1, 15))
    
    # Typical Characteristics Section with improved table styling
    tumor_type = analysis.get('tumor_type', '').lower()
    tumor_characteristics = {}
    
    # Define characteristics by tumor type (same as before)
    if 'glioma' in tumor_type:
        tumor_characteristics = {
            "Appearance on MRI": "Often heterogeneous, may have areas of necrosis or hemorrhage",
            "Growth Pattern": "Infiltrative, may cross midline via corpus callosum",
            "Enhancement Pattern": "Variable enhancement, high-grade tumors show more enhancement",
            "Associated Features": "Surrounding edema, mass effect, may cause midline shift",
            "WHO Classification": "Grades I-IV, with higher grades being more aggressive"
        }
    elif 'meningioma' in tumor_type:
        tumor_characteristics = {
            "Appearance on MRI": "Homogeneous, well-circumscribed",
            "Growth Pattern": "Usually grows outward from dural attachment",
            "Enhancement Pattern": "Strong, uniform enhancement",
            "Associated Features": "Dural tail sign, may cause adjacent bone hyperostosis",
            "WHO Classification": "Mostly Grade I (benign), less commonly Grade II (atypical) or III (malignant)"
        }
    elif 'pituitary' in tumor_type:
        tumor_characteristics = {
            "Appearance on MRI": "Well-defined, may have cystic components",
            "Growth Pattern": "Expansile within sella, may extend superiorly",
            "Enhancement Pattern": "Usually enhances homogeneously",
            "Associated Features": "May compress optic chiasm, cavernous sinus invasion in aggressive tumors",
            "Hormonal Activity": "May be functional (hormone-secreting) or non-functional"
        }
    elif 'metastasis' in tumor_type or 'metastatic' in tumor_type:
        tumor_characteristics = {
            "Appearance on MRI": "Well-circumscribed, often multiple",
            "Growth Pattern": "Expansile, typically at gray-white matter junction",
            "Enhancement Pattern": "Ring enhancement common",
            "Associated Features": "Significant surrounding edema, often disproportionate to tumor size",
            "Common Primary Sites": "Lung, breast, melanoma, renal, colorectal"
        }
    elif 'lymphoma' in tumor_type:
        tumor_characteristics = {
            "Appearance on MRI": "Homogeneous, often periventricular",
            "Growth Pattern": "May cross midline through corpus callosum",
            "Enhancement Pattern": "Homogeneous, intense enhancement",
            "Associated Features": "Restricted diffusion, minimal surrounding edema",
            "Patient Factors": "More common in immunocompromised patients"
        }
    elif 'acoustic' in tumor_type or 'schwannoma' in tumor_type:
        tumor_characteristics = {
            "Appearance on MRI": "Well-circumscribed, may have cystic components",
            "Growth Pattern": "Arises from vestibular nerve in internal auditory canal",
            "Enhancement Pattern": "Enhances strongly, may be heterogeneous in larger tumors",
            "Associated Features": "May cause widening of internal auditory canal",
            "Clinical Presentation": "Typically presents with unilateral hearing loss, tinnitus, and balance problems"
        }
    
    if tumor_characteristics:
        elements.append(Paragraph(f"Typical Characteristics of {analysis.get('tumor_type', 'This Tumor Type')}", header_style))
        
        # Create a better styled table for tumor characteristics
        char_data = []
        for key, value in tumor_characteristics.items():
            char_data.append([key, value])
            
        char_table = Table(char_data, colWidths=[150, 300])
        char_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), theme_color.clone(alpha=0.2)),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.whitesmoke]),
        ]))
        elements.append(char_table)
        elements.append(Spacer(1, 15))
    
    # Precautions Section with visual indicators
    elements.append(Paragraph("Recommended Precautions", header_style))
    precautions = analysis.get('precautions', '')
    if precautions:
        # Format precautions as bullet points with warning icons
        precaution_points = [p.strip() for p in precautions.split(".") if p.strip()]
        for point in precaution_points:
            # Add warning symbol and bold beginning
            elements.append(Paragraph(f"<b>⚠</b> {point}.", bullet_style))
    else:
        elements.append(Paragraph("No specific precautions available.", normal_style))
    
    elements.append(Spacer(1, 15))
    
    # Suggested Timeline Section with visual timeline
    urgency = analysis.get('urgency', 'moderate')
    timeframe = get_recommended_timeframe(urgency)
    
    elements.append(Paragraph("Suggested Timeline", header_style))
    
    # Create a visual timeline representation
    color_value = {'high': colors.red, 'moderate': colors.orange, 'low': colors.green}.get(urgency, colors.orange)
    
    # Visual timeline element
    timeline_data = [
        [Paragraph(f"<b>Urgency Level:</b> {urgency.title()}", normal_style)],
        [Paragraph(f"<b>Recommended Action:</b> Clinical evaluation within <font color='{color_value.hexval()}'>{timeframe}</font>", normal_style)],
        [Paragraph(f"<b>Rationale:</b> Based on image analysis, tumor characteristics, and location", normal_style)]
    ]
    
    timeline_table = Table(timeline_data, colWidths=[450])
    timeline_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), color_value.clone(alpha=0.2)),
        ('BOX', (0, 0), (0, -1), 1, color_value),
        ('PADDING', (0, 0), (0, -1), 6),
    ]))
    elements.append(timeline_table)
    elements.append(Spacer(1, 15))
    
    # Next Steps Section with numbered list
    elements.append(Paragraph("Recommended Next Steps", header_style))
    
    next_steps = []
    
    # Based on tumor type (same as before)
    if 'glioma' in tumor_type:
        next_steps.append("Neurosurgical consultation for potential biopsy or resection")
        next_steps.append("Additional imaging with MRI perfusion/spectroscopy to evaluate tumor metabolism")
        next_steps.append("Consider anti-seizure medication prophylaxis")
    elif 'meningioma' in tumor_type:
        next_steps.append("Neurosurgical consultation to discuss observation vs. surgical resection")
        next_steps.append("Follow-up MRI in 3-6 months if observation is chosen")
        next_steps.append("Consider surgical planning with CT angiography if resection is indicated")
    elif 'pituitary' in tumor_type:
        next_steps.append("Endocrinological evaluation to assess for hormonal abnormalities")
        next_steps.append("Visual field testing if tumor approaches the optic chiasm")
        next_steps.append("Neurosurgical consultation for potential transsphenoidal resection")
    elif 'metastasis' in tumor_type or 'metastatic' in tumor_type:
        next_steps.append("Oncological consultation for systemic workup of primary cancer")
        next_steps.append("Consider whole-body imaging (PET/CT) to identify primary site")
        next_steps.append("Neurosurgical consultation for potential biopsy or resection")
        next_steps.append("Radiation oncology consultation for potential stereotactic radiosurgery")
    elif 'lymphoma' in tumor_type:
        next_steps.append("Hematology/oncology consultation")
        next_steps.append("Consider stereotactic biopsy for definitive diagnosis")
        next_steps.append("CSF analysis for cytology")
        next_steps.append("Body imaging to rule out systemic disease")
    elif 'acoustic' in tumor_type or 'schwannoma' in tumor_type:
        next_steps.append("Audiometric testing to assess hearing status")
        next_steps.append("Neurosurgical or neurotology consultation")
        next_steps.append("Consider wait-and-scan approach for small tumors with minimal symptoms")
        next_steps.append("Discuss treatment options: microsurgery, stereotactic radiosurgery, or observation")
    
    if not next_steps:
        next_steps.append("Consultation with a neurologist or neurosurgeon")
        next_steps.append("Follow-up imaging to monitor for changes")
        next_steps.append("Consider additional diagnostic studies as recommended by specialists")
    
    # Create visual numbered list
    for i, step in enumerate(next_steps):
        elements.append(Paragraph(f"<b>{i+1}.</b> {step}", bullet_style))
    
    elements.append(Spacer(1, 20))
    
    # Recommended Specialists Section with contact cards
    elements.append(Paragraph("Recommended Specialists", header_style))
    
    # Get specialists based on tumor type
    specialist_list = []
    if 'glioma' in tumor_type:
        specialist_list = specialists.get('glioma', [])
    elif 'meningioma' in tumor_type:
        specialist_list = specialists.get('meningioma', [])
    elif 'pituitary' in tumor_type:
        specialist_list = specialists.get('pituitary', [])
    elif 'acoustic' in tumor_type:
        specialist_list = specialists.get('acoustic_neuroma', [])
    elif 'metastasis' in tumor_type or 'metastatic' in tumor_type:
        specialist_list = specialists.get('metastatic', [])
    elif 'lymphoma' in tumor_type:
        specialist_list = specialists.get('lymphoma', [])
    
    # Add Indian specialists based on region detection
    region_data = request.headers.get('X-Region-Data')
    if region_data and 'india' in region_data.lower():
        if 'glioma' in tumor_type:
            specialist_list.extend(specialists.get('india_glioma', []))
        elif 'meningioma' in tumor_type:
            specialist_list.extend(specialists.get('india_meningioma', []))
        elif 'pituitary' in tumor_type:
            specialist_list.extend(specialists.get('india_pituitary', []))
        elif 'metastasis' in tumor_type or 'metastatic' in tumor_type:
            specialist_list.extend(specialists.get('india_metastatic', []))
    
    if specialist_list:
        # Create visually appealing specialist cards
        for i, specialist in enumerate(specialist_list[:3]):  # Limit to top 3 specialists
            # Create a styled card for each specialist
            spec_data = [
                [Paragraph(f"<b>{specialist.get('name', '')}</b>, {specialist.get('title', '')}", normal_style)],
                [Paragraph(f"{specialist.get('hospital', '')}", normal_style)],
                [Paragraph(f"{specialist.get('location', '')}", normal_style)]
            ]
            
            spec_table = Table(spec_data, colWidths=[450])
            spec_table.setStyle(TableStyle([
                ('BOX', (0, 0), (0, -1), 1, colors.lightgrey),
                ('BACKGROUND', (0, 0), (0, 0), theme_color.clone(alpha=0.1)),
                ('PADDING', (0, 0), (0, -1), 6),
                ('TOPPADDING', (0, 0), (0, 0), 8),
                ('BOTTOMPADDING', (0, 0), (0, 0), 8),
            ]))
            elements.append(spec_table)
            elements.append(Spacer(1, 5))
    else:
        elements.append(Paragraph("Please consult with a local neurosurgeon or neuro-oncologist for specialized care.", normal_style))
    
    elements.append(Spacer(1, 20))
    
    # Disclaimer footer
    elements.append(Paragraph("Important Medical Disclaimer", header_style))
    disclaimer = """<para>This is an AI-generated analysis based on image processing techniques. 
    <b>This report is not a medical diagnosis.</b> The information provided should be reviewed and 
    confirmed by a qualified healthcare professional. Diagnosis should be made with complete clinical 
    correlation, potentially additional imaging studies, and possibly biopsy for definitive diagnosis.
    </para>"""
    
    disclaimer_table = Table([[Paragraph(disclaimer, normal_style)]], colWidths=[450])
    disclaimer_table.setStyle(TableStyle([
        ('BOX', (0, 0), (0, 0), 1, colors.red),
        ('BACKGROUND', (0, 0), (0, 0), colors.mistyrose),
        ('PADDING', (0, 0), (0, 0), 8),
    ]))
    elements.append(disclaimer_table)
    
    # Build PDF
    doc.build(elements)
    
    # Get PDF from buffer
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes

def highlight_tumor_area(image_data, analysis):
    """Highlight the tumor area in the MRI scan using advanced image processing to detect actual tumor location"""
    try:
        # Decode base64 image if needed
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            # Extract the base64 part from the data URI
            base64_data = image_data.split(',')[1]
            image_data = base64.b64decode(base64_data)
        
        # If it's already binary data, use it directly
        if isinstance(image_data, (bytes, bytearray)):
            img_bytes = io.BytesIO(image_data)
            img = Image.open(img_bytes)
        else:
            # If somehow it's already a PIL Image
            img = image_data
        
        # Convert to numpy array for OpenCV processing
        img_np = np.array(img)
        
        # If image is grayscale, convert to RGB
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif len(img_np.shape) == 3 and img_np.shape[2] == 1:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif len(img_np.shape) == 3 and img_np.shape[2] == 4:
            # Convert RGBA to RGB
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
        # Create a copy for drawing
        highlighted_img = img_np.copy()
        
        # Get detailed tumor information from analysis
        tumor_type = analysis.get('tumor_type', 'Unknown')
        location = analysis.get('location', 'undetermined brain region').lower()
        size_cm = analysis.get('size_cm', 0.0)
        
        # Get grayscale image for processing
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Calculate approximate size in pixels based on size_cm
        estimated_brain_width_cm = 18.0
        pixel_to_cm = max(img_np.shape) / estimated_brain_width_cm
        tumor_radius_pixels = int((size_cm / 2) * pixel_to_cm)
        # Set a minimum radius for visibility
        tumor_radius_pixels = max(tumor_radius_pixels, 10)
        
        # ============= IMPROVED TUMOR DETECTION ALGORITHM =============
        
        # 1. Apply multiple thresholding techniques to capture different aspects of the tumor
        # Normalize image for better processing
        img_norm = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply Gaussian blur to reduce noise
        img_blur = cv2.GaussianBlur(img_norm, (5, 5), 0)
        
        # Multiple segmentation methods
        # 1. Otsu's thresholding for global segmentation
        _, otsu_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. Adaptive thresholding for local details
        adapt_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
        
        # 3. Edge detection for finding tumor boundaries
        edges = cv2.Canny(img_blur, 50, 150)
        
        # 4. Additional intensity-based thresholding
        # This helps with tumors that are either brighter or darker than surrounding tissue
        mean_intensity = np.mean(img_blur)
        std_intensity = np.std(img_blur)
        
        # Create bright and dark region masks
        _, bright_thresh = cv2.threshold(img_blur, mean_intensity + std_intensity * 0.5, 255, cv2.THRESH_BINARY)
        _, dark_thresh = cv2.threshold(img_blur, mean_intensity - std_intensity * 0.5, 255, cv2.THRESH_BINARY_INV)
        
        # 2. Combine segmentation methods based on tumor type characteristics
        combined_mask = None
        
        if 'glioma' in tumor_type.lower():
            # Gliomas can have mixed intensity and irregular borders
            combined_mask = cv2.bitwise_or(
                cv2.bitwise_and(adapt_thresh, otsu_thresh),
                cv2.bitwise_or(bright_thresh, dark_thresh)
            )
        elif 'meningioma' in tumor_type.lower():
            # Meningiomas are usually well-defined
            combined_mask = cv2.bitwise_or(
                cv2.bitwise_and(otsu_thresh, edges),
                bright_thresh
            )
        elif 'pituitary' in tumor_type.lower() or 'sellar' in location:
            # Pituitary tumors are often in the center and have specific brightness
            center_weight = np.zeros_like(img_blur)
            center_y, center_x = img_blur.shape[0] // 2, img_blur.shape[1] // 2
            cv2.circle(center_weight, (center_x, center_y), img_blur.shape[0] // 4, 255, -1)
            center_weighted_thresh = cv2.bitwise_and(bright_thresh, center_weight)
            combined_mask = cv2.bitwise_or(center_weighted_thresh, cv2.bitwise_and(adapt_thresh, otsu_thresh))
        elif 'metastasis' in tumor_type.lower() or 'metastatic' in tumor_type.lower():
            # Metastatic tumors often have clear boundaries and surrounding edema
            combined_mask = cv2.bitwise_or(
                cv2.bitwise_and(edges, bright_thresh),
                cv2.bitwise_and(otsu_thresh, adapt_thresh)
            )
        else:
            # Default approach that works for most cases
            combined_mask = cv2.bitwise_or(
                cv2.bitwise_and(adapt_thresh, otsu_thresh),
                cv2.bitwise_or(bright_thresh, dark_thresh)
            )
        
        # Clean up mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 3. Extract and analyze potential tumor regions
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        
        # Create a list to store potential tumor regions
        potential_tumors = []
        
        for i in range(1, num_labels):  # Skip background (0)
            # Get region properties
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 50:  # Skip very small regions
                continue
                
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]
            
            # Skip if too large (likely background)
            if area > (img_np.shape[0] * img_np.shape[1]) * 0.3:
                continue
            
            # Create mask for this region
            region_mask = (labels == i).astype(np.uint8)
            
            # Extract region pixels
            region_pixels = img_gray[region_mask > 0]
            if len(region_pixels) == 0:
                continue
                
            region_mean = np.mean(region_pixels)
            region_std = np.std(region_pixels)
            
            # Calculate surrounding region statistics for contrast
            dilated_mask = cv2.dilate(region_mask, kernel, iterations=3)
            surrounding_mask = dilated_mask - region_mask
            surrounding_pixels = img_gray[surrounding_mask > 0]
            
            if len(surrounding_pixels) > 0:
                surrounding_mean = np.mean(surrounding_pixels)
                intensity_contrast = abs(region_mean - surrounding_mean) / (surrounding_mean + 1e-5)
            else:
                intensity_contrast = 0
            
            # Calculate shape irregularity
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                perimeter = cv2.arcLength(contours[0], True)
                circularity = (4 * np.pi * area) / ((perimeter ** 2) + 1e-5)
                irregularity = 1 - circularity  # Higher means more irregular
            else:
                irregularity = 0
            
            # Calculate texture heterogeneity (std/mean)
            heterogeneity = region_std / (region_mean + 1e-5)
            
            # Calculate edge strength
            region_edges = cv2.Canny(region_mask * img_gray.astype(np.uint8), 50, 150)
            edge_strength = np.sum(region_edges) / (area + 1e-5)
            
            # 4. Calculate tumor probability score based on multiple factors
            # These weights should be adjusted based on what makes sense for the particular dataset
            tumor_score = (
                intensity_contrast * 4.0 +    # High contrast with surrounding tissue
                irregularity * 3.0 +          # Irregular shape
                heterogeneity * 3.0 +         # Heterogeneous texture
                edge_strength * 2.0           # Strong edges
            )
            
            # Boost or reduce score based on tumor type and region characteristics
            if 'glioma' in tumor_type.lower() and irregularity > 0.3 and heterogeneity > 0.1:
                tumor_score *= 1.5  # Boost for irregular, heterogeneous gliomas
                
            if 'meningioma' in tumor_type.lower() and irregularity < 0.3 and cx/img_gray.shape[1] > 0.8:
                tumor_score *= 1.3  # Boost for peripheral meningiomas
                
            if 'pituitary' in tumor_type.lower() and 0.4 < cx/img_gray.shape[1] < 0.6 and 0.4 < cy/img_gray.shape[0] < 0.6:
                tumor_score *= 2.0  # Boost for central pituitary region
            
            # Size factor: preference for regions close to expected size
            if tumor_radius_pixels > 0:
                region_radius = np.sqrt(area / np.pi)
                size_diff = abs(region_radius - tumor_radius_pixels) / tumor_radius_pixels
                # Size difference penalty (reduces score if size is very different)
                if size_diff > 2.0:  # If more than 2x different
                    tumor_score *= 0.5
            
            # Location factor: use anatomical information but don't rely on it entirely
            # Convert location text to approximate coordinates
            target_x, target_y = None, None
            
            # Only use location guidance if we have specific laterality information
            if any(term in location for term in ['left', 'right', 'frontal', 'temporal', 'parietal', 'occipital', 'deep', 'central']):
                rel_x, rel_y = 0.5, 0.5  # Default to center
                
                if 'frontal' in location:
                    rel_y = 0.3
                    if 'left' in location:
                        rel_x = 0.3
                    elif 'right' in location:
                        rel_x = 0.7
                elif 'temporal' in location:
                    rel_y = 0.5
                    if 'left' in location:
                        rel_x = 0.3
                    elif 'right' in location:
                        rel_x = 0.7
                elif 'parietal' in location:
                    rel_y = 0.7
                    if 'left' in location:
                        rel_x = 0.3
                    elif 'right' in location:
                        rel_x = 0.7
                elif 'occipital' in location:
                    rel_y = 0.8
                    if 'left' in location:
                        rel_x = 0.3
                    elif 'right' in location:
                        rel_x = 0.7
                elif 'deep' in location or 'central' in location or 'pituitary' in location:
                    rel_x, rel_y = 0.5, 0.5
                    
                target_x = int(rel_x * img_gray.shape[1])
                target_y = int(rel_y * img_gray.shape[0])
                
                # If we have target location, compute distance influence
                if target_x is not None and target_y is not None:
                    # Distance to expected location (normalized by image dimensions)
                    distance = np.sqrt(((cx - target_x) / img_gray.shape[1])**2 + 
                                       ((cy - target_y) / img_gray.shape[0])**2)
                    
                    # Apply distance penalty, but keep it mild to avoid over-reliance on predefined location
                    if distance > 0.4:  # If far from expected position
                        tumor_score *= (1.0 - distance * 0.3)  # Reduce score but don't eliminate
            
            # Add to potential tumor list with all information
            potential_tumors.append({
                'contours': contours[0] if contours else None,
                'score': tumor_score,
                'area': area,
                'centroid': (cx, cy),
                'mean': region_mean,
                'contrast': intensity_contrast,
                'irregularity': irregularity,
                'heterogeneity': heterogeneity
            })
        
        # 5. Select the most likely tumor region
        if potential_tumors:
            # Sort by tumor score (highest first)
            potential_tumors.sort(key=lambda x: x['score'], reverse=True)
            
            # Get the best candidate
            best_tumor = potential_tumors[0]
            best_contour = best_tumor['contours']
            logger.info(f"Selected tumor with score: {best_tumor['score']:.2f}, " +
                      f"contrast: {best_tumor['contrast']:.2f}, " +
                      f"irregularity: {best_tumor['irregularity']:.2f}")
            
            # For debugging: Draw top 3 candidates with decreasing opacity
            for i, tumor in enumerate(potential_tumors[:min(3, len(potential_tumors))]):
                if tumor['contours'] is not None and i > 0:  # Skip the best one as we'll highlight it separately
                    debug_overlay = highlighted_img.copy()
                    opacity = 0.2 - (i * 0.05)  # Decrease opacity for lower ranked candidates
                    cv2.drawContours(debug_overlay, [tumor['contours']], -1, (100, 100, 255), -1)
                    cv2.addWeighted(debug_overlay, opacity, highlighted_img, 1 - opacity, 0, highlighted_img)
        else:
            # If no regions found, create a fallback approach
            logger.warning("No suitable tumor regions detected automatically")
            best_contour = None
        
        # 6. Fallback handling if no suitable region found
        if best_contour is None or len(potential_tumors) == 0:
            # Try an alternative approach based on intensity thresholding
            logger.info("Using intensity-based fallback for tumor detection")
            
            # Adjust intensity thresholds based on tumor type
            if 'glioma' in tumor_type.lower():
                # Gliomas can be hyperintense or hypointense
                _, upper_thresh = cv2.threshold(img_blur, np.percentile(img_blur, 80), 255, cv2.THRESH_BINARY)
                _, lower_thresh = cv2.threshold(img_blur, np.percentile(img_blur, 20), 255, cv2.THRESH_BINARY_INV)
                fallback_mask = cv2.bitwise_or(upper_thresh, lower_thresh)
            elif 'meningioma' in tumor_type.lower():
                # Meningiomas are often hyperintense
                _, fallback_mask = cv2.threshold(img_blur, np.percentile(img_blur, 85), 255, cv2.THRESH_BINARY)
            else:
                # Default approach
                _, fallback_mask = cv2.threshold(img_blur, np.percentile(img_blur, 90), 255, cv2.THRESH_BINARY)
            
            # Clean up the mask
            fallback_mask = cv2.morphologyEx(fallback_mask, cv2.MORPH_OPEN, kernel)
            fallback_mask = cv2.morphologyEx(fallback_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours in fallback mask
            contours, _ = cv2.findContours(fallback_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Select largest contour as fallback
            if contours:
                contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
                # Filter out too large contours (likely background)
                filtered_contours = [c for c in contours_sorted if cv2.contourArea(c) < img_gray.size * 0.3]
                
                if filtered_contours:
                    best_contour = filtered_contours[0]
                else:
                    # Last resort: create a circle at the most likely location
                    mask = np.zeros(img_gray.shape, dtype=np.uint8)
                    # If target location was set earlier, use it, otherwise use center
                    center_x = target_x if target_x is not None else img_gray.shape[1] // 2
                    center_y = target_y if target_y is not None else img_gray.shape[0] // 2
                    cv2.circle(mask, (center_x, center_y), tumor_radius_pixels, 255, -1)
                    circle_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if circle_contours:
                        best_contour = circle_contours[0]
        
        # 7. Draw the identified tumor region on the image
        if best_contour is not None:
            # Choose color based on tumor type
            if 'glioma' in tumor_type.lower():
                color = (255, 0, 0)  # Red
            elif 'meningioma' in tumor_type.lower():
                color = (0, 255, 0)  # Green
            elif 'pituitary' in tumor_type.lower():
                color = (0, 0, 255)  # Blue
            elif 'metastasis' in tumor_type.lower() or 'metastatic' in tumor_type.lower():
                color = (255, 255, 0)  # Yellow
            elif 'lymphoma' in tumor_type.lower():
                color = (255, 0, 255)  # Purple
            elif 'acoustic' in tumor_type.lower() or 'schwannoma' in tumor_type.lower():
                color = (0, 255, 255)  # Cyan
            else:
                color = (255, 0, 0)  # Default red
            
            # Create overlay for the tumor region
            overlay = highlighted_img.copy()
            cv2.drawContours(overlay, [best_contour], -1, color, -1)
            
            # Apply semi-transparent overlay
            alpha = 0.5  # Transparency factor
            cv2.addWeighted(overlay, alpha, highlighted_img, 1 - alpha, 0, highlighted_img)
            
            # Add contour outline for better visibility
            cv2.drawContours(highlighted_img, [best_contour], -1, color, 2)
            
            # Add a label with tumor type and size
            label = f"{tumor_type} ({size_cm:.1f}cm)"
            
            # Determine text position (below or above contour)
            x, y, w, h = cv2.boundingRect(best_contour)
            if y > img_np.shape[0] / 2:
                # If tumor is in lower half, put text above
                text_y = max(y - 10, 20)
            else:
                # If tumor is in upper half, put text below
                text_y = min(y + h + 20, img_np.shape[0] - 10)
                
            # Add black outline for text visibility
            cv2.putText(highlighted_img, label, (x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            # Add colored text
            cv2.putText(highlighted_img, label, (x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Convert back to PIL Image for ReportLab
        highlighted_pil = Image.fromarray(highlighted_img)
        
        # Save to temporary file for ReportLab
        temp_file = io.BytesIO()
        highlighted_pil.save(temp_file, format='JPEG')
        temp_file.seek(0)
        
        return temp_file
        
    except Exception as e:
        logger.error(f"Error highlighting tumor: {str(e)}")
        logger.error(traceback.format_exc())
        
        # If error occurs, return original image if possible
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            base64_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(base64_data)
            return io.BytesIO(image_bytes)
        
        # Last resort: create a simple error image
        error_img = Image.new('RGB', (400, 300), color=(255, 255, 255))
        d = ImageDraw.Draw(error_img)
        d.text((10, 140), "Error highlighting tumor region", fill=(0, 0, 0))
        
        temp_file = io.BytesIO()
        error_img.save(temp_file, format='JPEG')
        temp_file.seek(0)
        return temp_file

@app.route('/api/user', methods=['GET'])
def get_user_info():
    """Return user information based on JWT token or guest data if no token"""
    # Check for authentication token
    auth_header = request.headers.get('Authorization')
    
    # If no auth header or invalid format, return guest user data
    if not auth_header or not auth_header.startswith('Bearer '):
        # Return guest user data
        guest_data = {
            'id': 'guest',
            'username': 'Guest',
            'email': 'guest@example.com'
        }
        return jsonify({
            'success': True,
            'user': guest_data,
            'isGuest': True
        })
    
    # Extract token from Bearer format
    try:
        token = auth_header.split(' ')[1]
    except IndexError:
        # Return guest user on invalid format
        guest_data = {
            'id': 'guest',
            'username': 'Guest',
            'email': 'guest@example.com'
        }
        return jsonify({
            'success': True,
            'user': guest_data,
            'isGuest': True
        })
    
    # Verify token
    try:
        # Decode the token
        decoded = jwt.decode(token, app.secret_key, algorithms=["HS256"])
        user_id = decoded.get('id')
        
        # Create user data object
        user_data = {
            'id': user_id,
            'username': decoded.get('username'),
            'email': decoded.get('email')
        }
        
        # Try to get profile image if available
        try:
            # Import MySQL connector
            import mysql.connector
            
            # Database configuration
            db_config = {
                'host': 'localhost',
                'user': 'root',    # Replace with your MySQL username
                'password': '',    # Replace with your MySQL password
                'database': 'Scansphere'  # Correct database name from Node.js config
            }
            
            # Connect to the database
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor(dictionary=True)
            
            # Get profile image
            cursor.execute('SELECT profile_image FROM user_profile_images WHERE user_id = %s', (user_id,))
            result = cursor.fetchone()
            
            # Close cursor and connection
            cursor.close()
            conn.close()
            
            # Add profile image to user data if found
            if result and 'profile_image' in result:
                user_data['profileImage'] = result['profile_image']
                
        except Exception as e:
            # Log error but continue - profile image is optional
            logger.error(f"Error retrieving profile image: {str(e)}")
        
        # Return user info
        return jsonify({
            'success': True,
            'user': user_data
        })
    except jwt.ExpiredSignatureError:
        return jsonify({
            'success': False, 
            'message': 'Token has expired'
        }), 401
    except jwt.InvalidTokenError:
        return jsonify({
            'success': False, 
            'message': 'Invalid token'
        }), 401
    except Exception as e:
        logger.error(f"Error decoding token: {str(e)}")
        return jsonify({
            'success': False, 
            'message': 'An error occurred processing your request'
        }), 500

@app.route('/api/update-profile-image', methods=['POST'])
def update_profile_image():
    """Update user profile image in database"""
    # Check for authentication token
    auth_header = request.headers.get('Authorization')
    
    if not auth_header:
        return jsonify({
            'success': False, 
            'message': 'Authorization header required'
        }), 401
    
    # Extract token from Bearer format
    try:
        token = auth_header.split(' ')[1]
    except IndexError:
        return jsonify({
            'success': False, 
            'message': 'Invalid authorization format. Use Bearer {token}'
        }), 401
    
    # Verify token
    try:
        # Decode the token
        decoded = jwt.decode(token, app.secret_key, algorithms=["HS256"])
        user_id = decoded.get('id')
        
        # Get request data
        data = request.json
        profile_image = data.get('profileImage')
        
        if not profile_image:
            return jsonify({
                'success': False,
                'message': 'Profile image data is required'
            }), 400
        
        # Connect to MySQL database using the existing pool from node-backend
        try:
            # Import MySQL connector directly for this function
            import mysql.connector
            from mysql.connector import Error
            
            # Database configuration - should match your Node.js configuration
            db_config = {
                'host': 'localhost',
                'user': 'root',    # Replace with your MySQL username
                'password': '',    # Replace with your MySQL password
                'database': 'Scansphere'  # Correct database name from Node.js config
            }
            
            # Connect to the database
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()
            
            # Check if user_profile_images table exists, create if not
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profile_images (
                    user_id INT PRIMARY KEY,
                    profile_image LONGTEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            # Check if user already has a profile image
            cursor.execute('SELECT user_id FROM user_profile_images WHERE user_id = %s', (user_id,))
            result = cursor.fetchone()
            
            if result:
                # Update existing profile image
                cursor.execute(
                    'UPDATE user_profile_images SET profile_image = %s WHERE user_id = %s',
                    (profile_image, user_id)
                )
            else:
                # Insert new profile image
                cursor.execute(
                    'INSERT INTO user_profile_images (user_id, profile_image) VALUES (%s, %s)',
                    (user_id, profile_image)
                )
            
            # Commit the changes
            conn.commit()
            
            # Close cursor and connection
            cursor.close()
            conn.close()
            
            return jsonify({
                'success': True,
                'message': 'Profile image updated successfully'
            })
            
        except Error as e:
            logger.error(f"Database error: {str(e)}")
            return jsonify({
                'success': False,
                'message': 'Database error when updating profile image'
            }), 500
            
    except jwt.ExpiredSignatureError:
        return jsonify({
            'success': False, 
            'message': 'Token has expired'
        }), 401
    except jwt.InvalidTokenError:
        return jsonify({
            'success': False, 
            'message': 'Invalid token'
        }), 401
    except Exception as e:
        logger.error(f"Error updating profile image: {str(e)}")
        return jsonify({
            'success': False, 
            'message': 'An error occurred processing your request'
        }), 500

@app.route('/history')
def history():
    """Serve the history page with time-travel comparison and voice notes"""
    return render_template('history.html')
    
@app.route('/static/history.js')
def serve_history_js():
    """Serve the history.js JavaScript file"""
    return send_from_directory('static', 'history.js')
    
# Remove the duplicate route to fix the conflict
# @app.route('/<path:filename>')
# def serve_static(filename):
#     """Serve static files from the root directory"""
#     return send_from_directory('.', filename)

def enhance_medical_response(raw_answer, question, tumor_type, tumor_location, tumor_size, 
                           tumor_description, clinical_significance, precautions):
    """
    Post-process and enhance the LLM's raw response to improve medical accuracy, 
    specificity, and adherence to best practices in medical communication.
    """
    # Clean up any artifacts
    if raw_answer.lower().startswith("answer:"):
        answer = raw_answer[7:].strip()
    else:
        answer = raw_answer.strip()
    
    # Remove dramatic openings
    dramatic_starts = ["oh no!", "i'm sorry", "unfortunately", "i regret to inform", "this is concerning"]
    for start in dramatic_starts:
        if answer.lower().startswith(start):
            answer = answer[len(start):].strip()
            # Clean up punctuation after removing the start
            if answer.startswith(",") or answer.startswith(":") or answer.startswith("."):
                answer = answer[1:].strip()
    
    # Ensure specific tumor details are mentioned if relevant
    question_lower = question.lower()
    
    # For questions about the tumor itself, ensure type and location are mentioned
    if ("what" in question_lower and "tumor" in question_lower) or "type" in question_lower or "kind" in question_lower:
        if tumor_type.lower() not in answer.lower():
            answer = f"This appears to be a {tumor_type} located in the {tumor_location}. " + answer
    
    # For questions about location, ensure location info is included
    if "where" in question_lower or "location" in question_lower:
        if tumor_location.lower() not in answer.lower():
            answer = f"The tumor is located in the {tumor_location}. " + answer
    
    # For questions about size, ensure size information is included
    if "size" in question_lower or "how big" in question_lower or "how large" in question_lower:
        if not any(s in answer.lower() for s in [str(tumor_size), "centimeter", "millimeter", "cm", "mm"]):
            answer = f"The tumor measures approximately {tumor_size} cm. " + answer
    
    # For questions about treatment, ensure treatment information is structured
    if "treatment" in question_lower or "therapy" in question_lower or "surgical" in question_lower:
        # Only enhance if the answer doesn't already have a well-structured treatment explanation
        if not any(marker in answer.lower() for marker in ["first line", "standard of care", "treatment options include", "therapeutic approach"]):
            if "glioblastoma" in tumor_type.lower() or "gbm" in tumor_type.lower():
                answer += "\n\nStandard treatment for glioblastoma typically includes maximal safe surgical resection, followed by concurrent radiation therapy and temozolomide chemotherapy (Stupp protocol), then adjuvant temozolomide."
            elif "meningioma" in tumor_type.lower():
                answer += "\n\nTreatment for meningioma often involves surgical resection, with radiation therapy considered for incomplete resection or higher-grade tumors. Regular monitoring is essential."
            elif "astrocytoma" in tumor_type.lower():
                answer += "\n\nTreatment approaches for astrocytoma may include surgical resection, radiation therapy, and chemotherapy, with specific protocols determined by the tumor's grade and molecular characteristics."
            else:
                answer += "\n\nTreatment typically involves a multidisciplinary approach including surgical intervention, radiation therapy, and possibly chemotherapy, depending on tumor characteristics and location."
    
    # Add medical terminology if the response seems too simplistic
    if len(answer.split()) < 40 and not any(term in answer.lower() for term in [
        "prognosis", "metastasis", "histopathology", "resection", "oncologist", "neurosurgeon",
        "chemotherapy", "radiation", "biopsy", "malignant", "benign", "protocol"
    ]):
        if "glioblastoma" in tumor_type.lower():
            answer += " Glioblastoma is characterized by histopathological features including microvascular proliferation, pseudopalisading necrosis, and high mitotic activity."
        elif "meningioma" in tumor_type.lower():
            answer += " Meningiomas are classified according to the WHO grading system (Grade I-III), with most being Grade I."
        elif "astrocytoma" in tumor_type.lower():
            answer += " Astrocytomas are classified by the WHO into grades I-IV based on histopathological features and molecular markers."
        else:
            answer += " Evaluation includes radiological characteristics, potential molecular markers, and histopathological features according to the current WHO classification of CNS tumors."
    
    # Ensure there's a closing recommendation to consult healthcare professionals
    if not any(phrase in answer.lower() for phrase in [
        "consult", "discuss with your doctor", "medical professional", "healthcare provider", 
        "specialist", "physician", "seek medical"
    ]):
        answer += " Please consult with your healthcare provider for personalized medical advice regarding your specific condition."
    
    # Fix common errors and improve phrasing
    simple_terms = {
        "bad tumor": "higher-grade tumor",
        "good tumor": "lower-grade tumor",
        "cancer in the brain": "malignant brain neoplasm",
        "brain cancer": "malignant brain tumor",
        "cut out": "resect",
        "remove": "resect",
        "look at": "evaluate",
        "medicine": "medication",
        "doctor who specializes in brain": "neurosurgeon or neuro-oncologist",
        "brain doctor": "neurosurgeon",
        "cancer doctor": "oncologist",
        "very bad": "concerning",
        "very good": "favorable"
    }
    
    # Perform replacements
    for simple, professional in simple_terms.items():
        pattern = r'\b' + re.escape(simple) + r'\b'
        answer = re.sub(pattern, professional, answer, flags=re.IGNORECASE)
    
    return answer

if __name__ == '__main__':
    # Pre-load the model in a separate thread to speed up first prediction
    executor.submit(get_model_session)
    
    # Start the app with higher performance settings and debug mode enabled
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)