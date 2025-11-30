from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import json
from utils.forensics import (
    analyze_ela, 
    analyze_noise_residual,
    analyze_frequency_residual,
    extract_metadata, 
    detect_forgery,
    generate_heatmap,
    generate_pdf_report
)
from utils.database import (
    init_db,
    save_analysis,
    get_all_history,
    get_history_by_id,
    delete_history_entry,
    update_report_path,
    find_history_by_filename
)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', 'dev-secret-key')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ANALYSIS_FOLDER'] = 'static/analysis'
app.config['REPORTS_FOLDER'] = 'static/reports'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'webp'}

for folder in [app.config['UPLOAD_FOLDER'], app.config['ANALYSIS_FOLDER'], app.config['REPORTS_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if not file.filename or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not supported. Please upload JPG, PNG, or WEBP'}), 400
    
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(file.filename or 'image.jpg')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        from PIL import Image
        img = Image.open(filepath)
        width, height = img.size
        format_type = img.format
        file_size = os.path.getsize(filepath)
        
        if width < 100 or height < 100:
            os.remove(filepath)
            return jsonify({'error': 'Image too small. Minimum size is 100x100 pixels'}), 400
        
        return jsonify({
            'success': True,
            'filename': unique_filename,
            'filepath': filepath,
            'details': {
                'width': width,
                'height': height,
                'format': format_type,
                'size': file_size
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/api/analyze/<filename>')
def analyze_image(filename):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404
        
        existing_record = find_history_by_filename(filename)
        
        if not existing_record:
            ela_result = analyze_ela(filepath, app.config['ANALYSIS_FOLDER'], filename)
            noise_result = analyze_noise_residual(filepath, app.config['ANALYSIS_FOLDER'], filename)
            freq_result = analyze_frequency_residual(filepath, app.config['ANALYSIS_FOLDER'], filename)
            heatmap_result = generate_heatmap(filepath, app.config['ANALYSIS_FOLDER'], filename)
            forgery_result = detect_forgery(
                filepath,
                app.config['ANALYSIS_FOLDER'],
                filename,
                ela_result=ela_result,
                noise_result=noise_result,
                freq_result=freq_result
            )
            
            def normalize_path(path):
                if path and not path.startswith('/'):
                    return '/' + path
                return path
            
            forgery_normalized = forgery_result.copy()
            ela_normalized = {k: normalize_path(v) if isinstance(v, str) else v for k, v in ela_result.items()}
            noise_normalized = {k: normalize_path(v) if isinstance(v, str) else v for k, v in noise_result.items()}
            freq_normalized = {k: normalize_path(v) if isinstance(v, str) else v for k, v in freq_result.items()}
            heatmap_normalized = {k: normalize_path(v) if isinstance(v, str) else v for k, v in heatmap_result.items()}
            
            save_analysis(
                filename,
                normalize_path(filepath),
                forgery_normalized,
                heatmap_normalized,
                ela_normalized,
                noise_normalized,
                freq_normalized
            )
        else:
            forgery_result = {
                'verdict': existing_record['verdict'],
                'confidence': existing_record['confidence'],
                'color': existing_record['color'],
                'ela_score': existing_record['ela_score'],
                'noise_score': existing_record['noise_score'],
                'freq_score': existing_record.get('freq_score', 0)
            }
            ela_result = {'image_path': existing_record['ela_map_path']}
            noise_result = {'image_path': existing_record['noise_map_path']}
            freq_result = {'image_path': existing_record.get('freq_map_path', '')}
            heatmap_result = {
                'heatmap_path': existing_record['heatmap_path'],
                'overlay_path': existing_record['overlay_path']
            }
        
        return jsonify({
            'success': True,
            'forgery': forgery_result,
            'ela': ela_result,
            'noise': noise_result,
            'frequency': freq_result,
            'heatmap': heatmap_result
        })
    
    except Exception as e:
        return jsonify({'error': f'Analysis error: {str(e)}'}), 500

@app.route('/analysis/<filename>')
def analysis_page(filename):
    return render_template('analysis.html', filename=filename)

@app.route('/metadata/<filename>')
def metadata_page(filename):
    return render_template('metadata.html', filename=filename)

@app.route('/api/metadata/<filename>')
def get_metadata(filename):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404
        
        metadata = extract_metadata(filepath)
        
        return jsonify({
            'success': True,
            'metadata': metadata
        })
    
    except Exception as e:
        return jsonify({'error': f'Metadata extraction error: {str(e)}'}), 500

@app.route('/api/generate-report/<filename>')
def generate_report(filename):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404
        
        forgery_result = detect_forgery(filepath, app.config['ANALYSIS_FOLDER'], filename)
        analyze_ela(filepath, app.config['ANALYSIS_FOLDER'], filename)
        analyze_noise_residual(filepath, app.config['ANALYSIS_FOLDER'], filename)
        generate_heatmap(filepath, app.config['ANALYSIS_FOLDER'], filename)
        metadata = extract_metadata(filepath)
        
        report_path = generate_pdf_report(
            filepath,
            filename,
            forgery_result,
            metadata,
            app.config['ANALYSIS_FOLDER'],
            app.config['REPORTS_FOLDER']
        )
        
        download_url = url_for('download_report', filename=os.path.basename(report_path))
        
        history_record = find_history_by_filename(filename)
        if history_record:
            update_report_path(history_record['id'], download_url)
        
        return jsonify({
            'success': True,
            'report_path': report_path,
            'download_url': download_url
        })
    
    except Exception as e:
        return jsonify({'error': f'Report generation error: {str(e)}'}), 500

@app.route('/download-report/<filename>')
def download_report(filename):
    filepath = os.path.join(app.config['REPORTS_FOLDER'], filename)
    return send_file(filepath, as_attachment=True)

@app.route('/history')
def history_page():
    return render_template('history.html')

@app.route('/api/history')
def get_history():
    try:
        history = get_all_history()
        return jsonify({
            'success': True,
            'history': history
        })
    except Exception as e:
        return jsonify({'error': f'Error fetching history: {str(e)}'}), 500

@app.route('/api/history/delete/<int:history_id>', methods=['DELETE'])
def delete_history(history_id):
    try:
        success = delete_history_entry(history_id)
        if success:
            return jsonify({'success': True, 'message': 'Entry deleted'})
        else:
            return jsonify({'error': 'Entry not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Delete error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
