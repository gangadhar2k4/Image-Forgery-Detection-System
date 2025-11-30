import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import exifread
import os
import math
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import filters, exposure


def _get_exif_tag(tags, *possible_keys):
    for key in possible_keys:
        value = tags.get(key)
        if value:
            return value
    return None


def _ratio_to_float(value):
    try:
        return float(value.num) / float(value.den)
    except Exception:
        try:
            return float(value)
        except Exception:
            return 0.0


def _convert_gps_to_decimal(tag_value, ref_value):
    if not tag_value or ref_value is None:
        return None
    try:
        values = getattr(tag_value, 'values', tag_value)
        if not isinstance(values, (list, tuple)):
            values = [values]
        degrees = _ratio_to_float(values[0]) if len(values) > 0 else 0
        minutes = _ratio_to_float(values[1]) if len(values) > 1 else 0
        seconds = _ratio_to_float(values[2]) if len(values) > 2 else 0
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        ref = str(ref_value).upper()
        if ref in ['S', 'W']:
            decimal = -decimal
        return round(decimal, 6)
    except Exception:
        return None


def _normalize_capture_conditions(img):
    if img is None:
        raise ValueError("Image not provided for normalization")
    normalized = img.copy()
    height, width = normalized.shape[:2]
    max_dim = 1600
    if max(height, width) > max_dim:
        scale = max_dim / float(max(height, width))
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        normalized = cv2.resize(normalized, new_size, interpolation=cv2.INTER_AREA)
    normalized = cv2.bilateralFilter(normalized, d=7, sigmaColor=55, sigmaSpace=55)
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _load_preprocessed_bgr(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image")
    normalized = _normalize_capture_conditions(img)
    return normalized, img.shape[:2]


def _load_preprocessed_gray(image_path):
    color_img, original_shape = _load_preprocessed_bgr(image_path)
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray_img, original_shape


def _load_preprocessed_pil(image_path):
    color_img, _ = _load_preprocessed_bgr(image_path)
    rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)


def _calculate_image_complexity(image_path):
    gray_img, original_shape = _load_preprocessed_gray(image_path)
    texture_variance = float(cv2.Laplacian(gray_img, cv2.CV_64F).var())
    contrast = float(np.std(gray_img))
    brightness = float(np.mean(gray_img))
    height, width = original_shape
    size_factor = (height * width) / (512 * 512)
    size_factor = max(0.5, min(size_factor, 4.0))
    return {
        'texture': texture_variance,
        'contrast': contrast,
        'size_factor': size_factor,
        'brightness': brightness
    }


def _edge_inconsistency_score(image_path):
    try:
        img, _ = _load_preprocessed_gray(image_path)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobelx, sobely)
        blurred = cv2.GaussianBlur(magnitude, (7, 7), 0)
        inconsistency = cv2.absdiff(magnitude, blurred)
        edge_energy = float(np.mean(magnitude))
        anomaly_energy = float(np.mean(inconsistency))
        if edge_energy == 0:
            return 0.0
        score = (anomaly_energy / edge_energy)
        score = float(max(0.0, min(1.5, score * 0.8)))
        brightness = float(np.mean(img))
        if brightness > 190:
            score *= 0.85
        return score
    except Exception:
        return 0.0


def _regional_texture_divergence(image_path):
    try:
        gray, _ = _load_preprocessed_gray(image_path)
        height, width = gray.shape
        if min(height, width) < 64:
            return 0.0
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = cv2.GaussianBlur(laplacian, (3, 3), 0)
        h_start, h_end = height // 4, height - height // 4
        w_start, w_end = width // 4, width - width // 4
        center_patch = laplacian[h_start:h_end, w_start:w_end]
        if center_patch.size < 100:
            return 0.0
        center_var = float(center_patch.var())
        border_mask = np.ones_like(gray, dtype=bool)
        border_mask[h_start:h_end, w_start:w_end] = False
        border_values = laplacian[border_mask]
        if border_values.size < 100:
            return 0.0
        border_var = float(np.var(border_values))
        baseline = max(1e-5, max(center_var, border_var))
        divergence = abs(center_var - border_var) / baseline
        divergence /= (1.0 + (baseline / 1500.0))
        return float(max(0.0, min(1.2, divergence)))
    except Exception:
        return 0.0


def _chromatic_shift_score(image_path):
    try:
        img, _ = _load_preprocessed_bgr(image_path)
        height, width = img.shape[:2]
        if min(height, width) < 64:
            return 0.0
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        h_start, h_end = height // 4, height - height // 4
        w_start, w_end = width // 4, width - width // 4
        center_patch = lab[h_start:h_end, w_start:w_end]
        border_mask = np.ones((height, width), dtype=bool)
        border_mask[h_start:h_end, w_start:w_end] = False
        border_values = lab.reshape(-1, 3)[border_mask.reshape(-1)]
        if border_values.size < 300:
            return 0.0
        center_mean = np.mean(center_patch.reshape(-1, 3), axis=0)
        border_mean = np.mean(border_values.reshape(-1, 3), axis=0)
        luminance_shift = abs(center_mean[0] - border_mean[0])
        chroma_shift = np.linalg.norm(center_mean[1:] - border_mean[1:])
        global_chroma = lab[:, :, 1:3].reshape(-1, 2)
        chroma_std = float(np.mean(np.std(global_chroma, axis=0)))
        normalization = 35.0 + min(25.0, chroma_std)
        combined_shift = (0.6 * luminance_shift + 0.4 * chroma_shift) / normalization
        return float(max(0.0, min(1.2, combined_shift)))
    except Exception:
        return 0.0


def _normalized_score(value, baseline, scale):
    if value is None:
        return 0.0
    adjusted_scale = max(scale, 1e-3)
    raw_score = (value - baseline) / adjusted_scale
    return float(max(0.0, min(1.0, raw_score)))

def analyze_ela(image_path, output_folder, filename):
    try:
        quality = 90
        scale = 15
        
        original = _load_preprocessed_pil(image_path)
        
        temp_path = os.path.join(output_folder, f'temp_{filename}')
        original.save(temp_path, 'JPEG', quality=quality)
        
        compressed = Image.open(temp_path)
        
        ela_image = ImageChops.difference(original, compressed)
        
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        
        if max_diff == 0:
            max_diff = 1
        
        scale_factor = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale_factor * scale)
        
        ela_output_path = os.path.join(output_folder, f'ela_{filename}')
        ela_image.save(ela_output_path)
        
        os.remove(temp_path)
        
        ela_array = np.array(ela_image)
        avg_ela_value = np.mean(ela_array)
        
        return {
            'image_path': f'/static/analysis/ela_{filename}',
            'avg_value': float(avg_ela_value),
            'description': 'Error Level Analysis shows compression inconsistencies. Bright areas may indicate tampering.'
        }
    
    except Exception as e:
        return {'error': str(e)}

def analyze_noise_residual(image_path, output_folder, filename):
    try:
        img, _ = _load_preprocessed_bgr(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        noise = cv2.subtract(gray, blurred)
        
        noise_enhanced = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX)
        
        noise_color = cv2.applyColorMap(noise_enhanced, cv2.COLORMAP_JET)
        
        noise_output_path = os.path.join(output_folder, f'noise_{filename}')
        cv2.imwrite(noise_output_path, noise_color)
        
        noise_std = np.std(noise)
        
        return {
            'image_path': f'/static/analysis/noise_{filename}',
            'std_value': float(noise_std),
            'description': 'Noise residual analysis reveals inconsistencies in noise patterns across the image.'
        }
    
    except Exception as e:
        return {'error': str(e)}

def analyze_frequency_residual(image_path, output_folder, filename):
    try:
        img, _ = _load_preprocessed_gray(image_path)

        target_size = (512, 512)
        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        block_size = 32
        blocks_y = resized.shape[0] // block_size
        blocks_x = resized.shape[1] // block_size
        freq_map = np.zeros((blocks_y, blocks_x), dtype=np.float32)

        for by in range(blocks_y):
            for bx in range(blocks_x):
                block = resized[by * block_size:(by + 1) * block_size, bx * block_size:(bx + 1) * block_size]
                dct = cv2.dct(np.float32(block))
                total_energy = np.sum(np.abs(dct))
                if total_energy == 0:
                    freq_map[by, bx] = 0
                    continue

                cutoff = block_size // 3
                high_freq_energy = np.sum(np.abs(dct[cutoff:, cutoff:]))
                freq_map[by, bx] = high_freq_energy / total_energy

        upsampled = cv2.resize(freq_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        normalized = cv2.normalize(upsampled, None, 0, 255, cv2.NORM_MINMAX)
        colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_TURBO)

        freq_output_path = os.path.join(output_folder, f'freq_{filename}')
        cv2.imwrite(freq_output_path, colored)

        energy_ratio = float(np.mean(freq_map))

        return {
            'image_path': f'/static/analysis/freq_{filename}',
            'energy_ratio': energy_ratio,
            'description': 'Frequency residual analysis highlights abnormal high-frequency energy commonly introduced during compositing or copy-move operations.'
        }

    except Exception as e:
        return {'error': str(e)}

def generate_heatmap(image_path, output_folder, filename):
    try:
        img = cv2.imread(image_path)
        
        if img is None:
            raise ValueError("Could not load image")
        
        normalized = _normalize_capture_conditions(img)
        gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, 50, 150)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.subtract(gray, blurred)
        
        combined = cv2.addWeighted(edges.astype(float), 0.5, noise.astype(float), 0.5, 0)
        combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        heatmap = cv2.applyColorMap(combined, cv2.COLORMAP_HOT)
        if heatmap.shape[:2] != img.shape[:2]:
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        alpha = 0.6
        overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
        
        heatmap_output_path = os.path.join(output_folder, f'heatmap_{filename}')
        overlay_output_path = os.path.join(output_folder, f'overlay_{filename}')
        
        cv2.imwrite(heatmap_output_path, heatmap)
        cv2.imwrite(overlay_output_path, overlay)
        
        return {
            'heatmap_path': f'/static/analysis/heatmap_{filename}',
            'overlay_path': f'/static/analysis/overlay_{filename}',
            'description': 'Heatmap highlights potential tampered regions based on edge and noise analysis.'
        }
    
    except Exception as e:
        return {'error': str(e)}

def detect_forgery(image_path, output_folder, filename, ela_result=None, noise_result=None, freq_result=None):
    try:
        ela_result = ela_result or analyze_ela(image_path, output_folder, filename)
        noise_result = noise_result or analyze_noise_residual(image_path, output_folder, filename)
        freq_result = freq_result or analyze_frequency_residual(image_path, output_folder, filename)

        try:
            complexity = _calculate_image_complexity(image_path)
        except Exception:
            complexity = {'texture': 800.0, 'contrast': 30.0, 'size_factor': 1.0}

        texture_factor = max(1.0, complexity['texture'])
        contrast = max(1.0, complexity['contrast'])
        size_factor = max(0.5, complexity['size_factor'])
        brightness = complexity.get('brightness', 128.0)
        exposure_bias = max(0.0, (brightness - 180.0) / 75.0)
        low_light_bias = max(0.0, (100.0 - brightness) / 100.0)
        texture_raw = min(1.0, _regional_texture_divergence(image_path))
        texture_relax = 0.6 if texture_factor > 2000 else (0.75 if texture_factor > 1400 else 1.0)
        texture_divergence = min(1.0, texture_raw * texture_relax)
        chroma_raw = min(1.0, _chromatic_shift_score(image_path))
        chroma_relax = 1.0 - min(0.35, exposure_bias * 0.5)
        chromatic_shift = min(1.0, chroma_raw * max(0.6, chroma_relax))

        ela_score = _normalized_score(
            ela_result.get('avg_value'),
            baseline=18 + min(22, math.log1p(texture_factor) * 3.2) + exposure_bias * 12,
            scale=35 + min(25, texture_factor * 0.002) + exposure_bias * 10
        )
        noise_score = _normalized_score(
            noise_result.get('std_value'),
            baseline=8 + min(14, contrast * 0.12) + exposure_bias * 6,
            scale=18 + min(15, contrast * 0.08) + exposure_bias * 5
        )
        freq_score = _normalized_score(
            (freq_result or {}).get('energy_ratio'),
            baseline=0.08 * size_factor,
            scale=0.12 + 0.05 * size_factor
        )
        edge_score = min(1.0, _edge_inconsistency_score(image_path))
        if brightness > 200:
            edge_score *= 0.9

        metric_scores = np.array([
            ela_score,
            noise_score,
            freq_score,
            edge_score,
            texture_divergence,
            chromatic_shift
        ], dtype=np.float32)
        weights = np.array([0.28, 0.22, 0.16, 0.14, 0.12, 0.08], dtype=np.float32)
        if freq_score > 0.55:
            weights[2] += 0.05
        if edge_score > 0.6:
            weights[3] += 0.05
        if texture_divergence > 0.6:
            weights[4] += 0.05
        if chromatic_shift > 0.5:
            weights[5] += 0.04
        weights = weights / np.sum(weights)

        combined_score = float(np.dot(metric_scores, weights))
        variability = float(np.std(metric_scores))
        cooperation_bonus = min(0.12, variability * 0.2)
        high_ela_noise = 0.08 if (ela_score > 0.6 and noise_score > 0.45 and brightness < 190) else 0.0
        exposure_penalty = min(0.12, max(0.0, (brightness - 205.0) / 50.0))
        low_light_bonus = min(0.05, low_light_bias * 0.05)
        strong_indicators = int(np.sum(metric_scores > 0.55))
        consensus_penalty = 0.08 if strong_indicators < 2 else 0.0
        adjusted_score = combined_score + cooperation_bonus + high_ela_noise - exposure_penalty + low_light_bonus - consensus_penalty
        adjusted_score = max(0.0, min(1.0, adjusted_score))
        confidence = round(adjusted_score * 100, 2)

        if confidence < 50:
            verdict = "Likely Authentic"
            color = "green"
        elif confidence < 75:
            verdict = "Suspicious - Further Analysis Recommended"
            color = "orange"
        else:
            verdict = "Likely Tampered"
            color = "red"

        insights = []

        def _add_insight(level, title, detail):
            insights.append({
                'level': level,
                'title': title,
                'detail': detail
            })

        if ela_score > 0.65:
            _add_insight(
                'alert',
                'ELA anomaly detected',
                'Compression differences are concentrated in specific regions, commonly caused by copy-move or splice edits.'
            )
        elif ela_score < 0.2:
            _add_insight(
                'info',
                'Uniform compression',
                'ELA response is low, indicating consistent compression — often seen in camera-original assets.'
            )

        if noise_score > 0.55:
            _add_insight(
                'warning',
                'Noise residue spike',
                'Localized noise residuals suggest certain areas were edited or re-saved at different qualities.'
            )
        elif noise_score < 0.25:
            _add_insight(
                'info',
                'Balanced noise texture',
                'Noise variance is minimal and evenly distributed, matching typical in-camera signatures.'
            )

        if freq_score > 0.5:
            _add_insight(
                'warning',
                'Elevated high-frequency energy',
                'Spectral energy spikes indicate sharp cutouts or pasted objects introduced during compositing.'
            )

        if edge_score > 0.65:
            _add_insight(
                'warning',
                'Irregular edge transitions',
                'Edge field irregularities often arise when elements are blended or feathered post-capture.'
            )

        if texture_divergence > 0.55:
            _add_insight(
                'warning',
                'Texture mismatch detected',
                'Central textures differ sharply from surrounding regions, common in head-swaps or animal morphs.'
            )

        if chromatic_shift > 0.45:
            _add_insight(
                'warning',
                'Color grading inconsistency',
                'Foreground hues exhibit different tonal balance than the rest of the scene, suggesting compositing.'
            )

        if strong_indicators < 2:
            _add_insight(
                'info',
                'Low anomaly consensus',
                'Only one detector raised concerns; system down-weights isolated signals to avoid false positives.'
            )

        if complexity['texture'] < 400:
            _add_insight(
                'info',
                'Low texture surface',
                'Smooth, low-texture scenes can mask tampering; rely more on metadata and contextual review.'
            )
        elif complexity['texture'] > 1600:
            _add_insight(
                'info',
                'High texture environment',
                'Detailed scenes amplify forensic signals, increasing confidence in anomaly localization.'
            )

        if brightness > 200:
            _add_insight(
                'info',
                'High exposure scene',
                'Strong highlights detected. Exposure normalization applied to avoid confusing bright regions with tampering.'
            )
        elif brightness < 80:
            _add_insight(
                'info',
                'Low-light capture',
                'Scene captured in dim conditions. Noise thresholds relaxed to avoid false alarms.'
            )

        if confidence >= 75:
            _add_insight(
                'alert',
                'High-risk signature',
                'Multiple forensic channels agree on tampering indicators. Prioritize manual review.'
            )
        elif 50 <= confidence < 75:
            _add_insight(
                'warning',
                'Borderline authenticity',
                'Signals are mixed; corroborate with metadata, source files, or ground-truth references.'
            )
        else:
            _add_insight(
                'info',
                'Confidence low',
                'Detectors show minimal anomalies. Archived originals still recommended for mission-critical evidence.'
            )

        return {
            'is_fake': confidence >= 75,
            'confidence': confidence,
            'verdict': verdict,
            'color': color,
            'ela_score': round(ela_score * 100, 2),
            'noise_score': round(noise_score * 100, 2),
            'freq_score': round(freq_score * 100, 2),
            'edge_score': round(edge_score * 100, 2),
            'complexity': complexity,
            'insights': insights
        }

    except Exception as e:
        return {'error': str(e)}

def extract_metadata(image_path):
    metadata = {
        'basic': {},
        'camera': {},
        'gps': {},
        'timestamps': {},
        'software': {},
        'warnings': []
    }
    
    try:
        warnings = []
        
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
        except Exception:
            tags = {}
        
        if not tags or len(tags) == 0:
            warnings.append({
                'type': 'warning',
                'message': 'No EXIF metadata found - may have been removed'
            })
        
        try:
            img = Image.open(image_path)
            metadata['basic'] = {
                'filename': os.path.basename(image_path),
                'format': str(img.format) if img.format else 'Unknown',
                'mode': str(img.mode) if img.mode else 'Unknown',
                'size': f"{img.size[0]} x {img.size[1]}" if img.size else 'Unknown',
                'file_size': f"{os.path.getsize(image_path) / 1024:.2f} KB"
            }
        except Exception:
            metadata['basic'] = {
                'filename': os.path.basename(image_path),
                'format': 'Unknown',
                'mode': 'Unknown',
                'size': 'Unknown',
                'file_size': f"{os.path.getsize(image_path) / 1024:.2f} KB"
            }
        
        if tags:
            try:
                camera_make = tags.get('Image Make', None)
                camera_model = tags.get('Image Model', None)
                
                if camera_make:
                    metadata['camera']['make'] = str(camera_make)
                if camera_model:
                    metadata['camera']['model'] = str(camera_model)
            except Exception:
                pass
            
            try:
                software = _get_exif_tag(tags, 'Image Software', 'EXIF Software')
                if software:
                    software_str = str(software)
                    metadata['software']['editing_software'] = software_str
                    if any(editor in software_str.lower() for editor in ['photoshop', 'gimp', 'paint.net', 'lightroom', 'canva', 'pixlr']):
                        warnings.append({
                            'type': 'alert',
                            'message': f'Image edited using: {software_str}'
                        })
            except Exception:
                pass
            
            try:
                datetime_original = tags.get('EXIF DateTimeOriginal', None)
                datetime_digitized = tags.get('EXIF DateTimeDigitized', None)
                
                if datetime_original:
                    metadata['timestamps']['original'] = str(datetime_original)
                if datetime_digitized:
                    metadata['timestamps']['digitized'] = str(datetime_digitized)
            except Exception:
                pass
            
            try:
                gps_lat = _get_exif_tag(tags, 'GPS GPSLatitude')
                gps_lat_ref = _get_exif_tag(tags, 'GPS GPSLatitudeRef')
                gps_lon = _get_exif_tag(tags, 'GPS GPSLongitude')
                gps_lon_ref = _get_exif_tag(tags, 'GPS GPSLongitudeRef')
                if gps_lat and gps_lon:
                    lat_decimal = _convert_gps_to_decimal(gps_lat, gps_lat_ref)
                    lon_decimal = _convert_gps_to_decimal(gps_lon, gps_lon_ref)
                    if lat_decimal is not None and lon_decimal is not None:
                        metadata['gps']['latitude'] = lat_decimal
                        metadata['gps']['longitude'] = lon_decimal
                        metadata['gps']['coordinates'] = f"{lat_decimal}, {lon_decimal}"
            except Exception:
                pass
        
        if not metadata['camera']:
            warnings.append({
                'type': 'info',
                'message': 'No camera information found'
            })
        
        metadata['warnings'] = warnings
        
        return metadata
    
    except Exception as e:
        metadata['warnings'].append({
            'type': 'alert',
            'message': f'Error extracting metadata: {str(e)}'
        })
        return metadata

def generate_pdf_report(image_path, filename, forgery_result, metadata, analysis_folder, reports_folder):
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f'report_{timestamp}_{filename}.pdf'
        report_path = os.path.join(reports_folder, report_filename)
        
        doc = SimpleDocTemplate(report_path, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor='#2c3e50',
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        title = Paragraph("Image Forgery Detection Report", title_style)
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        
        timestamp_text = Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
        story.append(timestamp_text)
        story.append(Spacer(1, 0.2*inch))
        
        filename_text = Paragraph(f"<b>Image File:</b> {filename}", styles['Normal'])
        story.append(filename_text)
        story.append(Spacer(1, 0.3*inch))
        
        verdict_style = ParagraphStyle(
            'Verdict',
            parent=styles['Heading2'],
            fontSize=18,
            textColor=forgery_result.get('color', 'black'),
            spaceAfter=20
        )
        verdict = Paragraph(f"<b>Verdict:</b> {forgery_result.get('verdict', 'Unknown')}", verdict_style)
        story.append(verdict)
        
        confidence_text = Paragraph(f"<b>Confidence Score:</b> {forgery_result.get('confidence', 0)}%", styles['Normal'])
        story.append(confidence_text)
        story.append(Spacer(1, 0.3*inch))
        
        if os.path.exists(image_path):
            story.append(Paragraph("<b>Original Image:</b>", styles['Heading3']))
            story.append(Spacer(1, 0.1*inch))
            img = RLImage(image_path, width=4*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph("<b>Forensic Analysis:</b>", styles['Heading3']))
        story.append(Spacer(1, 0.1*inch))
        
        ela_text = Paragraph(f"ELA Score: {forgery_result.get('ela_score', 0)}%", styles['Normal'])
        story.append(ela_text)
        
        noise_text = Paragraph(f"Noise Analysis Score: {forgery_result.get('noise_score', 0)}%", styles['Normal'])
        story.append(noise_text)
        story.append(Spacer(1, 0.3*inch))
        
        heatmap_path = os.path.join(analysis_folder, f'heatmap_{filename}')
        if os.path.exists(heatmap_path):
            story.append(Paragraph("<b>Tampering Heatmap:</b>", styles['Heading3']))
            story.append(Spacer(1, 0.1*inch))
            heatmap_img = RLImage(heatmap_path, width=4*inch, height=3*inch)
            story.append(heatmap_img)
            story.append(Spacer(1, 0.3*inch))
        
        freq_map_path = os.path.join(analysis_folder, f'freq_{filename}')
        if os.path.exists(freq_map_path):
            story.append(Paragraph("<b>Frequency Residual Map:</b>", styles['Heading3']))
            story.append(Spacer(1, 0.1*inch))
            freq_img = RLImage(freq_map_path, width=4*inch, height=3*inch)
            story.append(freq_img)
            story.append(Spacer(1, 0.3*inch))

        story.append(PageBreak())
        story.append(Paragraph("<b>Metadata Analysis:</b>", styles['Heading3']))
        story.append(Spacer(1, 0.1*inch))
        
        if metadata.get('camera'):
            camera_info = metadata['camera']
            if 'make' in camera_info:
                story.append(Paragraph(f"Camera Make: {camera_info['make']}", styles['Normal']))
            if 'model' in camera_info:
                story.append(Paragraph(f"Camera Model: {camera_info['model']}", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        
        if metadata.get('timestamps'):
            ts_info = metadata['timestamps']
            if 'original' in ts_info:
                story.append(Paragraph(f"Date Taken: {ts_info['original']}", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        
        if metadata.get('warnings'):
            story.append(Paragraph("<b>Warnings:</b>", styles['Heading3']))
            for warning in metadata['warnings']:
                warning_text = Paragraph(f"• {warning['message']}", styles['Normal'])
                story.append(warning_text)
        
        doc.build(story)
        
        return report_path
    
    except Exception as e:
        raise Exception(f"PDF generation failed: {str(e)}")
