import sqlite3
import os
from datetime import datetime
import json

DATABASE_PATH = 'analysis_history.db'

def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_path TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            verdict TEXT,
            confidence REAL,
            color TEXT,
            ela_score REAL,
            noise_score REAL,
            freq_score REAL DEFAULT 0,
            heatmap_path TEXT,
            overlay_path TEXT,
            ela_map_path TEXT,
            noise_map_path TEXT,
            freq_map_path TEXT,
            report_path TEXT
        )
    ''')

    cursor.execute("PRAGMA table_info(analysis_history)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    if 'freq_score' not in existing_columns:
        cursor.execute('ALTER TABLE analysis_history ADD COLUMN freq_score REAL DEFAULT 0')
    if 'freq_map_path' not in existing_columns:
        cursor.execute("ALTER TABLE analysis_history ADD COLUMN freq_map_path TEXT DEFAULT ''")

    conn.commit()
    conn.close()

def save_analysis(filename, original_path, forgery_result, heatmap_result, ela_result, noise_result, freq_result=None, report_path=None):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO analysis_history 
        (filename, original_path, verdict, confidence, color, ela_score, noise_score, freq_score,
         heatmap_path, overlay_path, ela_map_path, noise_map_path, freq_map_path, report_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        filename,
        original_path,
        forgery_result.get('verdict', 'Unknown'),
        forgery_result.get('confidence', 0),
        forgery_result.get('color', 'gray'),
        forgery_result.get('ela_score', 0),
        forgery_result.get('noise_score', 0),
        forgery_result.get('freq_score', 0),
        heatmap_result.get('heatmap_path', '') if heatmap_result else '',
        heatmap_result.get('overlay_path', '') if heatmap_result else '',
        ela_result.get('image_path', '') if ela_result else '',
        noise_result.get('image_path', '') if noise_result else '',
        freq_result.get('image_path', '') if freq_result else '',
        report_path
    ))
    
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    
    return record_id

def get_all_history():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM analysis_history 
        ORDER BY timestamp DESC
    ''')
    
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            'id': row['id'],
            'filename': row['filename'],
            'original_path': row['original_path'],
            'timestamp': row['timestamp'],
            'verdict': row['verdict'],
            'confidence': row['confidence'],
            'color': row['color'],
            'ela_score': row['ela_score'],
            'noise_score': row['noise_score'],
            'freq_score': row['freq_score'],
            'heatmap_path': row['heatmap_path'],
            'overlay_path': row['overlay_path'],
            'ela_map_path': row['ela_map_path'],
            'noise_map_path': row['noise_map_path'],
            'freq_map_path': row['freq_map_path'],
            'report_path': row['report_path']
        })
    
    return history

def get_history_by_id(history_id):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM analysis_history WHERE id = ?', (history_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'id': row['id'],
            'filename': row['filename'],
            'original_path': row['original_path'],
            'timestamp': row['timestamp'],
            'verdict': row['verdict'],
            'confidence': row['confidence'],
            'color': row['color'],
            'ela_score': row['ela_score'],
            'noise_score': row['noise_score'],
            'freq_score': row['freq_score'],
            'heatmap_path': row['heatmap_path'],
            'overlay_path': row['overlay_path'],
            'ela_map_path': row['ela_map_path'],
            'noise_map_path': row['noise_map_path'],
            'freq_map_path': row['freq_map_path'],
            'report_path': row['report_path']
        }
    return None

def delete_history_entry(history_id):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM analysis_history WHERE id = ?', (history_id,))
    row = cursor.fetchone()
    
    if row:
        cursor.execute('DELETE FROM analysis_history WHERE id = ?', (history_id,))
        conn.commit()
        conn.close()
        return True
    
    conn.close()
    return False

def update_report_path(history_id, report_path):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE analysis_history 
        SET report_path = ?
        WHERE id = ?
    ''', (report_path, history_id))
    
    conn.commit()
    conn.close()

def find_history_by_filename(filename):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM analysis_history 
        WHERE filename = ?
        ORDER BY timestamp DESC
        LIMIT 1
    ''', (filename,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'id': row['id'],
            'filename': row['filename'],
            'original_path': row['original_path'],
            'timestamp': row['timestamp'],
            'verdict': row['verdict'],
            'confidence': row['confidence'],
            'color': row['color'],
            'ela_score': row['ela_score'],
            'noise_score': row['noise_score'],
            'freq_score': row['freq_score'],
            'heatmap_path': row['heatmap_path'],
            'overlay_path': row['overlay_path'],
            'ela_map_path': row['ela_map_path'],
            'noise_map_path': row['noise_map_path'],
            'freq_map_path': row['freq_map_path'],
            'report_path': row['report_path']
        }
    return None
