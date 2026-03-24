import sqlite3
import os

DB_PATH = 'qa_assessments.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            filename TEXT,
            decision TEXT,
            quality_score REAL,
            pass_resolution BOOLEAN,
            pass_card BOOLEAN,
            pass_face BOOLEAN,
            pass_blur BOOLEAN,
            pass_glare BOOLEAN,
            pass_noise BOOLEAN,
            pass_exposure BOOLEAN,
            pass_geometry BOOLEAN
        )
    ''')
    # Use REPLACE to allow the user to resubmit/toggle feedback securely.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assessment_id INTEGER,
            attribute TEXT,
            is_wrong BOOLEAN,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(assessment_id) REFERENCES assessments(id),
            UNIQUE(assessment_id, attribute)
        )
    ''')
    conn.commit()
    conn.close()

def log_assessment(data):
    """
    Records an assessment to the SQLite database.
    data: Dictionary with metrics and decisions.
    Returns: The ID of the inserted row.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    query = '''
        INSERT INTO assessments (
            filename, decision, quality_score, 
            pass_resolution, pass_card, pass_face, pass_blur, 
            pass_glare, pass_noise, pass_exposure, pass_geometry
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''
    cursor.execute(query, (
        data.get('filename'), data.get('decision'), data.get('quality_score'),
        data.get('pass_resolution'), data.get('pass_card'), 
        data.get('pass_face'), data.get('pass_blur'), 
        data.get('pass_glare'), data.get('pass_noise'), 
        data.get('pass_exposure'), data.get('pass_geometry')
    ))
    assessment_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return assessment_id

def log_feedback(assessment_id, attribute, is_wrong=True):
    """
    Records specific attribute feedback for a given assessment.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO feedback (assessment_id, attribute, is_wrong)
        VALUES (?, ?, ?)
    ''', (assessment_id, attribute, is_wrong))
    conn.commit()
    conn.close()
