# ID Card Quality Assessment

A Flask-based API to assess the quality of ID card images.

## Structure
- `app.py`: entry point
- `pipeline.py`: processing logic
- `modules/`: individual quality checks

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run app: `python app.py`
3. Send POST to `/assess` with an image file.
