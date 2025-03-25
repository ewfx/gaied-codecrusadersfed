import os
import email
import json
import re
import io
from datetime import datetime
from email.header import decode_header
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
from PIL import Image
import pdfplumber
import pytesseract
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

class EMLProcessor:
    def __init__(self):
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        openai.api_key = os.getenv('OPENAI_API_KEY')
    
    def process_eml(self, filepath):
        try:
            email_data = self._parse_eml(filepath)
            if not email_data:
                return {"error": "Failed to parse EML file"}, 400
            
            classification = self._classify_email(email_data)
            extracted_data = self._extract_fields(email_data, classification["primary_type"])
            routing = self._determine_routing(classification, extracted_data)
            
            return {
                "metadata": {
                    "sender": email_data["from"],
                    "subject": email_data["subject"],
                    "date": email_data["date"]
                },
                "classification": classification,
                "extracted_data": extracted_data,
                "routing": routing
            }, 200
        
        except Exception as e:
            return {"error": str(e)}, 500
    
    def _parse_eml(self, filepath):
        with open(filepath, "rb") as f:
            msg = email.message_from_binary_file(f)
            
            email_data = {
                "filename": os.path.basename(filepath),
                "subject": self._clean_subject(msg["Subject"]),
                "from": msg["From"],
                "date": msg["Date"],
                "body": "",
                "attachments": []
            }
            
            for part in msg.walk():
                content_type = part.get_content_type()
                disposition = str(part.get("Content-Disposition"))
                
                if "attachment" in disposition:
                    attachment = self._process_attachment(part)
                    if attachment:
                        email_data["attachments"].append(attachment)
                elif content_type == "text/plain":
                    body_part = part.get_payload(decode=True)
                    if body_part:
                        try:
                            email_data["body"] += body_part.decode()
                        except UnicodeDecodeError:
                            email_data["body"] += body_part.decode("latin-1")
            
            return email_data
    
    def _process_attachment(self, part):
        content_type = part.get_content_type()
        file_data = part.get_payload(decode=True)
        
        try:
            if content_type == "application/pdf":
                with pdfplumber.open(io.BytesIO(file_data)) as pdf:
                    return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
            elif content_type in ["image/jpeg", "image/png"]:
                image = Image.open(io.BytesIO(file_data))
                return pytesseract.image_to_string(image)
            elif content_type == "text/plain":
                return file_data.decode()
        except Exception as e:
            app.logger.error(f"Attachment processing failed: {str(e)}")
            return None
    
    def _clean_subject(self, subject):
        if subject is None:
            return ""
        decoded = decode_header(subject)
        return "".join(
            t[0].decode(t[1] if t[1] else "utf-8") if isinstance(t[0], bytes) else t[0]
            for t in decoded
        )
    
    def _classify_email(self, email_data):
        prompt = f"""
        Analyze this commercial banking email and classify it:
        - Primary type: Loan Modification, Payment Related, Documentation Request, General Inquiry
        - Sub-type: Appropriate sub-category
        - Reasoning: Brief explanation
        - Secondary requests: Any additional requests
        
        Return JSON format with keys: primary_type, sub_type, reasoning, secondary_requests
        
        Email:
        Subject: {email_data['subject']}
        From: {email_data['from']}
        Body: {email_data['body'][:2000]}
        Attachments: {len(email_data['attachments'])} found
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            app.logger.error(f"Classification failed: {str(e)}")
            return {
                "primary_type": "OTHERS",
                "sub_type": "OTHERS",
                "reasoning": "Please upload another file to process",
                "secondary_requests": []
            }
    
    def _extract_fields(self, email_data, request_type):
        patterns = {
            "loan_number": r"(?i)(loan|account)\s*(#|number|no\.?)\s*[:]?\s*(\w+)",
            "amount": r"\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?",
            "date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b"
        }
        
        return {
            "loan_number": self._extract_with_pattern(patterns["loan_number"], email_data),
            "amount": self._extract_amount(email_data, request_type, patterns["amount"]),
            "dates": self._extract_dates(email_data, patterns["date"]),
            "priority": self._determine_priority(email_data)
        }
    
    def _extract_with_pattern(self, pattern, email_data):
        match = re.search(pattern, email_data["subject"] + " " + email_data["body"])
        return match.group(3) if match and len(match.groups()) >= 3 else match.group(0) if match else None
    
    def _extract_amount(self, email_data, request_type, pattern):
        amounts = []
        if request_type == "PAYMENT":
            for attachment in email_data["attachments"]:
                amounts.extend(re.findall(pattern, attachment))
        
        if not amounts:
            amounts = re.findall(pattern, email_data["body"])
        
        return max(amounts, key=lambda x: float(x.replace("$", "").replace(",", ""))) if amounts else None
    
    def _extract_dates(self, email_data, pattern):
        dates = []
        for text in [email_data["body"]] + email_data["attachments"]:
            dates.extend(re.findall(pattern, text))
        
        parsed_dates = []
        for date_str in dates:
            try:
                parsed_dates.append(datetime.strptime(date_str, "%m/%d/%Y"))
            except ValueError:
                try:
                    parsed_dates.append(datetime.strptime(date_str, "%b %d, %Y"))
                except:
                    continue
        
        return sorted(parsed_dates)
    
    def _determine_priority(self, email_data):
        urgent_keywords = ["urgent", "immediate", "asap", "time sensitive"]
        content = f"{email_data['subject']} {email_data['body']}".lower()
        return "HIGH" if any(kw in content for kw in urgent_keywords) else "NORMAL"
    
    def _determine_routing(self, classification, extracted_data):
        routing_rules = {
            "LOAN_MOD": {
                "Extension": "LoanServicingTeam",
                "Rate Change": "UnderwritingTeam"
            },
            "PAYMENT": {
                "Missed Payment": "CollectionsTeam",
                "Payment Plan": "CustomerServiceTeam"
            },
            "DOCS": {"*": "DocumentationTeam"},
            "INQUIRY": {"*": "CustomerServiceTeam"}
        }
        
        primary_type = classification["primary_type"]
        sub_type = classification["sub_type"]
        
        recipient = routing_rules.get(primary_type, {}).get(sub_type) or \
                   routing_rules.get(primary_type, {}).get("*")
        
        if extracted_data.get("priority") == "HIGH":
            recipient = f"Urgent{recipient}"
        
        return {
            "assigned_to": recipient or "GeneralTeam",
            "priority": extracted_data.get("priority", "NORMAL")
        }

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'}), 400
    
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No selected files'}), 400
    
    saved_files = []
    for file in files:
        if file.filename == '':
            continue
        
        if file and file.filename.lower().endswith('.eml'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            saved_files.append(filename)
    
    return jsonify({
        'message': f'{len(saved_files)} files uploaded successfully',
        'files': saved_files
    })

@app.route('/process', methods=['POST'])
def process_files():
    processor = EMLProcessor()
    results = []
    
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.lower().endswith('.eml'):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            result, status_code = processor.process_eml(filepath)
            
            if status_code == 200:
                processed_path = os.path.join(PROCESSED_FOLDER, filename)
                os.rename(filepath, processed_path)
                results.append({
                    'filename': filename,
                    'result': result
                })
    
    return jsonify({
        'message': f'Processed {len(results)} files',
        'results': results
    })

@app.route('/processed/<filename>', methods=['GET'])
def get_processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)