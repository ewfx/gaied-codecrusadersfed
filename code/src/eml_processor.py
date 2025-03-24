import os
import email
import json
import re
import io
import logging
from datetime import datetime
from email.header import decode_header
from PIL import Image
import pdfplumber
import pytesseract
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
class Config:
    EML_DIRECTORY = "../../eml_files"
    OUTPUT_DIR = "../../processed_requests"
    OPENAI_API_KEY = "your-api-key"  # Replace with your actual key
    CLASSIFICATION_MODEL = "gpt-3.5-turbo"
    SIMILARITY_THRESHOLD = 0.9
    
    REQUEST_TYPES = {
        "LOAN_MOD": "Loan Modification",
        "PAYMENT": "Payment Related",
        "DOCS": "Documentation Request",
        "INQUIRY": "General Inquiry"
    }
    
    SUB_TYPES = {
        "LOAN_MOD": ["Extension", "Rate Change", "Principal Adjustment"],
        "PAYMENT": ["Missed Payment", "Payment Plan", "Fee Waiver"],
        "DOCS": ["Statement Request", "Tax Documents", "Loan Agreement"],
        "INQUIRY": ["Status Update", "Eligibility", "Other"]
    }
    
    ROUTING_RULES = {
        "LOAN_MOD": {
            "Extension": "LoanServicingTeam",
            "Rate Change": "UnderwritingTeam",
            "Principal Adjustment": "UnderwritingTeam"
        },
        "PAYMENT": {
            "Missed Payment": "CollectionsTeam",
            "Payment Plan": "CollectionsTeam",
            "Fee Waiver": "CustomerServiceTeam"
        },
        "DOCS": {
            "*": "DocumentationTeam"
        },
        "INQUIRY": {
            "*": "CustomerServiceTeam"
        }
    }

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('email_processing.log'),
        logging.StreamHandler()
    ]
)

class EMLProcessor:
    """Processes EML files through the complete workflow"""
    
    def __init__(self):
        os.makedirs(Config.EML_DIRECTORY, exist_ok=True)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        openai.api_key = Config.OPENAI_API_KEY
    
    def run(self):
        """Main execution method"""
        logging.info("Starting EML processing pipeline")
        
        eml_files = [f for f in os.listdir(Config.EML_DIRECTORY) 
                    if f.endswith(".eml")]
        
        if not eml_files:
            logging.warning(f"No EML files found in {Config.EML_DIRECTORY}")
            return
        
        processed_count = 0
        for filename in eml_files:
            try:
                filepath = os.path.join(Config.EML_DIRECTORY, filename)
                result = self.process_single_file(filepath)
                if result:
                    processed_count += 1
            except Exception as e:
                logging.error(f"Failed to process {filename}: {str(e)}")
        
        logging.info(f"Processing complete. {processed_count}/{len(eml_files)} files processed successfully")
    
    def process_single_file(self, filepath):
        """Process a single EML file through all stages"""
        # Stage 1: Parse EML file
        email_data = self.parse_eml(filepath)
        if not email_data:
            return False
        
        # Stage 2: Classify email
        classification = self.classify_email(email_data)
        
        # Stage 3: Extract data
        extracted_data = self.extract_fields(email_data, classification["primary_type"])
        
        # Stage 4: Determine routing
        routing = self.determine_routing(classification, extracted_data)
        
        # Stage 5: Save results
        self.save_results(
            os.path.basename(filepath),
            email_data,
            classification,
            extracted_data,
            routing
        )
        
        return True
    
    def parse_eml(self, filepath):
        """Parse EML file and extract content"""
        try:
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
        except Exception as e:
            logging.error(f"Failed to parse {filepath}: {str(e)}")
            return None
    
    def _process_attachment(self, part):
        """Process email attachments"""
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
            logging.warning(f"Attachment processing failed: {str(e)}")
            return None
    
    def _clean_subject(self, subject):
        """Decode email subject header"""
        if subject is None:
            return ""
        decoded = decode_header(subject)
        return "".join(
            t[0].decode(t[1] if t[1] else "utf-8") if isinstance(t[0], bytes) else t[0]
            for t in decoded
        )
    
    def classify_email(self, email_data):
        """Classify email using LLM"""
        prompt = self._build_classification_prompt(email_data)
        
        try:
            response = openai.ChatCompletion.create(
                model=Config.CLASSIFICATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500
            )
            return self._parse_classification(response.choices[0].message.content)
        except Exception as e:
            logging.error(f"Classification failed: {str(e)}")
            return self._fallback_classification()
    
    def _build_classification_prompt(self, email_data):
        """Construct the LLM prompt for classification"""
        return f"""
        Analyze this commercial banking email and:
        1. Classify the primary request type from: {list(Config.REQUEST_TYPES.values())}
        2. Classify the sub-request type from the appropriate sub-categories
        3. Explain your reasoning in 1-2 sentences
        4. Identify any secondary requests (if present)
        
        Return your response in JSON format with these keys:
        - primary_type (code from: {list(Config.REQUEST_TYPES.keys())})
        - sub_type
        - reasoning
        - secondary_requests (list)
        
        Email Details:
        Subject: {email_data['subject']}
        From: {email_data['from']}
        Body: {email_data['body'][:2000]}
        Attachments: {len(email_data['attachments'])} found
        """
    
    def _parse_classification(self, response_text):
        """Parse the LLM classification response"""
        try:
            # Find JSON portion in response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            json_str = response_text[start:end]
            return json.loads(json_str)
        except Exception as e:
            logging.warning(f"Failed to parse classification: {str(e)}")
            return self._fallback_classification()
    
    def _fallback_classification(self):
        """Default classification when parsing fails"""
        return {
            "primary_type": "UNKNOWN",
            "sub_type": "UNKNOWN",
            "reasoning": "Classification failed",
            "secondary_requests": []
        }
    
    def extract_fields(self, email_data, request_type):
        """Extract key fields from email content"""
        patterns = {
            "loan_number": r"(?i)(loan|account)\s*(#|number|no\.?)\s*[:]?\s*(\w+)",
            "amount": r"\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?",
            "date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b"
        }
        
        fields = {
            "loan_number": self._extract_with_pattern(patterns["loan_number"], email_data),
            "amount": self._extract_amount(email_data, request_type, patterns["amount"]),
            "dates": self._extract_dates(email_data, patterns["date"]),
            "priority": self._determine_priority(email_data)
        }
        
        return fields
    
    def _extract_with_pattern(self, pattern, email_data):
        """Extract field using regex pattern"""
        # Check subject first
        match = re.search(pattern, email_data["subject"])
        if match:
            return match.group(3) if len(match.groups()) >= 3 else match.group(0)
        
        # Then check body
        match = re.search(pattern, email_data["body"])
        return match.group(3) if match and len(match.groups()) >= 3 else match.group(0) if match else None
    
    def _extract_amount(self, email_data, request_type, amount_pattern):
        """Extract monetary amounts with priority rules"""
        amounts = []
        
        # Priority to attachments for payment-related requests
        if request_type == "PAYMENT":
            for attachment in email_data["attachments"]:
                amounts.extend(re.findall(amount_pattern, attachment))
        
        # Fallback to email body
        if not amounts:
            amounts = re.findall(amount_pattern, email_data["body"])
        
        if not amounts:
            return None
        
        # Return highest amount found (often the most relevant)
        try:
            return max(amounts, key=lambda x: float(x.replace("$", "").replace(",", "")))
        except:
            return amounts[0]
    
    def _extract_dates(self, email_data, date_pattern):
        """Extract and parse dates from email content"""
        dates = []
        for text in [email_data["body"]] + email_data["attachments"]:
            dates.extend(re.findall(date_pattern, text))
        
        parsed_dates = []
        for date_str in dates:
            try:
                # Try MM/DD/YYYY format
                parsed_dates.append(datetime.strptime(date_str, "%m/%d/%Y"))
            except ValueError:
                try:
                    # Try Month Day, Year format
                    parsed_dates.append(datetime.strptime(date_str, "%b %d, %Y"))
                except:
                    continue
        
        return sorted(parsed_dates)
    
    def _determine_priority(self, email_data):
        """Determine email priority based on content"""
        urgent_keywords = ["urgent", "immediate", "asap", "time sensitive"]
        content = f"{email_data['subject']} {email_data['body']}".lower()
        return "HIGH" if any(kw in content for kw in urgent_keywords) else "NORMAL"
    
    def determine_routing(self, classification, extracted_data):
        """Determine where to route the request"""
        primary_type = classification["primary_type"]
        sub_type = classification["sub_type"]
        
        # Get routing rules for this request type
        type_rules = Config.ROUTING_RULES.get(primary_type, {})
        
        # Check for specific sub-type routing
        recipient = type_rules.get(sub_type)
        
        # Fallback to wildcard routing
        if not recipient and "*" in type_rules:
            recipient = type_rules["*"]
        
        # Apply priority routing
        if extracted_data.get("priority") == "HIGH":
            recipient = f"Urgent{recipient}"
        
        return {
            "assigned_to": recipient or "GeneralTeam",
            "priority": extracted_data.get("priority", "NORMAL")
        }
    
    def save_results(self, filename, email_data, classification, extracted_data, routing):
        """Save processing results to JSON file"""
        result = {
            "metadata": {
                "original_file": filename,
                "processed_at": datetime.now().isoformat(),
                "sender": email_data["from"],
                "subject": email_data["subject"],
                "date": email_data["date"]
            },
            "classification": classification,
            "extracted_data": extracted_data,
            "routing": routing
        }
        
        output_filename = f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = os.path.join(Config.OUTPUT_DIR, output_filename)
        
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        
        logging.info(f"Saved results for {filename} to {output_path}")

if __name__ == "__main__":
    processor = EMLProcessor()
    processor.run()