"""
Web-based GUI for Markov Chain Text Predictor
Provides a real-time interface accessible through a web browser
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
from pathlib import Path
import webbrowser
import re
from datetime import datetime
from typing import List, Optional
import io

from chain.chain import MarkovChainTextPredictor


class WhatsAppProcessor:
    """Process WhatsApp chat exports to extract meaningful sentences."""
    
    def __init__(self):
        # Pattern to match WhatsApp export format: DD/MM/YY, H:MM pm/am - Sender: Message
        # Note: WhatsApp uses \u202f (narrow no-break space) before am/pm
        self.pattern = r'(\d{2}/\d{2}/\d{2}), (\d{1,2}:\d{2}\u202f[ap]m) - (.*?)(?=\n\d{2}/\d{2}/\d{2}, \d{1,2}:\d{2}\u202f[ap]m - |\Z)'
    
    def parse_datetime(self, date_str: str, time_str: str) -> datetime:
        """Convert WhatsApp date and time strings to datetime object"""
        # Replace the Unicode narrow no-break space with regular space
        time_str = time_str.replace('\u202f', ' ')
        datetime_str = f"{date_str} {time_str}"
        return datetime.strptime(datetime_str, "%d/%m/%y %I:%M %p")
    
    def process_chat(self, chat_data: str) -> List[dict]:
        """Process WhatsApp chat export and return list of message dictionaries"""
        messages = []
        
        # Find all matches in the chat data
        matches = re.findall(self.pattern, chat_data, re.DOTALL | re.MULTILINE)
        
        for match in matches:
            date_str, time_str, content = match
            
            # Split sender and message
            if ': ' in content:
                # Has a colon, so it's a user message
                sender, message = content.split(': ', 1)
            else:
                # No colon, it's a system message - treat as from "System"
                sender = "System"
                message = content
            
            # Clean up the message (remove extra whitespace)
            message = message.strip().replace('\n', ' ')
            
            # Parse datetime
            try:
                dt = self.parse_datetime(date_str, time_str)
                messages.append({
                    'date': dt,
                    'sender': sender.strip(),
                    'message': message
                })
            except ValueError:
                # Skip messages with invalid datetime
                continue
        
        return messages
    
    def get_sentences(self, chat_data: str) -> List[str]:
        """Extract all meaningful sentences from the chat data"""
        messages = self.process_chat(chat_data)
        sentences = []
        
        for msg in messages:
            # Skip null messages, media omitted, and system messages
            if msg['message'].lower() in ['null', '<media omitted>']:
                continue
            
            # Skip system messages
            if any(phrase in msg['message'].lower() for phrase in [
                'turned off disappearing messages',
                'messages and calls are end-to-end encrypted',
                'uses a default timer for disappearing messages',
                'new messages will disappear',
                'media omitted'
            ]):
                continue
            
            # Skip URLs
            if 'https://meet.google.com' in msg['message'] or msg['message'].startswith('http'):
                continue
            
            # Skip very short messages
            if len(msg['message'].strip()) < 5:
                continue
            
            # Split message into sentences
            text = msg['message']
            sentence_endings = re.split(r'[.!?]+\s+', text)
            
            for sentence in sentence_endings:
                sentence = sentence.strip()
                # Filter out very short fragments, URLs, and non-meaningful content
                if (sentence and 
                    len(sentence) > 10 and 
                    not sentence.startswith('http') and
                    not sentence.lower() in ['gn', 'sure', 'okay', 'ok', 'thanks', 'np'] and
                    not re.match(r'^[^\w]*$', sentence)):  # Skip non-word content
                    sentences.append(sentence)
        
        return sentences


from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from chain.chain import MarkovChainTextPredictor


class MarkovChainWebHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Markov Chain web interface."""
    
    predictor: 'MarkovChainTextPredictor'
    whatsapp_processor: 'WhatsAppProcessor'
    
    def __init__(self, *args, **kwargs):
        # These attributes will be set by the parent class
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/':
            self.serve_html()
        elif self.path == '/api/stats':
            self.serve_stats()
        elif self.path.startswith('/api/process'):
            self.serve_process_text()
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests."""
        if self.path == '/api/add_text':
            self.handle_add_text()
        elif self.path == '/api/import_whatsapp':
            self.handle_import_whatsapp()
        else:
            self.send_error(404)
    
    def serve_html(self):
        """Serve the main HTML interface."""
        html_content = self.get_html_content()
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def serve_stats(self):
        """Serve model statistics as JSON."""
        try:
            stats = self.predictor.get_stats()
            self.send_json_response(stats)
        except Exception as e:
            self.send_json_response({"error": str(e)}, 500)
    
    def serve_process_text(self):
        """Process text and return predictions."""
        try:
            # Parse query parameters
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            text = params.get('text', [''])[0]
            
            if text:
                result = self.predictor.process_text(text)
                self.send_json_response(result)
            else:
                self.send_json_response({"error": "No text provided"}, 400)
        except Exception as e:
            self.send_json_response({"error": str(e)}, 500)
    
    def handle_add_text(self):
        """Add text to the knowledge base."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            text = data.get('text', '')
            
            if text:
                # Split text into lines and add each line separately
                lines = text.strip().split('\n')
                success_count = 0
                
                for line in lines:
                    line = line.strip()
                    if line:  # Only add non-empty lines
                        success = self.predictor.add_text(line)
                        if success:
                            success_count += 1
                
                self.send_json_response({
                    "success": True,
                    "lines_processed": len(lines),
                    "lines_added": success_count
                })
            else:
                self.send_json_response({"error": "No text provided"}, 400)
        except Exception as e:
            self.send_json_response({"error": str(e)}, 500)
    
    def handle_import_whatsapp(self):
        """Import WhatsApp chat data and extract sentences for training."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            chat_data = data.get('chat_data', '')
            
            if not chat_data:
                self.send_json_response({"error": "No chat data provided"}, 400)
                return
            
            # Process WhatsApp chat data
            try:
                sentences = self.whatsapp_processor.get_sentences(chat_data)
                
                if not sentences:
                    self.send_json_response({
                        "error": "No meaningful sentences found in the chat data"
                    }, 400)
                    return
                
                # Add sentences to the knowledge base
                success_count = 0
                for sentence in sentences:
                    success = self.predictor.add_text(sentence)
                    if success:
                        success_count += 1
                
                # Get some statistics about the processed chat
                messages = self.whatsapp_processor.process_chat(chat_data)
                senders = set(msg['sender'] for msg in messages if msg['sender'] != 'System')
                
                self.send_json_response({
                    "success": True,
                    "total_messages": len(messages),
                    "sentences_extracted": len(sentences),
                    "sentences_added": success_count,
                    "unique_senders": list(senders),
                    "date_range": {
                        "start": min(msg['date'] for msg in messages).strftime("%Y-%m-%d") if messages else None,
                        "end": max(msg['date'] for msg in messages).strftime("%Y-%m-%d") if messages else None
                    }
                })
                
            except Exception as parse_error:
                self.send_json_response({
                    "error": f"Failed to parse WhatsApp chat: {str(parse_error)}"
                }, 400)
                
        except Exception as e:
            self.send_json_response({"error": str(e)}, 500)
    
    def send_json_response(self, data, status_code=200):
        """Send a JSON response."""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def get_html_content(self):
        """Generate the HTML content for the interface."""
        html_path = Path(__file__).parent / 'gui.html'
        if html_path.exists():
            with open(html_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            self.send_error(404, "HTML file not found")
            return ""
        return 
    
    def log_message(self, format, *args):
        """Override to reduce logging noise."""
        pass


class MarkovChainWebGUI:
    """Web-based GUI for the Markov Chain Text Predictor."""
    
    def __init__(self, model_path: str = "markov_model.pkl", port: int = 8080):
        self.model_path = model_path
        self.port = port
        self.predictor = None
        self.server = None
        
        # Initialize the predictor
        self.init_predictor()
        
        # Load existing model if available
        self.load_model()
    
    def init_predictor(self):
        """Initialize the Markov Chain predictor."""
        try:
            print("Initializing Markov Chain Text Predictor...")
            self.predictor = MarkovChainTextPredictor(
                ngram_sizes=[2, 3, 4, 5],
                smoothing_method='laplace',
                smoothing_alpha=0.1
            )
            print("✓ Predictor initialized successfully!")
        except Exception as e:
            print(f"❌ Failed to initialize predictor: {str(e)}")
            raise
    
    def load_model(self):
        """Load the model from file if it exists."""
        if self.predictor and Path(self.model_path).exists():
            try:
                success = self.predictor.load_model(self.model_path)
                if success:
                    print(f"✓ Model loaded from {self.model_path}")
                else:
                    print(f"❌ Failed to load model from {self.model_path}")
            except Exception as e:
                print(f"❌ Error loading model: {str(e)}")
        else:
            print("ℹ️  No existing model found, starting fresh")
    
    def save_model(self):
        """Save the current model to file."""
        if self.predictor:
            try:
                success = self.predictor.save_model(self.model_path)
                if success:
                    print(f"✓ Model saved to {self.model_path}")
                    return True
                else:
                    print(f"❌ Failed to save model to {self.model_path}")
                    return False
            except Exception as e:
                print(f"❌ Error saving model: {str(e)}")
                return False
        return False
    
    def create_handler(self):
        """Create a request handler with the predictor instance."""
        if self.predictor is None:
            raise RuntimeError("Predictor not initialized")
        
        predictor = self.predictor
        whatsapp_processor = WhatsAppProcessor()
        
        class Handler(MarkovChainWebHandler):
            def __init__(self, *args, **kwargs):
                self.predictor = predictor
                self.whatsapp_processor = whatsapp_processor
                super().__init__(*args, **kwargs)
        
        return Handler
    
    def run(self):
        """Start the web server."""
        try:
            handler_class = self.create_handler()
            self.server = HTTPServer(('localhost', self.port), handler_class)
            
            print(f"🚀 Markov Chain Web GUI started!")
            print(f"📱 Open your web browser and go to: http://localhost:{self.port}")
            print(f"💡 Ctrl+C to stop the server")
            print("=" * 50)
            
            # Try to open the browser automatically
            try:
                webbrowser.open(f"http://localhost:{self.port}")
            except:
                pass  # If browser opening fails, that's okay
            
            self.server.serve_forever()
            
        except KeyboardInterrupt:
            print("\\n🛑 Shutting down...")
            self.shutdown()
        except Exception as e:
            print(f"❌ Server error: {str(e)}")
            self.shutdown()
    
    def shutdown(self):
        """Shutdown the server and save the model."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        
        print("💾 Saving model...")
        if self.save_model():
            print("✓ Model saved successfully")
        else:
            print("❌ Failed to save model")
        
        print("👋 Goodbye!")


def main():
    """Main entry point for the web GUI application."""
    try:
        app = MarkovChainWebGUI(
            model_path=".data/model.pkl",
            port=8000
        )
        app.run()
    except Exception as e:
        print(f"❌ Failed to start web GUI: {str(e)}")