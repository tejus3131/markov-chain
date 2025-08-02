"""
Web-based GUI for Markov Chain Text Predictor
Provides a real-time interface accessible through a web browser
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
from pathlib import Path
import webbrowser

from chain.chain import MarkovChainTextPredictor


class MarkovChainWebHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Markov Chain web interface."""
    
    def __init__(self, predictor, *args, **kwargs):
        self.predictor = predictor
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
        predictor = self.predictor
        
        class Handler(MarkovChainWebHandler):
            def __init__(self, *args, **kwargs):
                self.predictor = predictor
                super(BaseHTTPRequestHandler, self).__init__(*args, **kwargs)
        
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