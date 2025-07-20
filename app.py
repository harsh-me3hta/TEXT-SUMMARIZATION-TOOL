from flask import Flask, request, jsonify
from flask_cors import CORS
from text_summarizer import TextSummarizer
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the summarizer
summarizer = TextSummarizer()

@app.route('/')
def home():
    """
    Home route - API information
    """
    return jsonify({
        "message": "Text Summarization API",
        "version": "1.0",
        "endpoints": {
            "/summarize": "POST - Generate text summary",
            "/health": "GET - Check API health"
        }
    })

@app.route('/health')
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "healthy",
        "message": "Text Summarization API is running"
    })

@app.route('/summarize', methods=['POST'])
def summarize_text():
    """
    Main summarization endpoint
    
    Expected JSON payload:
    {
        "text": "Your text to summarize...",
        "num_sentences": 3,  // optional, default: 3
        "method": "hybrid"   // optional, default: "hybrid" (options: frequency, tfidf, hybrid)
    }
    
    Returns:
    {
        "success": true,
        "summary": "Generated summary...",
        "metadata": {
            "method": "hybrid",
            "original_sentences": 10,
            "summary_sentences": 3,
            "compression_ratio": 70.0,
            "original_words": 150,
            "summary_words": 45
        }
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        # Extract parameters
        text = data.get('text', '').strip()
        num_sentences = data.get('num_sentences', 3)
        method = data.get('method', 'hybrid')
        
        # Validate inputs
        if not text:
            return jsonify({
                "success": False,
                "error": "Text parameter is required and cannot be empty"
            }), 400
        
        if len(text) < 50:
            return jsonify({
                "success": False,
                "error": "Text is too short to summarize. Please provide at least 50 characters."
            }), 400
        
        if not isinstance(num_sentences, int) or num_sentences < 1 or num_sentences > 10:
            return jsonify({
                "success": False,
                "error": "num_sentences must be an integer between 1 and 10"
            }), 400
        
        if method not in ['frequency', 'tfidf', 'hybrid']:
            return jsonify({
                "success": False,
                "error": "method must be one of: frequency, tfidf, hybrid"
            }), 400
        
        logger.info(f"Processing summarization request: method={method}, sentences={num_sentences}, text_length={len(text)}")
        
        # Generate summary
        result = summarizer.summarize(text, num_sentences=num_sentences, method=method)
        
        # Calculate word counts
        original_words = len(text.split())
        summary_words = len(result['summary'].split())
        
        # Prepare response
        response = {
            "success": True,
            "summary": result['summary'],
            "metadata": {
                "method": result['method'],
                "original_sentences": result['original_sentences'],
                "summary_sentences": result['summary_sentences'],
                "compression_ratio": result['compression_ratio'],
                "original_words": original_words,
                "summary_words": summary_words
            }
        }
        
        logger.info(f"Summary generated successfully: {result['compression_ratio']}% compression")
        return jsonify(response)
        
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        return jsonify({
            "success": False,
            "error": str(ve)
        }), 400
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "An unexpected error occurred while processing your request"
        }), 500

@app.route('/methods')
def get_methods():
    """
    Get available summarization methods
    """
    return jsonify({
        "methods": {
            "frequency": {
                "name": "Frequency-based",
                "description": "Uses word frequency analysis with stop word filtering"
            },
            "tfidf": {
                "name": "TF-IDF based", 
                "description": "Uses TF-IDF vectorization for sophisticated term weighting"
            },
            "hybrid": {
                "name": "Hybrid (Recommended)",
                "description": "Combines frequency and TF-IDF approaches for best results"
            }
        }
    })

@app.errorhandler(404)
def not_found(error):
    """
    Handle 404 errors
    """
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """
    Handle 405 errors
    """
    return jsonify({
        "success": False,
        "error": "Method not allowed"
    }), 405

if __name__ == '__main__':
    print("Starting Text Summarization API...")
    print("API will be available at: http://localhost:5000")
    print("Endpoints:")
    print("  GET  /            - API information")
    print("  GET  /health      - Health check") 
    print("  POST /summarize   - Generate summary")
    print("  GET  /methods     - Available methods")
    print("\nPress Ctrl+C to stop the server")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
