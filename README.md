# TEXT-SUMMARIZATION-TOOL

## Company: CODEC Technologies

## Name : Harsh Mehta

## Intern ID: CT06DH2139

## Domain: Artificial Intelligence 

## Duration   : 6 weeks

## Mentor: Neela Santhosh Kumar

---

## Summary


This project implements a comprehensive AI-powered text summarization system that transforms lengthy documents into concise, meaningful summaries using advanced machine learning techniques. The application combines multiple natural language processing algorithms with a modern web interface to deliver an intuitive and powerful text analysis tool.

## Core Technologies

### Backend Technologies
- **Python 3.x** - Primary programming language for ML implementation
- **Flask** - Lightweight web framework for RESTful API development
- **NLTK (Natural Language Toolkit)** - Advanced text processing and tokenization
- **scikit-learn** - Machine learning library for TF-IDF vectorization and cosine similarity
- **NumPy** - Numerical computing for matrix operations and statistical calculations
- **Flask-CORS** - Cross-origin resource sharing for frontend-backend communication

### Frontend Technologies
- **HTML5/CSS3** - Modern responsive web interface with glassmorphism design
- **Vanilla JavaScript** - Dynamic user interactions and API communication
- **CSS Grid/Flexbox** - Responsive layout system
- **Gradient Design System** - Contemporary visual aesthetics with backdrop filters

## Implementation Architecture

### Machine Learning Engine (`text_summarizer.py`)
The core ML component implements three distinct summarization algorithms:

**1. Frequency-Based Summarization**
- Analyzes word frequency patterns using Porter stemming
- Implements intelligent stop word filtering
- Applies positional weighting (first/last sentence bonuses)
- Provides content-aware scoring (numerical data recognition)

**2. TF-IDF Based Summarization**
- Utilizes Term Frequency-Inverse Document Frequency vectorization
- Performs cosine similarity calculations between sentences
- Implements advanced feature extraction with up to 100 key features
- Provides sophisticated term weighting for context understanding

**3. Hybrid Approach (Recommended)**
- Combines frequency-based (60%) and TF-IDF (40%) scoring methods
- Normalizes scores across different algorithms for optimal results
- Balances computational efficiency with summarization quality

### API Layer (`app.py`)
The Flask-based REST API provides:
- **POST /summarize** - Main summarization endpoint with comprehensive validation
- **GET /health** - System health monitoring
- **GET /methods** - Available algorithm information
- Production-ready error handling with structured JSON responses
- Request validation and sanitization
- Detailed logging for monitoring and debugging

### Frontend Interface (`index.html`)
Modern web application featuring:
- Responsive design optimized for desktop and mobile devices
- Real-time parameter adjustment (summary length, algorithm selection)
- Interactive statistics dashboard showing compression metrics
- Syntax highlighting for generated summaries
- Progressive loading indicators and error feedback
- Keyboard shortcuts for enhanced user experience

## Key Features

### Advanced Text Processing
- **Intelligent Preprocessing**: Removes noise while preserving semantic meaning
- **Sentence Segmentation**: Uses NLTK's punkt tokenizer for accurate sentence boundary detection
- **Stemming & Normalization**: Porter stemming algorithm for word root extraction
- **Multi-Algorithm Support**: Users can choose optimal summarization method for their content

### Performance Optimization
- **Scalable Architecture**: Modular design supporting easy algorithm additions
- **Efficient Processing**: Optimized matrix operations using NumPy
- **Memory Management**: Proper resource handling for large document processing
- **Caching Ready**: Architecture supports future implementation of result caching

### User Experience
- **Real-Time Feedback**: Instant compression ratio and word count statistics
- **Visual Enhancement**: Color-coded sentence highlighting in generated summaries
- **Accessibility**: WCAG-compliant design with proper contrast ratios
- **Cross-Platform**: Works seamlessly across different browsers and devices

## Technical Implementation Highlights

The system employs sophisticated NLP techniques including cosine similarity matrices for sentence relationship analysis, normalized TF-IDF scoring for semantic understanding, and heapq-based sentence ranking for efficient top-k selection. The hybrid algorithm intelligently weights different scoring methods to achieve optimal summarization quality across various text types.

The frontend implements modern web standards with CSS custom properties, smooth animations, and responsive design patterns. The API layer follows RESTful principles with proper HTTP status codes, structured error responses, and comprehensive input validation.

## Deployment & Scalability

The application is designed for easy deployment with Docker containerization support and can be scaled horizontally using load balancers. The modular architecture allows for future enhancements including multi-language support, advanced ML models integration, and enterprise-level authentication systems.

---

## Output

<img width="1919" height="966" alt="Image" src="https://github.com/user-attachments/assets/dc53e186-0311-4c22-9f49-84ed3e226858" />
<img width="1919" height="975" alt="Image" src="https://github.com/user-attachments/assets/16e3ef57-2f3b-44bd-9d5d-30bb1fa882d2" />
