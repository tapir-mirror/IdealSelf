Ideal Self Life Coach — LLM-powered Personal Mentor
===================================================

This Flask application hosts an AI-powered life coach modeled as your Ideal Self — a future or parallel dimension version of you designed to empower, guide, and support your growth journey.

Concept
-------
The Ideal Self AI blends multiple influences to create a unique coaching voice:
- 60% Your Own Voice — Your personal style and tone.
- 15% Historical Personal Hero — A figure from history who inspires you.
- 15% Contemporary Figure You Admire — A present-day role model.
- 10% Fictional Character — Someone whose traits you resonate with.

This composite mentor acts as a compassionate, insightful life coach who understands your aspirations and supports you with tailored advice, encouragement, and reflection.

Features
--------
- Interactive conversational interface powered by large language models (LLMs).
- Fetch and incorporate real-time YouTube transcripts to enrich context.
- Semantic understanding with Sentence Transformers for personalized responses.
- Persistent sessions with Flask and SQLAlchemy.
- Easy integration with Hugging Face transformers and model hub.

Getting Started
---------------
Prerequisites:
- Python 3.9+
- Virtualenv recommended
- Access to Hugging Face account/token for model downloads

Installation:
1. Clone the repo
   git clone https://github.com/yourusername/ideal-self-life-coach.git
   cd ideal-self-life-coach

2. Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate  # Linux / macOS
   .venv\Scripts\activate   # Windows

3. Install dependencies
   pip install -r requirements.txt

Configuration:
- Set your Hugging Face token:
  export HUGGINGFACE_TOKEN="your_token_here"

Running the app:
- flask run
- Visit http://localhost:5000

Usage
-----
- Input your thoughts, questions, or challenges.
- The AI responds in your ideal self’s voice.
- Save and revisit sessions for ongoing growth.

Dependencies
------------
- Flask & Flask_SQLAlchemy
- Transformers & Hugging Face Hub
- Sentence Transformers
- YouTube Transcript API
- yt-dlp
- BeautifulSoup4
- Torch
- Spacy

Future Work
-----------
- Add multi-modal inputs (voice, video).
- Deeper customization of persona blends.
- Integration with calendar, reminders, and goal trackers.
