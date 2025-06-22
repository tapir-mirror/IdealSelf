from flask import Flask, request, render_template, session, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
import subprocess, json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from torch.onnx.symbolic_opset11 import unsqueeze
from youtube_transcript_api import YouTubeTranscriptApi
import numpy as np
from yt_dlp import YoutubeDL
import requests
from bs4 import BeautifulSoup
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import spacy
import ssl
from huggingface_hub import login

import os
from huggingface_hub import snapshot_download

os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"


snapshot_download(
    repo_id="meta-llama/Meta-Llama-3-8B",
    local_dir="llama3_model",
    local_dir_use_symlinks=False
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tapir-mirrordb.db'
db = SQLAlchemy(app)
app.secret_key = "zacuscacuvinetesiciuperciafumate"

users_references = db.Table('users_references',
    db.Column('user_id', db.Integer, db.ForeignKey('users.id'), primary_key=True),
    db.Column('reference_id', db.Integer, db.ForeignKey('references.id'), primary_key=True)
)

users_strongpoints = db.Table('users_strongpoints',
    db.Column('user_id', db.Integer, db.ForeignKey('users.id'), primary_key=True),
    db.Column('strongpoint_id', db.Integer, db.ForeignKey('strongpoints.id'), primary_key=True)
)

users_improvables = db.Table('users_improvables',
    db.Column('user_id', db.Integer, db.ForeignKey('users.id'), primary_key=True),
    db.Column('improvable_id', db.Integer, db.ForeignKey('improvables.id'), primary_key=True)
)

users_actions = db.Table('users_actions',
    db.Column('user_id', db.Integer, db.ForeignKey('users.id'), primary_key=True),
    db.Column('action_id', db.Integer, db.ForeignKey('actions.id'), primary_key=True)
)

users_resources = db.Table('users_resources',
    db.Column('user_id', db.Integer, db.ForeignKey('users.id'), primary_key=True),
    db.Column('resource_id', db.Integer, db.ForeignKey('resources.id'), primary_key=True)
)

strongpoints_resources = db.Table('strongpoints_resources',
    db.Column('strongpoint_id', db.Integer, db.ForeignKey('strongpoints.id'), primary_key=True),
    db.Column('resource_id', db.Integer, db.ForeignKey('resources.id'), primary_key=True)
)

improvables_resources = db.Table('improvables_resources',
    db.Column('improvable_id', db.Integer, db.ForeignKey('improvables.id'), primary_key=True),
    db.Column('resource_id', db.Integer, db.ForeignKey('resources.id'), primary_key=True)
)

actions_resources = db.Table('actions_resources',
    db.Column('action_id', db.Integer, db.ForeignKey('actions.id'), primary_key=True),
    db.Column('resource_id', db.Integer, db.ForeignKey('resources.id'), primary_key=True)
)


# Models
class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False)
    password = db.Column(db.String(35))
    email = db.Column(db.String(70), unique=True)

    references = db.relationship('Reference', secondary=users_references, back_populates='users')
    strongpoints = db.relationship('Strongpoint', secondary=users_strongpoints, back_populates='users')
    improvables = db.relationship('Improvable', secondary=users_improvables, back_populates='users')
    actions = db.relationship('Action', secondary=users_actions, back_populates='users')
    resources = db.relationship('Resource', secondary=users_resources, back_populates='users')

    def __repr__(self):
        return f"<User {self.id}, {self.username}>"


class Reference(db.Model):
    __tablename__ = "references"
    id = db.Column(db.Integer, primary_key=True)
    reference = db.Column(db.String(20), nullable=False)
    trait = db.Column(db.String(35))
    category = db.Column(db.String(50))

    users = db.relationship('User', secondary=users_references, back_populates='references')

    def __repr__(self):
        return f"<Reference {self.id}, {self.reference}, {self.trait}>"


class Strongpoint(db.Model):
    __tablename__ = "strongpoints"
    id = db.Column(db.Integer, primary_key=True)
    strongpoint = db.Column(db.String(20), nullable=False)

    users = db.relationship('User', secondary=users_strongpoints, back_populates='strongpoints')
    resources = db.relationship('Resource', secondary=strongpoints_resources, back_populates='strongpoints')

    def __repr__(self):
        return f"<Strongpoint {self.id}, {self.strongpoint}>"


class Improvable(db.Model):
    __tablename__ = "improvables"
    id = db.Column(db.Integer, primary_key=True)
    improvable = db.Column(db.String(20), nullable=False)

    users = db.relationship('User', secondary=users_improvables, back_populates='improvables')
    resources = db.relationship('Resource', secondary=improvables_resources, back_populates='improvables')

    def __repr__(self):
        return f"<Improvable {self.id}, {self.improvable}>"


class Action(db.Model):
    __tablename__ = "actions"
    id = db.Column(db.Integer, primary_key=True)
    action = db.Column(db.String(200), nullable=False)

    users = db.relationship('User', secondary=users_actions, back_populates='actions')
    resources = db.relationship('Resource', secondary=actions_resources, back_populates='actions')

    def __repr__(self):
        return f"<Action {self.id}, {self.action}>"


class Resource(db.Model):
    __tablename__ = "resources"
    id = db.Column(db.Integer, primary_key=True)
    resource = db.Column(db.String(50), nullable=False)

    users = db.relationship('User', secondary=users_resources, back_populates='resources')
    strongpoints = db.relationship('Strongpoint', secondary=strongpoints_resources, back_populates='resources')
    improvables = db.relationship('Improvable', secondary=improvables_resources, back_populates='resources')
    actions = db.relationship('Action', secondary=actions_resources, back_populates='resources')

    def __repr__(self):
        return f"<Resource {self.id}, {self.resource}>"


with app.app_context():
    db.create_all()

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        print(user)
        if not user:
            return render_template("inspiration.html")
        else:
            if user.password == password:
                session["user_id"] = user.id
                return redirect(url_for("newsfeed"))
            else:
                flash("Incorrect password.")
                return render_template("signin.html")
    return render_template("signin.html")

@app.route('/user')
def logged_in_user():
    user_id = None
    if "user_id" in session:
        user_id = session["user_id"]
        return f"<h1> Hi User ID: {user_id}</h1>"
    else:
        flash("Please log in before using the app.")
        return redirect(url_for("login"))


def parse_references(raw_input):

    reference_map = {}
    if not raw_input:
        return reference_map

    entries = raw_input.split(',')
    buffer = []
    for entry in entries:
        part = entry.strip()
        if ':' in part:
            names_part, trait = part.split(':', 1)
            all_names = buffer + [name.strip() for name in names_part.split(',') if name.strip()]
            for name in all_names:
                reference_map[name] = trait.strip()
            buffer = []
        else:
            buffer.append(part)

    return reference_map

@app.route("/inspiration", methods=["GET", "POST"])
def get_references():
    if request.method == "POST":
        username=request.form.get('username')
        email=request.form.get('email')
        password=request.form.get('password')
        hist_map = parse_references(request.form.get('historical_references'))
        fict_map = parse_references(request.form.get('fictional_references'))
        cont_map = parse_references(request.form.get('contemporary_references'))

        all_reference_maps = {
            "historical": hist_map,
            "fictional": fict_map,
            "contemporary": cont_map,
        }

        # Create user
        new_user = User(
            username=username,
            email=email,
            password=password,
        )

        for category, ref_map in all_reference_maps.items():
            for name, trait in ref_map.items():
                existing_ref = Reference.query.filter_by(reference=name, trait=trait, category=category).first()
                if not existing_ref:
                    existing_ref = Reference(reference=name, trait=trait, category=category)
                    db.session.add(existing_ref)
                new_user.references.append(existing_ref)

        db.session.add(new_user)
        db.session.commit()

        session["user_id"] = new_user.id

        return render_template("newsfeed.html", references=all_reference_maps)

    return render_template("newsfeed.html", references=None)

def search_youtube_videos(query, max_results=5):
    result = subprocess.run(
        ["yt-dlp", f"ytsearch{max_results}:{query}", "--flat-playlist", "-J"],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print("yt-dlp failed:", result.stderr)
        return []

    try:
        data = json.loads(result.stdout)
        if "entries" in data and data["entries"]:
            return [entry["id"] for entry in data["entries"] if "id" in entry]
        else:
            print(f"No entries found for query: {query}")
            return []
    except json.JSONDecodeError as e:
        print("JSON decode error:", e)
        print("Output was:", result.stdout)
        return []

def fetch_transcripts(video_ids):
    transcripts = {}
    for vid in video_ids:
        try:
            transcripts[vid] = YouTubeTranscriptApi.get_transcript(vid)
        except:
            continue
    return transcripts

def get_wikiquote_quotes(query, max_quotes=10):
    search_url = f"https://en.wikiquote.org/w/index.php?search={query.replace(' ', '+')}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Try to find a direct match
    match = soup.find("ul", class_="mw-search-results")
    if match:
        link = match.find("a")["href"]
        page_url = f"https://en.wikiquote.org{link}"
    else:
        # Try first direct suggestion
        suggestion = soup.find("div", class_="searchdidyoumean")
        if suggestion and suggestion.find("a"):
            suggestion_link = suggestion.find("a")["href"]
            page_url = f"https://en.wikiquote.org{suggestion_link}"
        else:
            return [f"No quotes found for '{query}'"]

    page = requests.get(page_url)
    soup = BeautifulSoup(page.text, 'html.parser')
    quotes = soup.select("ul > li")

    extracted = []
    for li in quotes:
        text = li.get_text(strip=True)
        if len(text) > 40:  # Filter short or empty entries
            extracted.append(text)
        if len(extracted) >= max_quotes:
            break

    return extracted or [f"No quotes found on the page for '{query}'"]

@app.route('/newsfeed', methods=['GET', 'POST'])
def newsfeed():
    user_id = session.get("user_id")
    print(f"{user_id} is here")

    user = User.query.get(user_id)
    if not user:
        flash("User not found. Please log in again.")
        return redirect(url_for("login"))

    reference_quotes = []
    reference_transcripts = {}

    for reference in user.references:
        quote_list = get_wikiquote_quotes(reference.reference)
        reference_quotes.extend(quote_list)

        video_ids = search_youtube_videos(reference.reference)
        reference_transcripts.update(fetch_transcripts(video_ids))

    return render_template("newsfeed.html", user=user, reference_quotes=reference_quotes, reference_transcripts=reference_transcripts)


# Initialize spacy and sentencizer once
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

# Initialize SentenceTransformer once at module level
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define your CNN model
class MultiHeadCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters=100, filter_sizes=[3,4,5], dropout=0.5):
        super(MultiHeadCNN, self).__init__()

        def conv_layers():
            return nn.ModuleList([
                nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
            ])

        self.personal_convs = conv_layers()
        self.historical_convs = conv_layers()
        self.contemporary_convs = conv_layers()
        self.fictional_convs = conv_layers()

        self.dropout = nn.Dropout(dropout)

        total_filters = num_filters * len(filter_sizes) * 4  # 4 heads
        self.fc = nn.Linear(total_filters, 128)

    def forward(self, personal_x, historical_x, contemporary_x, fictional_x):
        # Add channel dim for conv2d: (batch, 1, seq_len, embed_dim)
        personal_x = personal_x.unsqueeze(1)
        historical_x = historical_x.unsqueeze(1)
        contemporary_x = contemporary_x.unsqueeze(1)
        fictional_x = fictional_x.unsqueeze(1)

        def apply_convs(convs, x):
            return [F.relu(conv(x)).squeeze(3) for conv in convs]

        def pool_outputs(conv_outs):
            return [F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outs]

        personal = [0.60 * f for f in pool_outputs(apply_convs(self.personal_convs, personal_x))]
        historical = [0.15 * f for f in pool_outputs(apply_convs(self.historical_convs, historical_x))]
        contemporary = [0.15 * f for f in pool_outputs(apply_convs(self.contemporary_convs, contemporary_x))]
        fictional = [0.10 * f for f in pool_outputs(apply_convs(self.fictional_convs, fictional_x))]

        concat = torch.cat(personal + historical + contemporary + fictional, dim=1)
        dropped = self.dropout(concat)
        return self.fc(dropped)

# Instantiate your CNN model once
embedding_dim = 384
model = MultiHeadCNN(embedding_dim)


# Load model once globally
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id, token="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")
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", token="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")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Prompt template
def llm_stylized_prompt(name, trait, tone):
    return f"""You are a stylized writer.

Write a motivational paragraph in the tone: **{tone}**.

Inspire the user by channeling the essence of **{name}**, known for **{trait}**.

Make the paragraph evocative, memorable, and no longer than 3 sentences."""

def generate_response_with_llm(style_vector, references, soul_weights=None):
    if soul_weights is None:
        soul_weights = {
            'personal': 0.60,
            'historical': 0.15,
            'contemporary': 0.15,
            'fictional': 0.10
        }

    vector_length = style_vector.shape[0]
    segment_size = vector_length // 4

    soul_values = {
        'personal': style_vector[:segment_size].mean().item(),
        'historical': style_vector[segment_size:2*segment_size].mean().item(),
        'contemporary': style_vector[2*segment_size:3*segment_size].mean().item(),
        'fictional': style_vector[3*segment_size:].mean().item()
    }

    composite_score = {
        soul: soul_values[soul] * soul_weights[soul] for soul in soul_values
    }

    dominant_soul = max(composite_score, key=composite_score.get)

    # Grouped reference logic
    grouped_refs = {
        "historical": [r for r in references if "Strategy" in r or "Empire" in r],
        "contemporary": [r for r in references if "Innovation" in r or "Tech" in r],
        "fictional": [r for r in references if "Magic" in r or "Imagination" in r],
        "personal": [r for r in references if "Resilience" in r or "Grace" in r]
    }

    ref_pool = grouped_refs.get(dominant_soul, references)
    reference = np.random.choice(ref_pool) if ref_pool else np.random.choice(references)

    if ":" in reference:
        name, trait = map(str.strip, reference.split(":"))
    else:
        name, trait = reference, "default"

    tone_map = {
        ("personal", "Resilience"): "gentle and grounded",
        ("historical", "Strategy"): "commanding and visionary",
        ("contemporary", "Innovation"): "bold and forward-thinking",
        ("fictional", "Imagination"): "poetic and whimsical",
        ("personal", "default"): "warm and encouraging",
        ("historical", "default"): "structured and wise",
        ("contemporary", "default"): "analytical and modern",
        ("fictional", "default"): "dreamy and symbolic"
    }

    tone = tone_map.get((dominant_soul, trait), tone_map.get((dominant_soul, "default")))

    # Prepare and run LLM prompt
    prompt = llm_stylized_prompt(name, trait, tone)
    output = generator(prompt, max_new_tokens=150, temperature=0.8, do_sample=True)

    return output[0]['generated_text']

@app.route('/stylized_response', methods=['GET'])
def stylized_response():
    user_id = session.get("user_id")
    user = User.query.get(user_id)

    if not user:
        return jsonify({"error": "User not found"}), 401

    # Load and embed personal corpus
    with open("personal-corpus.txt", "r") as f:
        personal_data = f.read()

    doc = nlp(personal_data)
    sentences = [sent.text for sent in doc.sents]

    # Use embed_model to get embeddings — NOT 'model'
    personal_embeddings = embed_model.encode(sentences[:50])  # limit to 50 sentences
    personal_input = torch.tensor(personal_embeddings).unsqueeze(0)  # (1, seq_len, embedding_dim)

    # Simulated inputs for other styles (replace with real data if available)
    historical_input = torch.randn(1, 50, embedding_dim)
    contemporary_input = torch.randn(1, 50, embedding_dim)
    fictional_input = torch.randn(1, 50, embedding_dim)

    # Forward pass through your CNN model
    style_vector = model(personal_input, historical_input, contemporary_input, fictional_input)

    # Assume user.references is a list of objects with a 'reference' string attribute
    references = [r.reference for r in user.references]

    response_text = generate_response_with_llm(style_vector[0], references)

    return jsonify({"response": response_text})


if __name__ == "__main__":
    app.run(debug=True)


#Strongpoints updated by progress




if __name__ == '__main__':
    app.run(debug=True)