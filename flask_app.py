from flask import Flask, render_template, request
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from rouge_score import rouge_scorer
from pydub import AudioSegment
import tempfile
import os
import requests

app = Flask(__name__)

# Load the saved model
model_path = 'bart_samsum_model'  # Use the correct path for your model
loaded_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Speech-to-text API configuration
API_URL = "https://api-inference.huggingface.co/models/facebook/s2t-large-librispeech-asr"
API_TOKEN = ""
headers = {"Authorization": f"Bearer {API_TOKEN}"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        user_input = request.form['user_input']

        # Check if audio files are provided
        audio_files = request.files.getlist('audio_input')
        audio_texts = []

        for index, audio_file in enumerate(audio_files):
            # Process audio file
            audio_path = save_audio_temporarily(audio_file, index)
            audio_text = convert_audio_to_text(audio_path)
            os.remove(audio_path)  # Remove temporary audio file
            audio_texts.append(audio_text)

        # Combine text and audio if both are provided
        combined_input = f"{user_input}\n{' '.join(audio_texts)}" if audio_texts else user_input

        # Generate summary using pipeline with length_penalty
        pipe = pipeline('summarization', model=model_path)
        gen_kwargs = {'length_penalty': 0.3, 'num_beams': 8, "max_length": 128}
        generated_summary = pipe(combined_input, **gen_kwargs)[0]['summary_text']

        # Evaluate the summary
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(generated_summary, combined_input)

        return render_template('index.html', user_input=combined_input, generated_summary=generated_summary, rouge_scores=scores)

def save_audio_temporarily(audio_file, index):
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{index}.wav")
    audio_file.save(temp_audio.name)
    return temp_audio.name

def convert_audio_to_text(audio_path):
    with open(audio_path, "rb") as f:
        data = f.read()
    
    response = requests.post(API_URL, headers=headers, data=data)
    
    # Assuming the response contains the transcribed text
    transcribed_text = response.json().get("text", "Error: Unable to transcribe audio.")
    
    return transcribed_text

if __name__ == '__main__':
    app.run(debug=True)
