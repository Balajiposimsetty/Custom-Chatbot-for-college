import torch
from flask import Flask, request, jsonify, render_template
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import json

app = Flask(__name__)

dataset_path = r"C:\Users\bhava\Desktop\Iportant\chatbot-project\chatbotdataset.json"
model_path = r"C:\Users\bhava\Desktop\Iportant\chatbot-project\fine_tuned_distilbert_model.pt"

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)
if 'input' not in df.columns or 'output' not in df.columns:
    raise ValueError("The JSON must contain 'input' and 'output' keys.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(device)
df['input_embedding'] = df['input'].apply(lambda x: sbert_model.encode(x, convert_to_tensor=True, device=device))

loaded_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(df)).to(device)
loaded_model.load_state_dict(torch.load(model_path, map_location=device))
loaded_model.eval()

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
role_prompt = "You are a virtual assistant for the CSE department."

def chatbot_predict(input_text):
    input_embedding = sbert_model.encode(input_text, convert_to_tensor=True, device=device)
    all_embeddings = torch.stack(df['input_embedding'].tolist()).to(device)
    cosine_scores = util.pytorch_cos_sim(input_embedding, all_embeddings)
    best_match_idx = torch.argmax(cosine_scores).item()
    input_text_with_role = role_prompt + " " + input_text
    inputs = tokenizer(input_text_with_role, return_tensors='pt', truncation=True, padding='max_length', max_length=128).to(device)
    with torch.no_grad():
        logits = loaded_model(**inputs).logits
    predicted_class = logits.argmax(dim=-1).item()
    output_response = df.iloc[best_match_idx]['output']
    return output_response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['input']
    bot_response = chatbot_predict(user_input)
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
