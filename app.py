from flask import Flask, request, jsonify, render_template
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

app = Flask(__name__)

model_dir = "lsylsy99/m2m_kovi_translate"
tokenizer = M2M100Tokenizer.from_pretrained(model_dir)
model = M2M100ForConditionalGeneration.from_pretrained(model_dir)

@app.route("/translate", methods=["GET"])
def home():
    return render_template('index.html')

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text = data.get("text", "")
    source_lang = data.get("source_lang", "ko")
    target_lang = data.get("target_lang", "vi")
    
    tokenizer.src_lang = source_lang
    
    inputs = tokenizer(text, return_tensors="pt")
    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    
    return jsonify({"translated_text": translated_text[7:]})

if __name__ == "__main__":
    app.run(port=5000, debug = True)
