import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load Hugging Face Model
MODEL_NAME = "roberta-large-mnli"
CACHE_DIR = "./model_cache"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model.eval()
except Exception as e:
    raise gr.Error(f"Failed to load model: {str(e)}")

# Initialize Groq API
try:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    raise gr.Error(f"Failed to initialize Groq client: {str(e)}")

def detect_bias(text):
    if not text.strip():
        return {"error": "Please enter some text to analyze"}
    
    try:
        # Bias detection
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
        bias_score = prediction[0][1].item() * 100  # Convert to percentage
        
        # Get Analysis from Groq API
        groq_response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """You are a bias detection expert. Analyze the given text for any type of bias 
                    (gender, racial, political, religious, etc.). Identify:
                    1. The type(s) of bias present
                    2. Why it might be problematic
                    3. A more neutral alternative version
                    Keep the analysis concise and professional."""
                },
                {
                    "role": "user", 
                    "content": f"Text to analyze: '{text}'"
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=500
        )
        analysis = groq_response.choices[0].message.content
        
        # Extract neutral alternative (assuming it's after "Neutral alternative:")
        neutral_alternative = analysis.split("Neutral alternative:")[-1].strip() if "Neutral alternative:" in analysis else "Not provided"
        
        return {
            "bias_score": f"{bias_score:.2f}%",
            "analysis": analysis,
            "neutral_alternative": neutral_alternative
        }
    except Exception as e:
        return {"error": f"An error occurred during analysis: {str(e)}"}

def interface(text):
    result = detect_bias(text)
    if "error" in result:
        return [f"**Error:** {result['error']}", "", ""]
    
    return [
        f"**Bias Probability Score:** {result['bias_score']}",
        result['analysis'],
        result['neutral_alternative']
    ]

with gr.Blocks(title="Advanced Bias Detection AI") as demo:
    gr.Markdown("# Advanced Bias Detection AI - XAS")
    gr.Markdown("Detect various types of biases in text with AI-powered analysis")
    
    with gr.Row():
        input_text = gr.Textbox(
            label="Input Text",
            placeholder="Enter text to analyze for potential biases...",
            lines=5
        )
    
    with gr.Row():
        analyze_btn = gr.Button("Analyze Text")
    
    with gr.Row():
        with gr.Column():
            score_output = gr.Markdown(label="Bias Score")
        with gr.Column():
            analysis_output = gr.Textbox(label="Bias Analysis", lines=6)
        with gr.Column():
            neutral_output = gr.Textbox(label="Neutral Alternative", lines=3)
    
    examples = gr.Examples(
        examples=[
            ["Women are too emotional to be effective leaders"],
            ["People from that region are all lazy and untrustworthy"],
            ["The younger generation doesn't understand the value of hard work"],
            ["All politicians are corrupt and only care about power"]
        ],
        inputs=input_text,
        label="Example Texts"
    )
    
    analyze_btn.click(
        fn=interface,
        inputs=input_text,
        outputs=[score_output, analysis_output, neutral_output]
    )

if __name__ == "__main__":
    demo.launch()