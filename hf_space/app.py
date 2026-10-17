"""Hugging Face Space : recommandations marketing par TinyLlama 1.1B + adaptateur LoRA.

Le prompt reproduit exactement le format utilisé à l'entraînement (ChatML TinyLlama,
même message système, mêmes champs KPI). Appelé par l'onglet « Fine-tuned AI » de
l'app Streamlit via l'API Gradio (api_name="recommend").
"""
import os

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:  # ZeroGPU : GPU alloué seulement pendant l'appel de la fonction décorée
    import spaces
    gpu = spaces.GPU(duration=60)
except ImportError:  # CPU basic ou exécution locale
    def gpu(fn):
        return fn

BASE_MODEL = os.getenv("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
ADAPTER = os.getenv("ADAPTER", "Doers97/marketing-lora-tinyllama")
SYSTEM = ("You are an expert marketing analyst for a Swiss media company. "
          "You analyze campaign KPIs and provide actionable, data-driven recommendations "
          "in a concise, business-ready format.")
CHANNELS = ["social", "search", "influencer", "media", "email"]
SEGMENTS = ["B2B", "B2C", "Premium", "Mass Market"]

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
model = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()  # fusion : inférence plus rapide
model.eval()


def build_prompt(channel, segment, spend, impressions, ctr, cvr, roi, roas, cpl, revenue):
    instruction = (
        "Analyze the following marketing campaign KPIs and provide strategic recommendations:\n\n"
        f"Channel: {channel}\n"
        f"Segment: {segment}\n"
        f"Budget (CHF): {float(spend):,.0f}\n"
        f"Impressions: {int(impressions):,}\n"
        f"CTR: {float(ctr):.2f}%\n"
        f"CVR: {float(cvr):.2f}%\n"
        f"ROI: {float(roi):.1f}%\n"
        f"ROAS: {float(roas):.2f}x\n"
        f"CPL: CHF {float(cpl):.0f}\n"
        f"Revenue: CHF {float(revenue):,.0f}"
    )
    return f"<|system|>\n{SYSTEM}\n<|user|>\n{instruction}\n<|assistant|>\n"


@gpu
def recommend(channel, segment, spend, impressions, ctr, cvr, roi, roas, cpl, revenue):
    if channel not in CHANNELS or segment not in SEGMENTS:
        raise gr.Error(f"channel ∈ {CHANNELS}, segment ∈ {SEGMENTS}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    prompt = build_prompt(channel, segment, spend, impressions, ctr, cvr, roi, roas, cpl, revenue)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=220, do_sample=False,
                             repetition_penalty=1.05, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.split("<|user|>")[0].strip()  # coupe si le modèle enchaîne un nouveau tour


with gr.Blocks(title="Marketing LoRA — TinyLlama") as demo:
    gr.Markdown("## Marketing recommendations — TinyLlama 1.1B + LoRA\n"
                f"Adapter: [{ADAPTER}](https://huggingface.co/{ADAPTER})")
    with gr.Row():
        channel = gr.Dropdown(CHANNELS, value="social", label="Channel")
        segment = gr.Dropdown(SEGMENTS, value="B2C", label="Segment")
        spend = gr.Number(10000, label="Budget (CHF)")
        impressions = gr.Number(200000, label="Impressions", precision=0)
    with gr.Row():
        ctr = gr.Number(2.5, label="CTR (%)")
        cvr = gr.Number(5.0, label="CVR (%)")
        roi = gr.Number(45.0, label="ROI (%)")
        roas = gr.Number(1.45, label="ROAS (x)")
        cpl = gr.Number(250, label="CPL (CHF)")
        revenue = gr.Number(14500, label="Revenue (CHF)")
    btn = gr.Button("Generate", variant="primary")
    output = gr.Textbox(label="Recommendation", lines=14)
    btn.click(recommend, [channel, segment, spend, impressions, ctr, cvr, roi, roas, cpl, revenue],
              output, api_name="recommend")

if __name__ == "__main__":
    demo.queue(max_size=8).launch()
