from flask import Flask, request, jsonify
import random, math, torch, numpy as np
import torch_geometric.nn as pyg_nn
from transformers import BertTokenizer, BertModel
import shap

app = Flask(__name__)

def get_positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

@app.route("/simulate", methods=["POST"])
def simulate_game():
    data = request.json
    team_a, team_b = data.get("team_a"), data.get("team_b")
    what_if = data.get("what_if", {})
    win_prob = round(random.uniform(0.4, 0.8), 3)
    return jsonify({
        "game": f"{team_a} vs {team_b}",
        "confidence": round(random.uniform(3.5, 5
