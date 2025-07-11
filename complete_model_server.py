# Serveur FastAPI pour tester un VLA (dummy ou SmolVLA) avec LIBERO

from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
import uvicorn
import numpy as np
import base64
import io
from PIL import Image
import os

# USE_SMOLVLA = os.getenv("USE_SMOLVLA", "false").lower() == "true"
USE_SMOLVLA = False  # For testing purposes, set to True to use SmolVLA

# ---------- 1. MODELES ----------

class DummyVLA(torch.nn.Module):
    def forward(self, obs):
        # Retourne une action constante ou aléatoire (ex: 7DOF + gripper)
        return np.random.uniform(-1, 1, size=(8,)).tolist()

if USE_SMOLVLA:
    from lerobot.models import load_smolvla
    model = load_smolvla("0.45B")
    model.eval()
else:
    model = DummyVLA()

# ---------- 2. API ----------

app = FastAPI()

class ObsInput(BaseModel):
    state: list  # états proprioceptifs (ex: joint positions)
    images: dict  # clé = nom caméra, valeur = image encodée base64
    instruction: str  # phrase en langage naturel


def decode_image(b64_img):
    im_bytes = base64.b64decode(b64_img.encode('utf-8'))
    img = Image.open(io.BytesIO(im_bytes)).convert("RGB")
    return torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0


@app.post("/predict")
async def predict(obs: ObsInput):
    try:
        # 1. Traitement des entrées
        images_tensor = torch.stack([decode_image(obs.images[key]) for key in sorted(obs.images.keys())])
        state_tensor = torch.tensor(obs.state).float()

        # 2. Appel modèle
        if USE_SMOLVLA:
            action = model(obs.instruction, images_tensor, state_tensor)
        else:
            action = model({"instruction": obs.instruction, "state": obs.state, "images": obs.images})

        # 3. Renvoi
        return {"action": action}

    except Exception as e:
        return {"error": str(e)}