from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
import base64
import io
from PIL import Image
import os
import logging
logger = logging.getLogger("uvicorn")

# Lire depuis les variables d'environnement, ou par défaut mettre à False
USE_SMOLVLA = os.getenv("USE_SMOLVLA", "false").lower() == "true"


# Initialisation du serveur
app = FastAPI()
model = None  # sera défini au startup

# ---------- Dummy model ----------

class DummyVLA(torch.nn.Module):
    def forward(self, obs):
        return np.random.uniform(-1, 1, size=(7,)).tolist()

    def select_action(self, batch):
        return self.forward(batch)  # compatible avec la logique SmolVLA

# ---------- Input model ----------

class ObsInput(BaseModel):
    state: list  # ex: positions des joints
    images: dict  # clé = nom caméra, valeur = image base64
    instruction: str  # ex: "pick up the red block"

# ---------- Image decoder ----------

def decode_image(b64_img):
    im_bytes = base64.b64decode(b64_img.encode('utf-8'))
    img = Image.open(io.BytesIO(im_bytes)).convert("RGB")
    return torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0

# ---------- Chargement du modèle ----------

@app.on_event("startup")
async def load_model():
    global model
    if USE_SMOLVLA:
        logger.info("🔧 Chargement du modèle SmolVLA...")
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        model = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        logger.info("✅ SmolVLA chargé avec succès.")
    else:
        logger.info("⚙️ Utilisation du DummyVLA")
        model = DummyVLA()

# ---------- Routes API ----------

@app.post("/predict")
async def predict(obs: ObsInput):
    try:
        if model is None:
            raise RuntimeError("Modèle non initialisé")

        images_tensor = torch.stack([decode_image(obs.images[key]) for key in sorted(obs.images.keys())])
        state_tensor = torch.tensor(obs.state).float()

        if USE_SMOLVLA:
            from lerobot.constants import OBS_STATE
            batch = {
                OBS_STATE: state_tensor,
                "task": obs.instruction,
                "camera_front": images_tensor,
            }
        
           
            logger.info("HERE")
            
            action = model.select_action(batch)
            logger.info("HERE2")

        else:
            batch = {
                "instruction": obs.instruction,
                "state": obs.state,
                "images": obs.images,
            }
            action = model.select_action(batch)

        return {"action": action.tolist() if isinstance(action, torch.Tensor) else action}

    except Exception as e:
        logger.info("❌ Erreur :", str(e))
        return {"error": str(e)}

@app.get("/status")
async def status():
    return {
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else "None",
        "use_smolvla": USE_SMOLVLA,
    }