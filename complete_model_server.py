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
USE_OPENVLA = os.getenv("USE_OPENVLA", "false").lower() == "true"


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

def decode_image_to_tensor(b64_img):
    im_bytes = base64.b64decode(b64_img.encode('utf-8'))
    img = Image.open(io.BytesIO(im_bytes)).convert("RGB")
    return torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
def decode_image(b64_img):
    im_bytes = base64.b64decode(b64_img.encode('utf-8'))
    img = Image.open(io.BytesIO(im_bytes)).convert("RGB")
    return img

# ---------- Chargement du modèle ----------

@app.on_event("startup")
async def load_model():
    global model
    global cfg
    global processor
    if USE_SMOLVLA:
        logger.info("🔧 Chargement du modèle SmolVLA...")
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        model = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        logger.info("✅ SmolVLA chargé avec succès.")
    elif USE_OPENVLA:
        logger.info("🔧 Chargement du modèle OpenVLA...")

        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
        import json

        def get_vla(cfg):
            """Loads and returns a VLA model from checkpoint."""
            # Load VLA checkpoint.
            print("[*] Instantiating Pretrained VLA model")
            print("[*] Loading in BF16 with Flash-Attention Enabled")

            # Load Processor & VLA
            processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
            vla = AutoModelForVision2Seq.from_pretrained(
                cfg.pretrained_checkpoint, 
                attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
                torch_dtype=torch.bfloat16, 
                low_cpu_mem_usage=True, 
                trust_remote_code=True
            ).to("cuda:0")

            # vla = AutoModelForVision2Seq.from_pretrained(
            #     cfg.pretrained_checkpoint,
            #     attn_implementation="flash_attention_2",
            #     torch_dtype=torch.bfloat16,
            #     load_in_8bit=cfg.load_in_8bit,
            #     load_in_4bit=cfg.load_in_4bit,
            #     low_cpu_mem_usage=True,
            #     trust_remote_code=True,
            # )

            # # Move model to device.
            # # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
            # #       already be set to the right devices and casted to the correct dtype upon loading.
            # if not cfg.load_in_8bit and not cfg.load_in_4bit:
            #     vla = vla.to(DEVICE)

            # Load dataset stats used during finetuning (for action un-normalization).
            dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
            if os.path.isfile(dataset_statistics_path):
                with open(dataset_statistics_path, "r") as f:
                    norm_stats = json.load(f)
                vla.norm_stats = norm_stats
            else:
                print(
                    "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
                    "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
                    "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
                )

            return vla
        def get_model(cfg, wrap_diffusion_policy_for_droid=False):
            """Load model for evaluation."""
            if cfg.model_family == "openvla":
                model = get_vla(cfg)
            else:
                raise ValueError("Unexpected `model_family` found in config.")
            print(f"Loaded model: {type(model)}")
            return model
        logger.info("PLOP41")

        from typing import Optional, Union
        from pathlib import Path
        from dataclasses import dataclass

        @dataclass
        class GenerateConfig:
            # fmt: off

            #################################################################################################################
            # Model-specific parameters
            #################################################################################################################
            model_family: str = "openvla"                    # Model family
            pretrained_checkpoint: Union[str, Path] = "openvla/openvla-7b-finetuned-libero-spatial"     # Pretrained checkpoint path
            load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
            load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

            center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

            #################################################################################################################
            # LIBERO environment-specific parameters
            #################################################################################################################
            task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
            num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
            num_trials_per_task: int = 50                    # Number of rollouts per task

            #################################################################################################################
            # Utils
            #################################################################################################################
            run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
            local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

            use_wandb: bool = True                          # Whether to also log results in Weights & Biases
            wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
            wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

            seed: int = 7 
                                               # Random Seed (for reproducibility)
        logger.info("PLOP42")
        def get_processor(cfg):
            """Get VLA model's Hugging Face processor."""
            processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
            return processor
        logger.info("PLOP43")

        cfg=GenerateConfig()
        # [OpenVLA] Set action un-normalization key
        assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
        if "image_aug" in cfg.pretrained_checkpoint:
            assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
        assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
        
        import random
        def set_seed_everywhere(seed: int):
            """Sets the random seed for Python, NumPy, and PyTorch functions."""
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ["PYTHONHASHSEED"] = str(seed)
        # Set random seed
        set_seed_everywhere(cfg.seed)

        # [OpenVLA] Set action un-normalization key
        cfg.unnorm_key = cfg.task_suite_name
        logger.info("PLOP5")

        model = get_model(cfg)
        
        logger.info("PLOP6")
        
        # # Utiliser le OpenVLA pour les tests
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

        logger.info("PLOP7")

        processor = get_processor(cfg)
        logger.info("✅ OpenVLA chargé avec succès.")
    else:
        logger.info("⚙️ Utilisation du DummyVLA")
        model = DummyVLA()

# ---------- Routes API ----------

@app.post("/predict")
async def predict(obs: ObsInput):
    try:
        if model is None:
            raise RuntimeError("Modèle non initialisé")

        images_decoded= np.stack([decode_image(obs.images[key]) for key in sorted(obs.images.keys())]).squeeze()

        if USE_SMOLVLA:
            from lerobot.constants import OBS_STATE
            images_tensor = torch.stack([decode_image_to_tensor(obs.images[key]) for key in sorted(obs.images.keys())])
            state_tensor = torch.tensor(obs.state).float()
            batch = {
                OBS_STATE: state_tensor,
                "task": obs.instruction,
                "camera_front": images_tensor,
            }
        
           
            logger.info("HERE")
            
            action = model.select_action(batch)
            # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
            action = normalize_gripper_action(action, binarize=True)
            logger.info("HERE2")
        elif USE_OPENVLA:
            import tensorflow as tf
            # Initialize system prompt for OpenVLA v0.1.
            OPENVLA_V01_SYSTEM_PROMPT = (
                "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions."
            )
            DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
            ACTION_DIM = 7
            def crop_and_resize(image, crop_scale, batch_size):
                """
                Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
                to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
                distribution shift at test time.

                Args:
                    image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
                        values between [0,1].
                    crop_scale: The area of the center crop with respect to the original image.
                    batch_size: Batch size.
                """
                # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
                assert image.shape.ndims == 3 or image.shape.ndims == 4
                expanded_dims = False
                if image.shape.ndims == 3:
                    image = tf.expand_dims(image, axis=0)
                    expanded_dims = True

                # Get height and width of crop
                new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
                new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

                # Get bounding box representing crop
                height_offsets = (1 - new_heights) / 2
                width_offsets = (1 - new_widths) / 2
                bounding_boxes = tf.stack(
                    [
                        height_offsets,
                        width_offsets,
                        height_offsets + new_heights,
                        width_offsets + new_widths,
                    ],
                    axis=1,
                )

                # Crop and then resize back up
                image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

                # Convert back to 3D Tensor (H, W, C)
                if expanded_dims:
                    image = image[0]

                return image
            def get_action(cfg, model, obs, task_label, processor=None):
                """Queries the model to get an action."""
                if cfg.model_family == "openvla":
                    action = get_vla_action(
                        model, processor, cfg.pretrained_checkpoint, obs, task_label, cfg.unnorm_key, center_crop=cfg.center_crop
                    )
                    assert action.shape == (ACTION_DIM,)
                else:
                    raise ValueError("Unexpected `model_family` found in config.")
                return action
            def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
                """Generates an action with the VLA policy."""
                image = Image.fromarray(obs["full_image"])
                image = image.convert("RGB")

                # (If trained with image augmentations) Center crop image and then resize back up to original size.
                # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
                #            the original height and width by sqrt(0.9) -- not 0.9!
                if center_crop:
                    batch_size = 1
                    crop_scale = 0.9

                    # Convert to TF Tensor and record original data type (should be tf.uint8)
                    image = tf.convert_to_tensor(np.array(image))
                    orig_dtype = image.dtype

                    # Convert to data type tf.float32 and values between [0,1]
                    image = tf.image.convert_image_dtype(image, tf.float32)

                    # Crop and then resize back to original size
                    image = crop_and_resize(image, crop_scale, batch_size)

                    # Convert back to original data type
                    image = tf.clip_by_value(image, 0, 1)
                    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

                    # Convert back to PIL Image
                    image = Image.fromarray(image.numpy())
                    image = image.convert("RGB")

                # Build VLA prompt
                if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
                    prompt = (
                        f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
                    )
                else:  # OpenVLA
                    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

                # Process inputs.
                inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

                # Get action.
                action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
                return action
            def normalize_gripper_action(action, binarize=True):
                """
                Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
                Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
                Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
                the dataset wrapper.

                Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
                """
                # Just normalize the last action to [-1,+1].
                orig_low, orig_high = 0.0, 1.0
                action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

                if binarize:
                    # Binarize to -1 or +1.
                    action[..., -1] = np.sign(action[..., -1])

                return action
            def invert_gripper_action(action):
                """
                Flips the sign of the gripper action (last dimension of action vector).
                This is necessary for some environments where -1 = open, +1 = close, since
                the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
                """
                action[..., -1] = action[..., -1] * -1.0
                return action

            logger.info("PLOP")
            observation = {
                        "full_image": images_decoded,
                        "state":obs.state
                    }
            logger.info("PLOP1")
            
            action = get_action(
                        cfg,
                        model,
                        observation,
                        obs.instruction,
                        processor=processor,
                    )
            logger.info("PLOP11")
            # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
            action = normalize_gripper_action(action, binarize=True)
            logger.info("PLOP2")
            # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
            action = invert_gripper_action(action)
            logger.info("PLOP3")


        else:
            batch = {
                "instruction": obs.instruction,
                "state": obs.state,
                "images": images_decoded,
            }
            action = model.select_action(batch)
            # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
            action = normalize_gripper_action(action, binarize=True)

        return {"action": action.tolist() if isinstance(action, torch.Tensor) else action}

    except Exception as e:
        logger.info("❌ Erreur :",str(e))
        return {"error": str(e)}

@app.get("/status")
async def status():
    return {
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else "None",
        "use_smolvla": USE_SMOLVLA
    }