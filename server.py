import argparse
import logging
from diffusers.utils import load_image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import flask
from flask import request, jsonify
import waitress
from flask_cors import CORS
import torch
import warnings
import time
import traceback
import os
import yaml

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/config.custom.yaml")
args = parser.parse_args()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

# host = config["local_inference_endpoint"]["host"]
port = config["local_inference_endpoint"]["port"]

local_deployment = config["local_deployment"]
device = config.get("device", "cuda:0") 

PROXY = None
if config["proxy"]:
    PROXY = {
        "https": config["proxy"],
    }

app = flask.Flask(__name__)
CORS(app)

start = time.time()

local_fold = "models"

def load_pipes(local_deployment):
    other_pipes = {}
    standard_pipes = {}
    controlnet_sd_pipes = {}
    if local_deployment in ["full"]:
        other_pipes = {
            "nlpconnect/vit-gpt2-image-captioning":{
                "model": VisionEncoderDecoderModel.from_pretrained(f"{local_fold}/nlpconnect/vit-gpt2-image-captioning"),
                "feature_extractor": ViTImageProcessor.from_pretrained(f"{local_fold}/nlpconnect/vit-gpt2-image-captioning"),
                "tokenizer": AutoTokenizer.from_pretrained(f"{local_fold}/nlpconnect/vit-gpt2-image-captioning"),
                "device": device
            },
        }

    if local_deployment in ["full", "standard"]:
        standard_pipes = {}

    if local_deployment in ["full", "standard", "minimal"]:
        controlnet_sd_pipes = {}

    pipes = {**standard_pipes, **other_pipes, **controlnet_sd_pipes}

    return pipes

pipes = load_pipes(local_deployment)

end = time.time()
during = end - start

print(f"[ ready ] {during}s")

@app.route('/running', methods=['GET'])
def running():
    return jsonify({"running": True})

@app.route('/status/<path:model_id>', methods=['GET'])
def status(model_id):
    disabled_models = ["microsoft/trocr-base-printed", "microsoft/trocr-base-handwritten"]
    if model_id in pipes.keys() and model_id not in disabled_models:
        print(f"[ check {model_id} ] success")
        return jsonify({"loaded": True})
    else:
        print(f"[ check {model_id} ] failed")
        return jsonify({"loaded": False})

@app.route('/models/<path:model_id>', methods=['POST'])
def models(model_id):
    while "using" in pipes[model_id] and pipes[model_id]["using"]:
        print(f"[ inference {model_id} ] waiting")
        time.sleep(0.1)
    pipes[model_id]["using"] = True
    print(f"[ inference {model_id} ] start")

    start = time.time()

    pipe = pipes[model_id]["model"]
    
    if "device" in pipes[model_id]:
        try:
            pipe.to(pipes[model_id]["device"])
        except:
            pipe.device = torch.device(pipes[model_id]["device"])
            pipe.model.to(pipes[model_id]["device"])
    
    result = None
    try:
        # image to text
        if model_id == "nlpconnect/vit-gpt2-image-captioning":
            image = load_image(request.get_json()["img_url"]).convert("RGB")
            pixel_values = pipes[model_id]["feature_extractor"](images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(pipes[model_id]["device"])
            generated_ids = pipe.generate(pixel_values, **{"max_length": 200, "num_beams": 1})
            generated_text = pipes[model_id]["tokenizer"].batch_decode(generated_ids, skip_special_tokens=True)[0]
            result = {"generated text": generated_text}

        # TODO: people can add new models here

    except Exception as e:
        print(e)
        traceback.print_exc()
        result = {"error": {"message": "Error when running the model inference."}}

    if "device" in pipes[model_id]:
        try:
            pipe.to("cpu")
            torch.cuda.empty_cache()
        except:
            pipe.device = torch.device("cpu")
            pipe.model.to("cpu")
            torch.cuda.empty_cache()

    pipes[model_id]["using"] = False

    if result is None:
        result = {"error": {"message": "model not found"}}
    
    end = time.time()
    during = end - start
    print(f"[ complete {model_id} ] {during}s")
    print(f"[ result {model_id} ] {result}")

    return jsonify(result)


if __name__ == '__main__':
    # temp folders
    if not os.path.exists("public/audios"):
        os.makedirs("public/audios")
    if not os.path.exists("public/images"):
        os.makedirs("public/images")
    if not os.path.exists("public/videos"):
        os.makedirs("public/videos")
        
    waitress.serve(app, host="0.0.0.0", port=port)
