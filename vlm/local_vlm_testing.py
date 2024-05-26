import base64
import json
from typing import Dict, List
import pandas as pd
import requests
from tqdm import tqdm
from pathlib import Path
from scoring.vlm_eval import vlm_eval
from dotenv import load_dotenv
import os
import io
from PIL import Image
from PIL import ImageDraw
from src.VLMManager import VLMManager

load_dotenv()

TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")

def main():
    input_dir = Path("/home/neonu/OneDrive/Lecture notes/Semester 7/Brainhack/advanced")
    # input_dir = Path("inputs").absolute()
    # input_dir = Path(f"../../{TEAM_TRACK}")
    # input_dir = Path(f"../../data/{TEAM_TRACK}/train")
    # results_dir = Path(f"/home/jupyter/{TEAM_NAME}")
    results_dir = Path("results")

    results_dir.mkdir(parents=True, exist_ok=True)
    instances = []
    truths = []
    counter = 0

    with open(input_dir / "vlm.jsonl", "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            instance = json.loads(line.strip())
            with open(input_dir / "images" / instance["image"], "rb") as file:
                image_bytes = file.read()
                for annotation in instance["annotations"]:
                    instances.append(
                        {
                            "key": counter,
                            "caption": annotation["caption"],
                            "b64": base64.b64encode(image_bytes).decode("ascii"),
                        }
                    )
                    truths.append(
                        {
                            "key": counter,
                            "caption": annotation["caption"],
                            "bbox": annotation["bbox"],
                        }
                    )
                    counter += 1

    assert len(truths) == len(instances)
    results = run_batched(instances)
    df = pd.DataFrame(results)
    assert len(truths) == len(results)
    df.to_csv(results_dir / "vlm_results.csv", index=False)
    # calculate eval
    eval_result = vlm_eval(
        [truth["bbox"] for truth in truths],
        [result["bbox"] for result in results],
    )
    print(f"IoU@0.5: {eval_result}")

def flatten(lst):
    if not lst:
        return lst
    if type(lst[0]) == list and len(lst[0]) > 0:
        return flatten(lst[0]) + flatten(lst[1:])
    return lst[:1] + flatten(lst[1:])

def identify(instance):
    """
    Performs Object Detection and Identification given an image frame and a text query.
    """
    vlm_manager = VLMManager()
    # get base64 encoded string of image, convert back into bytes
    input_json = json.loads(instance)
    image = Image.open(io.BytesIO(base64.b64decode(input_json["instances"][0]["b64"])))
    draw = ImageDraw.Draw(image)
    predictions = []
    for instance in input_json["instances"]:
        # each is a dict with one key "b64" and the value as a b64 encoded string
        image_bytes = base64.b64decode(instance["b64"])

        results = vlm_manager.identify(image_bytes, instance["caption"])
        bbox = results["boxes"].tolist()
        if bbox == []:
            bbox = [0, 0, 0, 0]
        else:
            bbox = flatten(bbox)
        predictions.append(bbox)
        
        # Image preview
        label = instance["caption"]
        xmin, ymin, xmax, ymax = bbox
        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
        draw.text((xmin, ymin), f"{label}", fill="white")
        
    # Save image to folder
    image.save(f"results/{input_json['instances'][0]['key']}.png")
    return {"predictions": predictions}

def run_batched(
    instances: List[Dict[str, str | int]], batch_size: int = 4
) -> List[Dict[str, str | int]]:
    # split into batches
    results = []
    for index in tqdm(range(0, len(instances), batch_size)):
        _instances = instances[index : index + batch_size]
        response = identify(
            json.dumps(
                {
                    "instances": [
                        {field: _instance[field] for field in ("key", "caption", "b64")}
                        for _instance in _instances
                    ]
                }
            ),
        )
        _results = [i for i in response["predictions"]]
        results.extend(
            [
                {
                    "key": _instances[i]["key"],
                    "bbox": _results[i],
                }
                for i in range(len(_instances))
            ]
        )
    return results


if __name__ == "__main__":
    main()
