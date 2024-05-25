import base64
from fastapi import FastAPI, Request

from VLMManager import VLMManager


app = FastAPI()

vlm_manager = VLMManager()

def flatten(lst):
    if not lst:
        return lst
    if type(lst[0]) == list and len(lst[0]) > 0:
        return flatten(lst[0]) + flatten(lst[1:])
    return lst[:1] + flatten(lst[1:])

@app.get("/health")
def health():
    return {"message": "health ok"}


@app.post("/identify")
async def identify(instance: Request):
    """
    Performs Object Detection and Identification given an image frame and a text query.
    """
    # get base64 encoded string of image, convert back into bytes
    input_json = await instance.json()

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

    return {"predictions": predictions}
