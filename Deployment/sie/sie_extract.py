from PIL import Image
from sie_sdk import SIEClient
from sie_sdk.types import Item
import io

SIE_URL = "http://localhost:8080"

def image_to_bytes(image: Image.Image) -> bytes:
    with io.BytesIO() as output:
        image.save(output, format="jpeg")
        return output.getvalue()

image = Image.open("invoice_sample.jpg")
image_bytes = image_to_bytes(image)

print(type(image_bytes))  # <class 'bytes'>

# zai-org/GLM-OCR

client = SIEClient(SIE_URL)
# result = client.extract(
#     "urchade/gliner_multi-v2.1",
#     Item(text="Tim Cook is the CEO of Apple."),
#     labels=["person", "organization"]
# )
# print(result["entities"])



# result = client.extract(
#     "PaddlePaddle/PaddleOCR-VL-1.5",
#     Item(images=[image_bytes]))
# print(result)

result = client.extract(
    "IDEA-Research/grounding-dino-base",
    Item(images=[image_bytes]),
    labels=["invoice_number"]
    )
print(result)

bbox = result['objects'][0]['bbox']
print(bbox)  # [42, 390, 49, 20]

# Apply the bounding box to the loaded invoice image and save the crop.
if len(bbox) == 4:
    x0, y0, x2, y2 = bbox
    if x2 > x0 and y2 > y0 and y2 <= image.height:
        crop_box = (x0, y0, x2, y2)
    else:
        crop_box = (x0, y0, x0 + x2, y0 + y2)
    cropped_image = image.crop(crop_box)
    cropped_image.save("invoice_crop.jpg")
    print(f"Saved cropped invoice region to invoice_crop.jpg with box {crop_box}")
else:
    print("Unexpected bbox format; cannot crop image.")



