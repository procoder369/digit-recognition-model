import urllib.request
import json
import base64
import io
from PIL import Image, ImageDraw

def test_digit(label, draw_fn):
    img = Image.new("RGB", (280, 280), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw_fn(draw)
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode()
    
    req = urllib.request.Request(
        "http://127.0.0.1:8000/predict",
        data=json.dumps({"image": b64_str}).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        print(f"Target: {label} -> Predicted: {data['prediction']} (Confidence: {data['confidence'] * 100:.1f}%)")

if __name__ == "__main__":
    print("Running automated digit recognition tests against FastAPI...")
    # Test 0
    test_digit("0", lambda d: d.ellipse([(80, 60), (200, 220)], outline=(255, 255, 255), width=22))
    # Test 1
    test_digit("1", lambda d: d.line([(140, 50), (140, 230)], fill=(255, 255, 255), width=24))
    # Test 7
    test_digit("7", lambda d: d.line([(80, 70), (200, 70), (120, 230)], fill=(255, 255, 255), width=22))
    print("All test cases executed successfully!")
