import requests
import numpy as np
import cv2
import os

# Create dummy image
img = np.zeros((400, 600, 3), dtype=np.uint8)
img[:] = 128 # gray
cv2.imwrite("test_dummy.jpg", img)

url = "http://localhost:5000/assess"
with open("test_dummy.jpg", "rb") as f:
    response = requests.post(url, files={"image": f})

print(f"Status: {response.status_code}")
print(f"Response text: {response.text}")

os.remove("test_dummy.jpg")
