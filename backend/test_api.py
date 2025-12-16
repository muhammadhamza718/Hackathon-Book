import requests
import json

# Test the Hugging Face backend
url = "https://muhammadshamza7718-physical-ai-backend.hf.space/api/ask"

payload = {
    "question": "What is ROS2?"
}

print(f"Testing: {url}")
print(f"Payload: {json.dumps(payload, indent=2)}")
print("\nSending request...")

try:
    response = requests.post(url, json=payload, timeout=30)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"\nError: {e}")
