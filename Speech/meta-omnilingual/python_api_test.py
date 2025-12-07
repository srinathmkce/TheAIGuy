import requests

# Health check
health_url = "http://localhost:8080/health"
response = requests.get(health_url)
print(response.json())

# API test
url = "http://localhost:8080/v1/audio/speech"
payload = {
    "text": "The API layer makes this incredibly efficient.",
    "voice": "bm_george",
    "speed": 1.5
}

response = requests.post(url, json=payload)

with open(f"api_output_{payload['voice']}.wav", "wb") as f:
    f.write(response.content)
    print("Audio saved to api_output.wav")
