import base64
import requests
import os
from dotenv import load_dotenv

# Ensure environment variables are loaded in this specific file
load_dotenv(override=True)

def download_twilio_image_as_base64(media_url: str) -> str:
    # 1. Fetch credentials
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")

    # 2. Safety Check: Did Python actually find the .env file?
    if not sid or not token:
        raise ValueError("❌ Missing Twilio credentials! Python cannot find TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN.")

    # 3. Clean the credentials (removes hidden spaces or newlines from copy-pasting)
    sid = sid.strip()
    token = token.strip()

    print(f"\n[Helper] Attempting to securely download image using SID: {sid[:5]}...")

    # 4. Make the secure request to Twilio
    response = requests.get(
        media_url,
        auth=(sid, token),
        allow_redirects=True
    )

    # 5. Provide a detailed error if Twilio still rejects it
    if response.status_code != 200:
        print(f"❌ Twilio Download Failed! Status Code: {response.status_code}")
        print(f"❌ Twilio Reason: {response.text}")

    # Trigger the crash if it failed, but now we have the printed logs above!
    response.raise_for_status()

    # 6. Convert to Base64
    encoded = base64.b64encode(response.content).decode("utf-8")
    mime_type = response.headers.get("Content-Type", "image/jpeg")

    return f"data:{mime_type};base64,{encoded}"