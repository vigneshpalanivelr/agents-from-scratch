"""
1. How to start uvicorn `uvicorn entrypoint:app --reload --port 8000`
2. Using ngrok to expose your local server to the internet: `ngrok http 8000`
3. Configure your Twilio WhatsApp sandbox to point to the ngrok URL (e.g https://<your-ngrok-subdomain>.ngrok.io/webhook/whatsapp)
"""
from langchain_core.messages import HumanMessage

from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse

from helper import download_twilio_image_as_base64
from workflow import build_langgraph_app


import logging
from dotenv import load_dotenv
load_dotenv(override=True)


# Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
)
web_logger = logging.getLogger("WhatsApp.Web")


# Initialize application
web_logger.info("Starting FastAPI Server...")
app = FastAPI()
langgraph_app = build_langgraph_app()


@app.post("/webhook/whatsapp")
async def webhook_whatsapp(request: Request):
    web_logger.info("INCOMING WHATSAPP MESSAGE")

    form_data = await request.form()
    image_url = None

    user_message = form_data.get("Body", "")
    from_number = form_data.get("From")
    num_media = int(form_data.get("NumMedia", 0))

    web_logger.info(f"From: {from_number}")
    web_logger.info(f"Text Body: '{user_message}'")
    web_logger.info(f"Media Attached: {num_media} files")

    # Handle Images
    if num_media > 0:
        media_type = form_data.get("MediaContentType0", "")
        web_logger.info(f"Processing Media Type: {media_type}")

        if "image" in media_type:
            raw_url = form_data.get("MediaUrl0")
            web_logger.info(f"Downloading Image from Twilio: {raw_url}")
            image_url = download_twilio_image_as_base64(raw_url)
            web_logger.info("Image successfully converted to Base64")

    # Prepare Graph Config
    config = {"configurable": {"thread_id": from_number}}
    web_logger.info(f"Invoking Graph with thread_id: {from_number}...")

    # Invoke the Agent Graph
    result = langgraph_app.invoke(
        {
            "messages": [HumanMessage(content=user_message)],
            "image_url": image_url
        },
        config=config
    )

    # Process Graph Output
    result_content = result["messages"][-1].content
    web_logger.info(f"Graph Execution Complete.")

    # Format TwiML
    web_logger.info("Packaging response into Twilio TwiML (XML)...")
    twilio_response = MessagingResponse()
    twilio_response.message(result_content)

    web_logger.info("SENDING RESPONSE TO TWILIO")
    web_logger.info("="*50)

    return Response(
        content=str(twilio_response),
        media_type="application/xml"
    )