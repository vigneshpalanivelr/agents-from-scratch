"""
1. How to start uvicorn `uvicorn main:app --reload --port 8000`
2. Using ngrok to expose your local server to the internet: `ngrok http 8000`
3. Configure your Twilio WhatsApp sandbox to point to the ngrok URL (e.g https://<your-ngrok-subdomain>.ngrok.io/webhook/whatsapp)
"""
import logging
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse


# Configurations
logging.basicConfig(
    level=logging.INFO,
    # The %(name)s column will dynamically change based on which logger we call
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
)
sys_logger = logging.getLogger("WhatsApp_Bot.SYS")
req_logger = logging.getLogger("WhatsApp_Bot.REQ")
res_logger = logging.getLogger("WhatsApp_Bot.RES")

load_dotenv(override=True)


app = FastAPI()

# Initialize LLMs (Using Groq API as configured previously)
def get_groq_llm(model_name="openai/gpt-oss-20b"):
    return ChatOpenAI(
        model=model_name,
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=2000
    )


sys_logger.info("Starting up LLM connections...")
# extraction_llm = get_groq_llm("openai/gpt-oss-20b")
chat_llm = get_groq_llm("meta-llama/llama-4-scout-17b-16e-instruct")
sys_logger.info("All LLMs successfully initialized and ready.")


@app.get("/")
def read_root():
    req_logger.info("Health check endpoint '/' was pinged.")
    return {"message": "Hello, World!"}


@app.post("/webhook/whatsapp")
async def webhook_whatsapp(request: Request):
    req_logger.info("--- Incoming WhatsApp Webhook Triggered ---")
    """
    request: Incoming HTTP package sent to your server from the outside world
    Twilio sends a lot of metadata when a user messages you (their number, their location, media URLs, etc.).
    The Request object gives you the tools to read all that data.

    By default, FastAPI expects JSON.
    But Twilio sends data as application/x-www-form-urlencoded (like a traditional HTML web form).
    You use await request.form() to parse this specific format.
    """
    form_data = await request.form()
    user_msg = form_data.get("Body")
    from_num = form_data.get("From")
    req_logger.info(f"Sender: {from_num} | Message Received: '{user_msg}'")

    if not user_msg:
        sys_logger.warning(f"Message body was empty from {from_num}. Sending fallback response.")
        reply = "I didn't receive your message. Please try again."
    else:
        sys_logger.info("Invoking LLM to generate a response...")
        response = chat_llm.invoke(user_msg)
        reply = response.content
        sys_logger.info(f"LLM Generation Complete.")

    # Create a TwiML response
    """
    A helper class from the Twilio Python SDK used to easily write TwiML (Twilio Markup Language).
    TwiML is simply XML with special Twilio tags.
    This creates an empty Twilio XML document.
    """
    res_logger.info("Constructing Twilio TwiML (XML) response...")
    twilio_resp = MessagingResponse()
    """
    <Response></Response>
    """
    twilio_resp.message(reply)
    """
    <Response>
        <Message>Hello there!</Message>
    </Response>
    """

    # It represents the outgoing HTTP package your server sends back to the internet.
    # By default, if you just return {"reply": "Hi"} in FastAPI, it automatically formats it as JSON (application/json).
    # However, Twilio will crash if you send it JSON. Twilio explicitly requires XML.
    # The Response class allows you to manually override FastAPI's default behavior and force it to return an XML string.
    xml_output = str(twilio_resp)
    res_logger.info(f"Final Outgoing XML Payload:\n{xml_output}")
    res_logger.info("--- Webhook Lifecycle Complete ---")
    return Response(
        content=xml_output,
        media_type="application/xml"
    )