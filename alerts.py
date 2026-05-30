import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import os
from dotenv import load_dotenv

load_dotenv()

# Set EMAIL_ALERTS=true in .env to enable real email sending.
EMAIL_ALERTS_ENABLED = os.getenv("EMAIL_ALERTS", "false").lower() == "true"


def send_email_alert(subject: str, message: str, image_path: str = None):
    if not EMAIL_ALERTS_ENABLED:
        print(f"📧 [Email disabled] {subject}")
        return

    sender   = os.getenv("EMAIL_SENDER",   "")
    receiver = os.getenv("EMAIL_RECEIVER",  "")
    password = os.getenv("EMAIL_PASSWORD",  "")

    if not sender or not receiver or not password:
        print("⚠️  Email credentials missing in .env — skipping send")
        return

    try:
        msg = MIMEMultipart()
        msg["From"]    = sender
        msg["To"]      = receiver
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain", "utf-8"))

        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                part = MIMEImage(f.read())
                part.add_header("Content-Disposition", "attachment",
                                filename=os.path.basename(image_path))
                msg.attach(part)
            print(f"📎 Snapshot attached: {image_path}")

        # Use port 465 with SSL for better reliability
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as server:
                server.login(sender, password)
                server.send_message(msg)
        except (smtplib.SMTPException, OSError, TimeoutError):
            # Fallback to port 587 with STARTTLS if 465 fails
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as server:
                server.starttls()
                server.login(sender, password)
                server.send_message(msg)

        print("✅ Email sent")

    except smtplib.SMTPAuthenticationError:
        print(f"⚠️  Email failed: Invalid email or password")
    except smtplib.SMTPException as e:
        print(f"⚠️  Email failed: SMTP error - {str(e)[:100]}")
    except (OSError, TimeoutError) as e:
        print(f"⚠️  Email failed: Network error - {str(e)[:100]}")
    except Exception as e:
        print(f"⚠️  Email failed: {type(e).__name__} - {str(e)[:100]}")


def send_whatsapp_alert(message: str):
    """Optional WhatsApp via Twilio — configure TWILIO_SID, TWILIO_TOKEN,
    TWILIO_FROM, TWILIO_TO in .env to enable."""
    sid   = os.getenv("TWILIO_SID", "")
    token = os.getenv("TWILIO_TOKEN", "")
    from_ = os.getenv("TWILIO_FROM", "")
    to_   = os.getenv("TWILIO_TO", "")

    if not all([sid, token, from_, to_]):
        return

    try:
        from twilio.rest import Client
        client = Client(sid, token)
        msg = client.messages.create(body=message, from_=from_, to=to_)
        print(f"✅ WhatsApp sent: {msg.sid}")
    except ImportError:
        print("⚠️  twilio package not installed — pip install twilio")
    except Exception as e:
        print(f"⚠️  WhatsApp failed: {e}")
