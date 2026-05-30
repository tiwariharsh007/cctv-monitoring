"""
Generate alert messages using Google Gemini API with vision capabilities
Analyzes snapshot images and generates meaningful alert descriptions
"""
import google.generativeai as genai
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()


class GeminiAlertGenerator:
    def __init__(self, api_key: str = None):
        """
        Initialize with Gemini API key
        Get API key from: https://aistudio.google.com/apikey
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            print("⚠️ No Gemini API key found!")
            print("Set GEMINI_API_KEY in .env file")
            return

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def generate_alert_message(self, alert_type: str, image_path: str = None, detail: str = None) -> str:
        """
        Generate a meaningful alert message using Gemini vision

        Args:
            alert_type: Type of alert (intrusion, fall, loitering, crowd, running, etc)
            image_path: Path to snapshot image from the incident
            detail: Additional context about the alert

        Returns:
            str: Generated alert message based on image analysis
        """
        if not self.api_key:
            return self._default_message(alert_type, detail)

        # If image exists, use vision to analyze it
        if image_path and os.path.exists(image_path):
            return self._analyze_image(alert_type, image_path)

        # Fallback to text-based message if no image
        return self._generate_text_message(alert_type, detail)

    def _analyze_image(self, alert_type: str, image_path: str) -> str:
        """
        Analyze snapshot image using Gemini Vision and generate alert message
        """
        try:
            # Read and encode the image
            with open(image_path, "rb") as img_file:
                image_data = img_file.read()

            # Determine MIME type
            ext = Path(image_path).suffix.lower()
            mime_types = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }
            mime_type = mime_types.get(ext, "image/jpeg")

            # Create alert-specific prompts
            prompts = {
                "Intrusion": f"""Analyze this surveillance image where unauthorized access was detected.
Generate a concise alert message (2-3 sentences max) that includes:
1) How many people are visible and their appearance/clothing
2) Where they are in the image
3) What they're doing
4) Any concerning behavior or threat indicators
Start with: '🚨 INTRUSION ALERT: '""",

                "Fall": f"""Analyze this surveillance image where a person has fallen.
Generate a concise alert message (2-3 sentences max) that includes:
1) The person's position on the ground
2) Their posture and apparent condition
3) Surrounding context and location
4) Whether help appears urgently needed
Start with: '🚨 FALL ALERT: '""",

                "Crowd": f"""Analyze this surveillance image where multiple people gathered.
Generate a concise alert message (2-3 sentences max) that includes:
1) Exact count of people visible
2) Their arrangement and positioning
3) What they're doing
4) Density and potential safety concerns
Start with: '⚠️ CROWD ALERT: '""",

                "Running": f"""Analyze this surveillance image where running/fast movement was detected.
Generate a concise alert message (2-3 sentences max) that includes:
1) The person's movement and speed
2) Their appearance and direction
3) What triggered the alert
4) Potential emergency context
Start with: '⚠️ RUNNING ALERT: '""",

                "Inactivity": f"""Analyze this surveillance image where a person appears inactive.
Generate a concise alert message (2-3 sentences max) that includes:
1) Person's position and posture
2) How long they appear stationary
3) Their location
4) Potential concern (medical emergency, etc)
Start with: '⚠️ INACTIVITY ALERT: '""",

                "AbandonedObject": f"""Analyze this surveillance image where an abandoned object was detected.
Generate a concise alert message (2-3 sentences max) that includes:
1) Description of the object
2) Where it's located
3) How long it appears abandoned
4) Potential security risk
Start with: '⚠️ ABANDONED OBJECT ALERT: '""",

                "Tailgating": f"""Analyze this surveillance image where tailgating/unauthorized entry was detected.
Generate a concise alert message (2-3 sentences max) that includes:
1) Number of people and their appearance
2) The entry point being used
3) Their behavior (rushed, hiding, etc)
4) Threat level assessment
Start with: '⚠️ TAILGATING ALERT: '""",

                "Accident": f"""Analyze this surveillance image where an accident/collision was detected.
Generate a concise alert message (2-3 sentences max) that includes:
1) Type of incident and objects involved
2) Number of people affected
3) Apparent severity of incident
4) Immediate assistance needed
Start with: '🚨 ACCIDENT ALERT: '""",
            }

            prompt = prompts.get(alert_type, f"""Analyze this surveillance alert image.
Generate a concise alert message (2-3 sentences) describing what's happening, who's involved, where, and why it's concerning.
Alert Type: {alert_type}""")

            # Send image and prompt to Gemini
            response = self.model.generate_content([
                prompt,
                {
                    "mime_type": mime_type,
                    "data": image_data
                }
            ])

            if response.text:
                return response.text.strip()

        except Exception as e:
            print(f"⚠️ Image analysis error: {e}")

        return self._default_message(alert_type, None)

    def _generate_text_message(self, alert_type: str, detail: str) -> str:
        """Generate text-based message when image is unavailable"""
        prompts = {
            "Intrusion": f"Generate a brief security alert message (2 sentences) about unauthorized access/intrusion. {f'Context: {detail}' if detail else ''}",
            "Fall": f"Generate a brief urgent alert (2 sentences) about person falling. {f'Context: {detail}' if detail else ''}",
            "Crowd": f"Generate a brief alert (2 sentences) about crowd gathering. {f'Context: {detail}' if detail else ''}",
            "Running": f"Generate a brief alert (2 sentences) about fast movement/running. {f'Context: {detail}' if detail else ''}",
            "Inactivity": f"Generate a brief alert (2 sentences) about person inactive. {f'Context: {detail}' if detail else ''}",
            "AbandonedObject": f"Generate a brief alert (2 sentences) about abandoned object. {f'Context: {detail}' if detail else ''}",
            "Tailgating": f"Generate a brief alert (2 sentences) about unauthorized entry. {f'Context: {detail}' if detail else ''}",
            "Accident": f"Generate a brief alert (2 sentences) about accident detected. {f'Context: {detail}' if detail else ''}",
        }

        prompt = prompts.get(alert_type, f"Generate a brief alert (2 sentences) for {alert_type}. {f'Context: {detail}' if detail else ''}")

        try:
            response = self.model.generate_content(prompt)
            if response.text:
                return response.text.strip()
        except Exception as e:
            print(f"⚠️ Text generation error: {e}")

        return self._default_message(alert_type, detail)

    def _default_message(self, alert_type: str, detail: str) -> str:
        """Fallback messages when API unavailable"""
        messages = {
            "Intrusion": "🚨 INTRUSION ALERT: Unauthorized access detected in restricted zone.",
            "Fall": "🚨 FALL ALERT: Person has fallen. Immediate assistance may be needed.",
            "Crowd": "⚠️ CROWD ALERT: Multiple people gathered in monitored zone.",
            "Running": "⚠️ RUNNING ALERT: Fast movement detected - possible emergency.",
            "Inactivity": "⚠️ INACTIVITY ALERT: Person inactive for extended period.",
            "AbandonedObject": "⚠️ ABANDONED OBJECT: Unattended item detected in monitored area.",
            "Tailgating": "⚠️ TAILGATING ALERT: Unauthorized entry detected.",
            "Accident": "🚨 ACCIDENT ALERT: Collision or accident detected.",
        }
        return messages.get(alert_type, f"Alert: {alert_type}")


# Initialize globally
alert_gen = None

def init_caption_generator(api_key: str = None):
    """Initialize the alert generator (backward compatible name)"""
    global alert_gen
    alert_gen = GeminiAlertGenerator(api_key)
    return alert_gen

def generate_smart_alert_message(alert_type: str, image_path: str = None) -> str:
    """
    Generate a smart alert message using Gemini Vision API
    Analyzes the snapshot image and creates meaningful alert content for email

    Args:
        alert_type: Type of alert (fall, intrusion, loitering, etc)
        image_path: Path to snapshot image from the incident

    Returns:
        str: Generated alert message based on image analysis
    """
    global alert_gen

    if alert_gen is None:
        alert_gen = GeminiAlertGenerator()

    return alert_gen.generate_alert_message(alert_type, image_path)
