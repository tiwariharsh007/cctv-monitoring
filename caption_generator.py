"""
Generate captions for alert snapshots using Hugging Face API
"""
import requests
import base64
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class CaptionGenerator:
    def __init__(self, hf_api_key: str = None):
        """
        Initialize with Hugging Face API key
        Get free API key from: https://huggingface.co/settings/tokens
        """
        self.hf_api_key = hf_api_key or os.getenv("HUGGINGFACE_API_KEY")
        
        if not self.hf_api_key:
            print("⚠️ No Hugging Face API key found!")
            print("Set HUGGINGFACE_API_KEY environment variable")
            print("Or pass it to CaptionGenerator(hf_api_key='xxx')")
        
        # Using LLaVA - excellent for surveillance scenarios
        self.model_url = "https://api-inference.huggingface.co/models/llava-hf/llava-1.5-7b-hf"
    
    def generate_caption(self, image_path: str, alert_type: str = None) -> str:
        """
        Generate a descriptive caption for an image
        
        Args:
            image_path: Path to the image file
            alert_type: Type of alert (fall, intrusion, loitering, crowd, etc)
        
        Returns:
            str: Generated caption description
        """
        if not self.hf_api_key:
            return "Alert triggered"
        
        # Create detailed prompts based on alert type
        prompts = {
            "fall": "A person has fallen in this surveillance footage. Describe in detail: 1) Who fell (male/female, appearance), 2) Where they fell (location/objects nearby), 3) Their posture/position, 4) Any injuries visible, 5) Surrounding context",
            "intrusion": "This shows an unauthorized person in a restricted area. Describe: 1) How many people, 2) Their appearance and clothing, 3) Their actions and behavior, 4) What area they're in, 5) Any threat indicators",
            "loitering": "A person is loitering in a monitored area. Provide: 1) How long they appear to be there, 2) Their appearance and clothing, 3) What they're doing, 4) The location they're at, 5) Why this might be suspicious",
            "crowd": "Multiple people have gathered. Describe: 1) Exact number of people, 2) Their arrangement/clustering, 3) What they're doing, 4) The location, 5) Density and movement patterns",
            "inactivity": "A person appears inactive/stationary. Detail: 1) Their exact position, 2) How long they've been there, 3) Their posture, 4) The location, 5) Any potential concern indicators"
        }
        
        # Use specific prompt for alert type, or default detailed prompt
        if alert_type and alert_type in prompts:
            prompt = prompts[alert_type]
        else:
            prompt = "Provide a detailed surveillance alert description. Include: 1) People count and descriptions, 2) Their exact actions and behavior, 3) Location/area details, 4) Any objects or hazards, 5) Threat assessment"
        
        try:
            # Read and encode image
            with open(image_path, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode("utf-8")
            
            # Determine image format
            ext = Path(image_path).suffix.lower()
            format_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".bmp": "image/bmp",
            }
            media_type = format_map.get(ext, "image/jpeg")
            
            # Prepare payload
            payload = {
                "inputs": {
                    "image": f"data:{media_type};base64,{image_data}",
                    "text": prompt
                }
            }
            
            headers = {
                "Authorization": f"Bearer {self.hf_api_key}",
                "Content-Type": "application/json"
            }
            
            # Call API — short timeout, no retries so offline use doesn't block
            session = requests.Session()
            session.max_redirects = 3
            adapter = requests.adapters.HTTPAdapter(max_retries=0)
            session.mount("https://", adapter)
            response = session.post(
                self.model_url,
                headers=headers,
                json=payload,
                timeout=5,
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract caption from response
                if isinstance(result, list) and len(result) > 0:
                    caption = result[0].get("generated_text", "Alert triggered").strip()
                    return caption
                elif isinstance(result, dict):
                    caption = result.get("generated_text", "Alert triggered")
                    return caption.strip()
            else:
                print(f"⚠️ API Error: {response.status_code}")
                print(response.text)
                return "Alert triggered"
        
        except requests.exceptions.Timeout:
            print("⚠️ API timeout - using default message")
            return "Alert triggered (timeout)"
        except FileNotFoundError:
            print(f"⚠️ Image not found: {image_path}")
            return "Alert triggered"
        except Exception as e:
            print(f"⚠️ Caption generation error: {e}")
            return "Alert triggered"


# Initialize globally
caption_gen = None

def init_caption_generator(api_key: str = None):
    """Initialize the caption generator"""
    global caption_gen
    caption_gen = CaptionGenerator(api_key)
    return caption_gen

def generate_smart_alert_message(alert_type: str, image_path: str = None) -> str:
    """
    Generate a smart alert message with AI caption
    
    Args:
        alert_type: Type of alert (fall, intrusion, loitering, etc)
        image_path: Path to snapshot image
    
    Returns:
        str: Enhanced alert message
    """
    global caption_gen
    
    if caption_gen is None:
        caption_gen = CaptionGenerator()
    
    if image_path and os.path.exists(image_path):
        # Pass alert_type to generate more specific captions
        caption = caption_gen.generate_caption(image_path, alert_type)
    else:
        caption = None
    
    # Create context-aware messages
    alert_messages = {
        "fall":      f"🚨 FALL DETECTED\n{caption}"      if caption else "Person has fallen",
        "intrusion": f"🚨 INTRUSION DETECTED\n{caption}" if caption else "Unauthorized access detected",
        "loitering": f"⚠️ LOITERING DETECTED\n{caption}" if caption else "Person loitering in restricted area",
        "crowd":     f"⚠️ CROWD ALERT\n{caption}"        if caption else "Crowd detected in monitored area",
        "inactivity":f"⚠️ INACTIVITY DETECTED\n{caption}" if caption else "Person inactive for extended period",
        "fighting":  f"🚨 FIGHT DETECTED\n{caption}"     if caption else "Physical altercation detected between persons",
    }
    
    return alert_messages.get(alert_type, f"Alert: {alert_type}\n{caption}" if caption else f"Alert: {alert_type}")
