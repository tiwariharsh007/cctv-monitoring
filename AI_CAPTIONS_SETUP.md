# 🚨 AI-Enhanced Alert Captions - Setup Guide

## What's New?

When alerts are triggered (FALL, INTRUSION, LOITERING, etc.), the system now:

1. ✅ Captures a snapshot of the frame
2. ✅ Sends it to Hugging Face's LLaVA model
3. ✅ Generates a smart, descriptive caption
4. ✅ Sends an enhanced email with the caption + snapshot

## Before and After

### Before:

```
Email Subject: ⚠️ Alert
Message: "Fall @ 2024-01-15 14:32:45"
```

### After:

```
Email Subject: 🚨 FALL DETECTED
Message: "Person lying motionless on the floor near the entrance.
Possible fall detected requiring immediate assistance."
Attachment: Snapshot image
```

---

## Setup Instructions

### 1. Get Hugging Face API Key (Free)

1. Go to: https://huggingface.co/settings/tokens
2. Sign up or login
3. Click "New token"
4. Copy your token (you'll use this in the next step)

### 2. Create `.env` File

In the project root directory, create a file named `.env`:

```
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
```

Replace `hf_xxx...` with your actual API key from Hugging Face.

### 3. Install New Dependencies

```bash
pip install -r requirements.txt
```

New packages added:

- `huggingface-hub` - For API calls
- `python-dotenv` - For loading .env file
- `pillow` - For image handling

---

## How It Works

### File Structure

```
RealTimeSurveillanceSystem/
├── caption_generator.py    # 🆕 AI caption generation
├── .env                    # 🆕 API key (local only)
├── .env.example            # 🆕 Example env file
├── snapshots/              # 🆕 Saved alert snapshots
├── main.py                 # Updated with snapshot saving
├── alerts.py               # Updated with image attachment support
└── ...
```

### Alert Flow

```
Event Detected (e.g., FALL)
    ↓
save_snapshot(frame, "fall", camera_id)
    ↓ Saves to snapshots/camera_id_fall_YYYYMMDD_HHMMSS.jpg
    ↓
handle_alert() → caption_generator.generate_caption()
    ↓ Calls Hugging Face API with image
    ↓
generate_smart_alert_message() → Enhanced message
    ↓ "Person lying on floor..."
    ↓
send_email_alert() with image attachment
    ↓ Sends email with caption + snapshot
```

---

## Running the System

### Option 1: With Hugging Face (Recommended)

```bash
# Make sure .env is set up with API key
python main.py
```

Emails will now include:

- ✅ AI-generated smart captions
- ✅ Snapshot images

### Option 2: Fallback Mode (Without API Key)

If you don't have the API key, the system will still work:

- Emails will use generic messages ("Alert triggered")
- **But no** AI captions or images

---

## Cost & Rate Limiting

### Free Tier

- **Cost**: FREE ✅
- **Limit**: ~50 requests/day (for API inference)
- **Speed**: ~10-30 seconds per image

### Paid Tier (Optional)

- Start at ~$0.01 per image
- Unlimited requests
- Faster responses

---

## Troubleshooting

### API Key Not Working

```
⚠️ No Hugging Face API key found!
```

**Fix**: Make sure `.env` file exists and has valid key

### API Timeout

```
⚠️ API timeout - using default message
```

**Fix**: Hugging Face servers busy. Emails will still send without caption.

### Image Not Attached

```
⚠️ Could not attach image: [error]
```

**Fix**: Check snapshots/ folder permissions, ensure enough disk space

---

## Advanced Usage

### Modify Alert Messages

Edit `caption_generator.py`, the `generate_smart_alert_message()` function:

```python
alert_messages = {
    "fall": f"Custom message: {caption}",
    "intrusion": f"Custom message: {caption}",
    # Add more...
}
```

### Add Custom Instructions

Modify the prompt in `caption_generator.py`:

```python
"text": "Your custom instruction here. Caption: {image}"
```

### Monitor Snapshots

All snapshots are saved to `snapshots/`:

```
snapshots/
├── store_front_fall_20240115_143245.jpg
├── back_exit_intrusion_20240115_143300.jpg
└── ...
```

---

## Summary

✅ **Implemented**: AI captions for all alert types  
✅ **Integrated**: Email with snapshot attachment  
✅ **Configured**: Hugging Face API integration  
✅ **Tested**: Fallback mode without API key

Happy monitoring! 📹
