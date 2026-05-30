# 📹 Real-Time Surveillance System - Project Overview

## 🎯 Project Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SURVEILLANCE SYSTEM                      │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
            ┌───▼───┐   ┌────▼───┐   ┌────▼────┐
            │ VIDEO │   │ PERSON │   │  POSE   │
            │ INPUT │   │DETECTOR│   │DETECTOR │
            └───┬───┘   └────┬───┘   └────┬────┘
                │             │            │
                └─────────────┼────────────┘
                              │
                        ┌─────▼──────┐
                        │   TRACKER  │
                        │  & COUNTER │
                        └─────┬──────┘
                              │
                ┌─────────────┼──────────────────┐
                │             │                  │
           ┌────▼────┐   ┌────▼─────┐   ┌──────▼──────┐
           │DETECTORS│   │ SNAPSHOT │   │ALERTS LOGIC │
           │(Fall,   │   │  SAVER   │   │  (Fall,     │
           │Intrusion│   │          │   │ Intrusion)  │
           └────┬────┘   └────┬─────┘   └──────┬──────┘
                │             │                 │
                └─────────────┼─────────────────┘
                              │
                        ┌─────▼─────────┐
                        │ AI CAPTION    │
                        │ GENERATOR     │
                        │(Hugging Face) │
                        └─────┬─────────┘
                              │
                ┌─────────────┼──────────────┐
                │             │              │
            ┌───▼──┐   ┌──────▼───┐   ┌─────▼──┐
            │EMAIL │   │WHATSAPP  │   │DATABASE│
            │ALERT │   │ALERT     │   │LOGGING │
            └──────┘   └──────────┘   └────────┘
                              │
                        ┌─────▼──────────┐
                        │   STREAMLIT    │
                        │   DASHBOARD    │
                        └────────────────┘
```

---

## 🏗️ System Components

| Component            | Purpose                         | Technology                      |
| -------------------- | ------------------------------- | ------------------------------- |
| **Video Processing** | Load & process video streams    | OpenCV, YOLOv8                  |
| **Person Detection** | Detect people in frames         | YOLOv8 (Ultralytics)            |
| **Tracking**         | Track people across frames      | Centroid Tracker                |
| **Pose Detection**   | Detect body posture             | MediaPipe                       |
| **Alert Detection**  | Detect events (fall, intrusion) | Custom logic                    |
| **Snapshot Capture** | Save frames for analysis        | OpenCV, PIL                     |
| **AI Captions**      | Generate descriptions           | Hugging Face LLaVA              |
| **Notifications**    | Send alerts                     | SMTP (Email), Twilio (WhatsApp) |
| **Logging**          | Store events                    | SQLite                          |
| **Dashboard**        | Visualize data                  | Streamlit                       |

---

## 📊 Data Flow

```
Video Input (mp4)
    ↓
Frame Processing
    ├→ Person Detection (boxes)
    ├→ Tracking (centroids)
    ├→ Pose Analysis (landmarks)
    └→ Counter (in/out)
    ↓
Event Detection
    ├→ Fall? → Snapshot + AI Caption
    ├→ Intrusion? → Snapshot + AI Caption
    ├→ Loitering? → Snapshot + AI Caption
    ├→ Crowd? → Snapshot + AI Caption
    └→ Inactivity? → Snapshot + AI Caption
    ↓
Alert Generation
    ├→ Generate Smart Caption (Hugging Face)
    ├→ Save to Database
    └→ Send Email + WhatsApp
    ↓
Dashboard Display
    └→ Analytics & Logs Visualization
```

---

## 🎯 Use Cases

### 1. **Retail Store Monitoring**

- ✅ Track customer flow (in/out counts)
- ✅ Detect loitering in restricted areas
- ✅ Alert on crowd formation
- ✅ Monitor store exits for unauthorized items

### 2. **Healthcare Facilities**

- ✅ Detect patient falls in hospital wards
- ✅ Monitor restricted area intrusions
- ✅ Alert on prolonged inactivity
- ✅ Track movement patterns

### 3. **Office/Workplace Safety**

- ✅ Emergency fall detection
- ✅ Unauthorized area access alerts
- ✅ Crowd safety monitoring
- ✅ Activity pattern analysis

### 4. **Public Transportation**

- ✅ Station crowd monitoring
- ✅ Platform intrusion detection
- ✅ Emergency response for falls
- ✅ Passenger flow analysis

### 5. **Industrial Sites**

- ✅ Safety equipment compliance
- ✅ Restricted zone monitoring
- ✅ Emergency incident detection
- ✅ Worker activity tracking

---

## 🚀 Key Features

✅ **Real-Time Detection**

- Processes video streams in real-time
- Instant alerts on events

✅ **AI-Powered Descriptions**

- Hugging Face LLaVA for smart captions
- Detailed, contextual alert messages
- Image attachments in emails

✅ **Multi-Alert System**

- Email notifications with snapshots
- WhatsApp alerts
- Dashboard analytics

✅ **Person Tracking**

- Unique ID tracking across frames
- Entry/exit counting
- Posture analysis

✅ **Event Logging**

- SQLite database storage
- Historical analytics
- Streamlit dashboard visualization

---

## 🔮 Future Scope

### Phase 2 - Advanced Features

- [ ] **Face Recognition**: Identify known persons/VIPs
- [ ] **Activity Recognition**: Detect running, fighting, theft attempts
- [ ] **Heat Mapping**: Visualize crowd density over time
- [ ] **Behavior Analytics**: Suspicious pattern detection
- [ ] **Multi-Camera Fusion**: Cross-camera tracking

### Phase 3 - Integration & Deployment

- [ ] **Cloud Deployment**: AWS/Azure integration
- [ ] **API Endpoints**: REST API for third-party integration
- [ ] **Mobile App**: iOS/Android push notifications
- [ ] **Mobile Alerts**: Real-time SMS alerts
- [ ] **Role-Based Access**: Admin/Supervisor/Operator dashboards

### Phase 4 - Intelligence & Learning

- [ ] **Custom Model Training**: Fine-tune for specific environments
- [ ] **Anomaly Detection**: Learn normal patterns, detect deviations
- [ ] **Predictive Alerts**: Anticipate potential incidents
- [ ] **Privacy Enhancement**: Automatic face blur/anonymization
- [ ] **Reporting**: Scheduled automated reports

### Phase 5 - Enterprise Features

- [ ] **Multi-Site Management**: Manage multiple locations
- [ ] **Compliance Reports**: GDPR/CCPA compliance
- [ ] **Audit Trails**: Complete event logging
- [ ] **Integration**: SIEM, incident management systems
- [ ] **Performance Optimization**: GPU acceleration, batch processing

---

## 📦 Tech Stack

| Layer               | Technology                            |
| ------------------- | ------------------------------------- |
| **Computer Vision** | OpenCV, YOLOv8, MediaPipe             |
| **AI/ML**           | Hugging Face LLaVA, Transformers      |
| **Backend**         | Python, FastAPI (future)              |
| **Database**        | SQLite (current), PostgreSQL (future) |
| **Frontend**        | Streamlit (current), React (future)   |
| **Notifications**   | SMTP, Twilio                          |
| **Deployment**      | Docker, Kubernetes (future)           |

---

## 📈 Performance Metrics

| Metric                          | Current    | Target       |
| ------------------------------- | ---------- | ------------ |
| **FPS**                         | ~10-15 fps | 30+ fps      |
| **Detection Latency**           | ~100ms     | <50ms        |
| **Alert Response Time**         | ~5-10s     | <2s          |
| **Accuracy (Person Detection)** | ~95%       | 98%+         |
| **Scalability**                 | 2 cameras  | 100+ cameras |

---

## 🔧 Configuration

### Alert Thresholds

```
- Loitering: > 30 seconds in same location
- Crowd: > 10 people in frame
- Inactivity: > 15 frames (1-2 seconds)
- Fall: Posture classified as "Lying"
```

### Cooldown Periods

```
- Default: 10 seconds between same alert type
- Prevents alert spam
- Configurable per alert type
```

---

## 🎓 Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
echo "HUGGINGFACE_API_KEY=your_key" > .env

# 3. Run processing
python main.py

# 4. View dashboard
streamlit run dashboard_streamlit.py
```

See [AI_CAPTIONS_SETUP.md](AI_CAPTIONS_SETUP.md) for detailed setup.

---

## 📝 File Structure

```
RealTimeSurveillanceSystem/
├── main.py                    # Main processing pipeline
├── detector.py               # Person & face detection
├── tracker.py                # Centroid tracking
├── pose_utils.py             # Pose detection
├── posture_classifier.py     # Fall detection
├── caption_generator.py      # AI caption generation ✨
├── alerts.py                 # Email/WhatsApp alerts
├── db.py                     # Database operations
├── dashboard_streamlit.py    # Analytics dashboard
├── snapshots/                # Alert snapshots
├── logs/                     # Database & CSV logs
└── data/test_videos/         # Test video files
```

---

## 💡 Future Vision

**Year 1**: Enhanced analytics, multi-site support  
**Year 2**: AI-powered behavior analysis, cloud integration  
**Year 3**: Enterprise-grade system with compliance & reporting  
**Year 5**: Industry-leading surveillance platform

---

**Last Updated**: May 2026  
**Status**: Active Development  
**Contributors**: Surveillance Team
