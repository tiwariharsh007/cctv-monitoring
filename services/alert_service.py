from datetime import datetime


def handle_alert(alert_type: str, details: str, image_path: str = None) -> dict:
    """Log alert to console and return structured data. Email is handled by caller."""
    severity_map = {
        "fall": "HIGH", "intrusion": "HIGH", "capacity": "HIGH",
        "abandoned": "HIGH", "after_hours": "HIGH", "accident": "HIGH",
        "crowd": "MEDIUM", "loitering": "MEDIUM",
        "running": "MEDIUM", "inactivity": "MEDIUM", "tailgating": "MEDIUM",
    }
    severity  = severity_map.get(alert_type.lower(), "LOW")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{severity}] {alert_type.upper()} → {details}")
    return {"time": timestamp, "type": alert_type, "severity": severity, "details": details}
