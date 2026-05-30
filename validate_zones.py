import json

def validate_zone_json(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("Root element should be a dictionary (camera_id: list of zones)")

        for camera, zones in data.items():
            if not isinstance(zones, list):
                raise ValueError(f"Zones under '{camera}' should be a list")
            for zone in zones:
                if "id" not in zone or "type" not in zone or "points" not in zone:
                    raise ValueError(f"Zone in '{camera}' missing 'id', 'type', or 'points'")
                if not isinstance(zone["points"], list) or len(zone["points"]) < 3:
                    raise ValueError(f"Zone '{zone['id']}' in '{camera}' must have at least 3 points")

        print("✅ Zone config is valid!")
    except Exception as e:
        print(f"❌ Validation failed: {e}")

if __name__ == "__main__":
    path = input("Enter path to zone_config.json: ")
    validate_zone_json(path)
