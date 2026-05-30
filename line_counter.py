from datetime import datetime

class LineCounter:
    def __init__(self, line_position):
        self.line_y = line_position
        self.count_in = 0
        self.count_out = 0
        self.history = []
        self.previous_positions = {}

    def update(self, tracked_objects):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        in_this_frame = 0
        out_this_frame = 0

        for obj_id, (cx, cy) in tracked_objects.items():
            if obj_id in self.previous_positions:
                prev_y = self.previous_positions[obj_id]
                if prev_y < self.line_y and cy >= self.line_y:
                    in_this_frame += 1
                    self.count_in += 1
                elif prev_y > self.line_y and cy <= self.line_y:
                    out_this_frame += 1
                    self.count_out += 1
            self.previous_positions[obj_id] = cy

        if in_this_frame > 0 or out_this_frame > 0:
            self.history.append({
                "time": timestamp,
                "in": in_this_frame,
                "out": out_this_frame
            })
