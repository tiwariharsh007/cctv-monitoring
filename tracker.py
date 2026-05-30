import numpy as np
from collections import OrderedDict


class CentroidTracker:
    def __init__(self, max_disappeared=10, max_history=30):
        self.next_id = 0

        self.objects = OrderedDict()        # object_id -> centroid
        self.disappeared = OrderedDict()    # object_id -> disappeared count

        self.object_history = {}            # object_id -> list of centroids
        self.max_history = max_history

        self.trail_map = []                 # global trail (optional visualization)
        self.max_trail = 20000

        self.max_disappeared = max_disappeared

    # ---------------- REGISTER ---------------- #
    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.object_history[self.next_id] = [centroid]
        self.next_id += 1

    # ---------------- DEREGISTER ---------------- #
    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
        if object_id in self.object_history:
            del self.object_history[object_id]

    # ---------------- UPDATE ---------------- #
    def update(self, rects):
        # No detections
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1

                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

            return self.objects

        # Compute centroids
        input_centroids = np.array([
            (int((x1 + x2) / 2), int((y1 + y2) / 2))
            for (x1, y1, x2, y2) in rects
        ])

        # No existing objects → register all
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)

        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute distance matrix
            D = np.linalg.norm(
                np.array(object_centroids)[:, None] - input_centroids[None, :],
                axis=2
            )

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            # Match existing objects
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Unmatched rows → disappeared
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Unmatched cols → new objects
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col])

        # ---------------- HISTORY UPDATE (FIXED) ---------------- #
        for object_id, centroid in self.objects.items():
            if object_id not in self.object_history:
                self.object_history[object_id] = []

            self.object_history[object_id].append(centroid)

            # keep only last N points
            self.object_history[object_id] = self.object_history[object_id][-self.max_history:]

        # ---------------- GLOBAL TRAIL ---------------- #
        for c in input_centroids:
            self.trail_map.append(c)
            if len(self.trail_map) > self.max_trail:
                self.trail_map.pop(0)

        return self.objects