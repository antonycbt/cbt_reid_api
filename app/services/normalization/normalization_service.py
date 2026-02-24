from datetime import datetime
from collections import defaultdict
import json

class NormalizationService:
    def __init__(self, raw_repo, normalized_repo):
        self.raw_repo = raw_repo
        self.normalized_repo = normalized_repo

    def pick_next_table(self):
        return self.raw_repo.get_next_unprocessed_table()

    def normalize_table(self, table_name: str):
        print(f"[NORMALIZER] processing {table_name}")

        rows = self.raw_repo.stream_table(table_name)

        try:
            normalized_rows = self._transform(rows)
        except Exception as e:
            import traceback
            print("[NORMALIZER] _transform crashed:", e)
            print(traceback.format_exc())
            return

        if not normalized_rows:
            print("[NORMALIZER] No data to normalize — skipping mark")
            return

        success = self.normalized_repo.bulk_insert(normalized_rows)

        if success:
            self.raw_repo.mark_normalized(table_name)
            print("[NORMALIZER] Marked as normalized ✅")
        else:
            print("[NORMALIZER] Insert failed — NOT marking ❌")

    def _transform(self, rows):
        rows = list(rows)
        print(f"[TRANSFORM] incoming rows count: {len(rows)}", flush=True)

        if not rows:
            print("-----------------------------------------ROWS is empty", flush=True)
            return []

        # debug: show a few sample rows (don't dump everything if many rows)
        for i, r in enumerate(rows[:5]):
            try:
                # try to show attribute dict; adjust if row is tuple/dict/model
                print(f"[TRANSFORM] row[{i}]: type={type(r)} repr=", getattr(r, '__dict__', repr(r)), flush=True)
            except Exception as ex:
                print("[TRANSFORM] failed to print row preview:", ex, flush=True)

        grouped = defaultdict(list)
        for r in rows:
            # adapt depending on row shape (namedtuple / SQLAlchemy row proxy / dict)
            # try attribute access, then dict access, then index 0 fallback
            guest_temp_id = None
            member_id = None
            camera_id = None
            guest_data_vector = None
            match_value = None
            data_ts = None

            try:
                # try attribute style (ORM object)
                guest_temp_id = getattr(r, "guest_temp_id", None)
                member_id = getattr(r, "member_id", None)
                camera_id = getattr(r, "camera_id", None)
                guest_data_vector = getattr(r, "guest_data_vector", None)
                match_value = getattr(r, "match_value", None)
                data_ts = getattr(r, "data_ts", None)
            except Exception:
                pass

            # fallback if row is dict-like
            if isinstance(r, dict):
                guest_temp_id = r.get("guest_temp_id", guest_temp_id)
                member_id = r.get("member_id", member_id)
                camera_id = r.get("camera_id", camera_id)
                guest_data_vector = r.get("guest_data_vector", guest_data_vector)
                match_value = r.get("match_value", match_value)
                data_ts = r.get("data_ts", data_ts)

            # final fallback for sequences/tuples: don't rely on this unless you know schema
            # group by guest_temp_id if present else member_id else camera+ts window
            key = guest_temp_id if guest_temp_id else f"member_{member_id}"
            grouped[key].append({
                "member_id": member_id,
                "guest_temp_id": guest_temp_id,
                "camera_id": camera_id,
                "guest_data_vector": guest_data_vector,
                "match_value": match_value,
                "data_ts": data_ts,
            })

        results = []
        for key, group in grouped.items():
            # collect vectors that are proper lists of numbers
            vectors = []
            match_vals = []
            timestamps = []

            for item in group:
                vec = item["guest_data_vector"]
                if vec is None:
                    pass
                else:
                    # If JSON string, parse it. If dict with key 'vector' adapt as needed.
                    if isinstance(vec, str):
                        try:
                            vec = json.loads(vec)
                        except Exception as e:
                            print("[TRANSFORM] failed to json.loads vector:", e, "vec repr:", vec, flush=True)
                            vec = None

                    # if it's dict maybe vector is under some key; print for diagnosis
                    if isinstance(vec, dict):
                        # print("[TRANSFORM] vector is dict; item:", vec, flush=True)
                        # try common keys:
                        if "vector" in vec:
                            vec = vec["vector"]
                        else:
                            vec = None

                    if isinstance(vec, list):
                        # ensure elements are numeric
                        try:
                            vec = [float(x) for x in vec]
                            vectors.append(vec)
                        except Exception as e:
                            print("[TRANSFORM] vector contains non-numeric item:", e, "vec:", vec, flush=True)
                    # else:
                    #     if vec is not None:
                    #         print("[TRANSFORM] unsupported vector type:", type(vec), "value:", vec, flush=True)

                if item["match_value"] is not None:
                    try:
                        match_vals.append(float(item["match_value"]))
                    except Exception:
                        print("[TRANSFORM] bad match_value:", item["match_value"], flush=True)

                if item["data_ts"] is not None:
                    timestamps.append(item["data_ts"])

            # compute averages
            avg_vector = None
            if vectors:
                L = len(vectors[0])
                # ensure all vectors same length
                if not all(len(v) == L for v in vectors):
                    print("[TRANSFORM] vector lengths differ within group; skipping average for group", key, flush=True)
                    avg_vector = None
                else:
                    avg_vector = []
                    for i in range(L):
                        col_sum = sum(v[i] for v in vectors)
                        avg_vector.append(col_sum / len(vectors))

            avg_match = (sum(match_vals) / len(match_vals)) if match_vals else None
            movement_ts = max(timestamps) if timestamps else datetime.utcnow()

            example = group[0]
            results.append({
                "member_id": example["member_id"],
                "guest_temp_id": example["guest_temp_id"],
                "average_guest_data_vector": avg_vector,
                "camera_id": example["camera_id"],
                "movement_type": 1,
                "movement_ts": movement_ts,
                "average_match_value": avg_match,
            })

        print(f"[TRANSFORM] produced {len(results)} normalized rows", flush=True)
        return results
