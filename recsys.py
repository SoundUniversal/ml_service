import numpy as np
import random

DATABASE = [
    {"id": 1, "artist": "Nirvana", "title": "Smells Like Teen Spirit", "genre_id": 0, "tempo": 120, "mood": 0.8},
    {"id": 2, "artist": "Metallica", "title": "Master of Puppets", "genre_id": 1, "tempo": 190, "mood": 0.9},
    {"id": 3, "artist": "Lofi Girl", "title": "Study Beat", "genre_id": 31, "tempo": 80, "mood": 0.5},
    {"id": 4, "artist": "Daft Punk", "title": "Get Lucky", "genre_id": 11, "tempo": 116, "mood": 1.0},
    {"id": 5, "artist": "Skrillex", "title": "Bangarang", "genre_id": 45, "tempo": 140, "mood": 0.9},
]


class RecommendationEngine:
    def __init__(self, db):
        self.db = db

    def _get_vector(self, track):
        return np.array([
            track['genre_id'] / 5.0,
            track['tempo'] / 200.0,
            track['mood']
        ])

    def recommend(self, user_id, history_ids, favorites_ids, top_k=5):
        vectors = []
        for tid in history_ids + favorites_ids * 2:
            track = next((t for t in self.db if t['id'] == tid), None)
            if track: vectors.append(self._get_vector(track))

        if not vectors: return random.sample(self.db, k=min(len(self.db), top_k))

        user_vec = np.mean(vectors, axis=0)
        scores = []
        blacklist = set(history_ids + favorites_ids)

        for track in self.db:
            if track['id'] in blacklist: continue
            dist = np.linalg.norm(user_vec - self._get_vector(track))
            scores.append((dist, track))

        scores.sort(key=lambda x: x[0])
        recs = [s[1] for s in scores[:top_k]]

        if len(self.db) > top_k and random.random() < 0.2:
            recs[-1] = random.choice(self.db)

        return recs


if __name__ == "__main__":
    eng = RecommendationEngine(DATABASE)
    print("🤖 Демонстрация рекомендаций для фаната Металла:")
    print(eng.recommend(1, [2], []))