import os
import shutil
import uvicorn
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List

# Импортируем наш конфиг и функцию предикта
import config
from predict import analyze_track

# Настройка логов через config
os.makedirs(config.LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(config.LOG_DIR, "server.log"),
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    encoding='utf-8'
)

app = FastAPI(title="SoundUniverse ML API")


class UserHistory(BaseModel):
    user_id: int
    history_ids: List[int]
    favorite_ids: List[int]


@app.post("/analyze_genre")
async def analyze(file: UploadFile = File(...)):
    # Сохраняем временный файл
    temp = f"temp_{file.filename}"
    logging.info(f"📥 Запрос анализа файла: {file.filename}")
    try:
        with open(temp, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Вызываем функцию из predict.py
        result = analyze_track(temp)

        if "error" in result:
            logging.error(f"❌ Ошибка анализа {file.filename}: {result['error']}")
            raise HTTPException(400, result["error"])

        logging.info(f"✅ Успех: {result['verdict']} ({result['confidence']}%)")
        return {"filename": file.filename, "result": result}

    except Exception as e:
        logging.error(f"💥 Критическая ошибка: {e}")
        raise HTTPException(500, str(e))
    finally:
        # Удаляем временный файл
        if os.path.exists(temp): os.remove(temp)


@app.post("/recommend")
async def recommend(data: UserHistory):
    logging.info(f"📥 Запрос рекомендаций для User {data.user_id}")
    return {
        "user": data.user_id,
        "recommendations": [
            {"id": 99, "artist": "AI Bot", "title": "Recommendation Placeholder"}
        ]
    }


if __name__ == "__main__":
    print(f"🚀 Сервер запущен! Логи пишутся по пути {config.LOG_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8000)