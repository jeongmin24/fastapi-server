from fastapi import FastAPI, APIRouter
from app.services.train_service import get_recent_months, train_for_station_line

router = APIRouter()

@router.post("")
def train_endpoint(line: str, station: str):
    months = get_recent_months(6)
    train_for_station_line(months, line, station)
    return {"status": "ok", "line": line, "station": station}