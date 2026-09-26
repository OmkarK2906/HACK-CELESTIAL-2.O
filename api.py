"""
Phase 5 — FastAPI wrapper.

Exposes the engine over HTTP in exactly the shapes agreed in
contracts/mock-data.json, so the frontend can swap the mock file for this API
without changing anything else.

Run:  uvicorn api:app --reload
Docs: http://127.0.0.1:8000/docs
"""

import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from engine.forecast import DATA_DIR, forecast_all
from engine.model import load_model, predict_all, train
from engine.recommend import recommend_for_zone
from engine.risk import detect_all
from engine.simulate import simulate_redistribution
from engine.hotels import build_hotels

try:
    from engine.model import predict_all as model_predict
    from engine.model import train as model_train
    ML_AVAILABLE = True
except ImportError:                       # scikit-learn not installed
    ML_AVAILABLE = False

app = FastAPI(
    title="EventPulse API",
    description="Intelligent capacity and crowd management for mega-events.",
    version="0.1.0",
)

# The React dev server runs on a different port, so allow it through.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default "now" within the simulated 24-hour window.
DEFAULT_HOUR = 110   # the Final — peak pressure, the moment worth demoing


def _require_data():
    zones_path = os.path.join(DATA_DIR, "zones.csv")
    if not os.path.exists(zones_path):
        raise HTTPException(
            status_code=503,
            detail="No data generated yet. Run: python generate_city.py",
        )


@app.get("/api")
def api_index():
    return {
        "service": "EventPulse API",
        "endpoints": ["/zones", "/forecast", "/alerts", "/recommendations/{zone_id}",
                      "/summary", "/transport", "/history", "/hotels", "/events",
                      "/simulate/{zone_id}",
                      "/ml/forecast", "/ml/benchmark"],
        "ml_available": ML_AVAILABLE,
    }


@app.get("/zones")
def get_zones(hour: int = DEFAULT_HOUR):
    """All zones with their current occupancy at the given hour."""
    _require_data()

    zones = pd.read_csv(os.path.join(DATA_DIR, "zones.csv"))
    occupancy = pd.read_csv(os.path.join(DATA_DIR, "occupancy_timeline.csv"))

    snapshot = occupancy[occupancy["hour_offset"] == hour]
    current = dict(zip(snapshot["zone_id"], snapshot["occupancy_pct"]))
    occupied = dict(zip(snapshot["zone_id"], snapshot["occupancy"]))

    out = []
    for _, z in zones.iterrows():
        pct = float(current.get(z["zone_id"], 0.0))
        out.append({
            "id": z["zone_id"],
            "name": z["name"],
            "lat": float(z["lat"]),
            "lng": float(z["lng"]),
            "hotel_capacity": int(z["hotel_capacity"]),
            "current_occupancy": int(occupied.get(z["zone_id"], 0)),
            "occupancy_pct": round(pct, 2),
            "spare_capacity": max(0, int(z["hotel_capacity"] * (100 - pct) / 100)),
            "avg_nightly_rate": int(z["avg_nightly_rate"]),
            "is_venue_zone": bool(z["is_venue_zone"]),
            "distance_to_venue_km": float(z["distance_to_venue_km"]),
        })

    return out


@app.get("/forecast")
def get_forecast(hour: int = DEFAULT_HOUR, horizon: int = 12):
    """Projected occupancy and time-to-saturation for every zone."""
    _require_data()
    occupancy = pd.read_csv(os.path.join(DATA_DIR, "occupancy_timeline.csv"))
    return forecast_all(occupancy, current_hour=hour, horizon=horizon)


@app.get("/alerts")
def get_alerts(hour: int = DEFAULT_HOUR):
    """Active alerts, most urgent first."""
    _require_data()
    return detect_all(current_hour=hour)


@app.get("/recommendations/{zone_id}")
def get_recommendations(zone_id: str, hour: int = DEFAULT_HOUR):
    """Ranked alternative zones for a zone under pressure."""
    _require_data()
    return recommend_for_zone(zone_id, current_hour=hour)


@app.get("/summary")
def get_summary(hour: int = DEFAULT_HOUR):
    """Headline numbers for the dashboard's top bar."""
    _require_data()

    zones = get_zones(hour)
    alerts = detect_all(current_hour=hour)

    total_capacity = sum(z["hotel_capacity"] for z in zones)
    total_occupied = sum(z["current_occupancy"] for z in zones)
    saturated = [z for z in zones if z["occupancy_pct"] >= 95]

    return {
        "hour": hour,
        "total_capacity": total_capacity,
        "total_occupied": total_occupied,
        "citywide_occupancy_pct": round(total_occupied / total_capacity * 100, 1),
        "spare_capacity": max(0, total_capacity - total_occupied),
        "zones_saturated": len(saturated),
        "zones_total": len(zones),
        "alerts_critical": len([a for a in alerts if a["severity"] == "critical"]),
        "alerts_warning": len([a for a in alerts if a["severity"] == "warning"]),
    }


@app.get("/simulate/{zone_id}")
def simulate(zone_id: str, move: int, hour: int = DEFAULT_HOUR,
             compliance: float = 0.75):
    """
    What-if: move `move` visitors out of a zone and see where they land.

    Returns the before/after occupancy for the source and every receiving zone,
    plus whether the alert clears.
    """
    _require_data()
    result = simulate_redistribution(
        zone_id, move, current_hour=hour, compliance=compliance
    )
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/events")
def events(hour: int = DEFAULT_HOUR):
    """
    The event schedule around the hour being viewed.

    Occupancy rises because something is coming; without the schedule on screen
    the dashboard shows pressure with no explanation of why.
    """
    _require_data()

    ev = pd.read_csv(os.path.join(DATA_DIR, "events.csv"))
    zones = pd.read_csv(os.path.join(DATA_DIR, "zones.csv"))
    names = dict(zip(zones["zone_id"], zones["name"]))

    out = []
    for _, e in ev.iterrows():
        start, end = int(e["start_hour"]), int(e["end_hour"])

        if hour < start:
            state, hours_away = "upcoming", start - hour
        elif hour <= end:
            state, hours_away = "running", 0
        else:
            state, hours_away = "finished", hour - end

        out.append({
            "event_id": e["event_id"],
            "name": e["name"],
            "zone_id": e["zone_id"],
            "zone_name": names.get(e["zone_id"], e["zone_id"]),
            "start_hour": start,
            "end_hour": end,
            "expected_attendance": int(e["expected_attendance"]),
            "state": state,
            "hours_away": int(hours_away),
        })

    # Anything still to come, plus whatever is running now, soonest first.
    live = [x for x in out if x["state"] == "running"]
    ahead = sorted([x for x in out if x["state"] == "upcoming"],
                   key=lambda x: x["hours_away"])

    return {"hour": hour, "schedule": live + ahead, "all": out}


@app.get("/hotels")
def hotels(hour: int = DEFAULT_HOUR):
    """
    Bookable inventory across every zone, with live availability.

    The attendee-facing view of the same data the command view works from, so
    both sides agree on what is actually free.
    """
    _require_data()
    return build_hotels(current_hour=hour)


@app.get("/history")
def get_history(hour: int = DEFAULT_HOUR, window: int = 24, top: int = 5):
    """
    Occupancy trajectory for the busiest zones over a trailing window.

    Feeds the trend chart: enough history to show where a zone has come from,
    ending at the hour being viewed so the line grows as the clock advances.
    """
    _require_data()

    occ = pd.read_csv(os.path.join(DATA_DIR, "occupancy_timeline.csv"))
    zones = pd.read_csv(os.path.join(DATA_DIR, "zones.csv"))
    names = dict(zip(zones["zone_id"], zones["name"]))

    lo = max(0, hour - window)
    span = occ[(occ["hour_offset"] >= lo) & (occ["hour_offset"] <= hour)]

    # Rank by pressure at the current hour, not across the whole window.
    now = span[span["hour_offset"] == hour].sort_values("occupancy_pct", ascending=False)
    keep = list(now["zone_id"].head(top))

    series = []
    for zid in keep:
        rows = span[span["zone_id"] == zid].sort_values("hour_offset")
        series.append({
            "zone_id": zid,
            "zone_name": names.get(zid, zid),
            "points": [
                {"hour": int(r["hour_offset"]), "pct": float(r["occupancy_pct"])}
                for _, r in rows.iterrows()
            ],
        })

    return {"from_hour": lo, "to_hour": hour, "series": series}


@app.get("/transport")
def get_transport(hour: int = DEFAULT_HOUR):
    """Transport links with their load at the given hour, for the map view."""
    _require_data()

    t = pd.read_csv(os.path.join(DATA_DIR, "transport.csv"))

    load_path = os.path.join(DATA_DIR, "transport_load.csv")
    if os.path.exists(load_path):
        load = pd.read_csv(load_path)
        snap = load[load["hour_offset"] == hour]
        t = t.merge(
            snap[["link_id", "current_load", "load_pct"]],
            on="link_id", how="left",
        )
        t["load_pct"] = t["load_pct"].fillna(0.0)
        t["current_load"] = t["current_load"].fillna(0).astype(int)

    return t.to_dict("records")


@app.get("/ml/forecast")
def ml_forecast(hour: int = None):
    """Gradient boosting forecast, 6 hours ahead."""
    if not ML_AVAILABLE:
        raise HTTPException(503, "scikit-learn not installed")

    _require_data()
    preds = model_predict(current_hour=hour)
    if preds is None:
        raise HTTPException(
            503, "Model not trained yet. Run: python -m engine.model"
        )
    return preds


@app.get("/ml/benchmark")
def ml_benchmark():
    """
    Model scorecard against the trend baseline on held-out data.

    Retrains on request. At this data size that takes under a second, and it
    means the figures shown are always the real ones rather than something
    cached from an earlier dataset.
    """
    if not ML_AVAILABLE:
        raise HTTPException(503, "scikit-learn not installed")

    _require_data()
    _model, _df, m = model_train(verbose=False)

    return {
        "horizon_hours": m["horizon_hours"],
        "train_rows": m["train_rows"],
        "test_rows": m["test_rows"],
        "split_at_hour": m["split_at_hour"],
        "baseline": {
            "name": "Trend projection",
            "mae": round(m["baseline_mae"], 2),
            "r2": round(m["baseline_r2"], 3),
        },
        "model": {
            "name": "Gradient boosting",
            "mae": round(m["model_mae"], 2),
            "r2": round(m["model_r2"], 3),
        },
        "mae_improvement_pct": m["improvement_pct"],
    }


# --------------------------------------------------------------------------
# Serve the dashboard
# --------------------------------------------------------------------------
# In deployment there is one service, not two: the API also serves the page, so
# the frontend can call it same-origin and there is a single URL to share.
# Mounted last so it never shadows an API route.
FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend"
)

if os.path.isdir(FRONTEND_DIR):
    @app.get("/")
    def dashboard():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
