import requests
import csv
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

def get_last_datetime(file_path):
    """Lấy thời điểm cuối cùng trong file CSV"""
    if not Path(file_path).exists():
        return None

    df = pd.read_csv(file_path)
    if df.empty:
        return None

    last_time = pd.to_datetime(df["datetime"]).max()
    return last_time


def crawl_weather_incremental(
    output_dir="data/raw/",
    start_date="2025-01-01"
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    end_date = now.strftime("%Y-%m-%d")

    cities = [
        {"name": "hanoi", "lat": 21.03, "lon": 105.85},
        {"name": "danang", "lat": 16.05, "lon": 108.20},
        {"name": "hcm", "lat": 10.82, "lon": 106.63},
    ]

    url = "https://archive-api.open-meteo.com/v1/archive"

    features = ",".join([
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "windspeed_10m",
        "surface_pressure"
    ])

    for city in cities:
        print(f"\n=== {city['name']} ===")

        file_path = f"{output_dir}{city['name']}_hourly.csv"

        last_time = get_last_datetime(file_path)

        if last_time:
            start_time = last_time + timedelta(hours=1)
            print(f"🔁 Cập nhật từ: {start_time}")
        else:
            start_time = datetime.strptime(start_date, "%Y-%m-%d")
            print(f"🆕 Crawl từ đầu: {start_time}")
        if start_time >= now:
            print("⏭ Không có dữ liệu mới")
            continue

        params = {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "start_date": start_time.strftime("%Y-%m-%d"),
            "end_date": end_date,
            "hourly": features,
            "timezone": "Asia/Ho_Chi_Minh"
        }

        try:
            res = requests.get(url, params=params, timeout=30)

            if res.status_code != 200:
                print("Lỗi:", res.status_code)
                continue

            data = res.json()["hourly"]

            times = data["time"]
            file_exists = Path(file_path).exists()

            with open(file_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "datetime",
                        "temperature",
                        "humidity",
                        "rain",
                        "wind",
                        "pressure"
                    ])

                for i in range(len(times)):
                    writer.writerow([
                        times[i],
                        data["temperature_2m"][i],
                        data["relative_humidity_2m"][i],
                        data["precipitation"][i],
                        data["windspeed_10m"][i],
                        data["surface_pressure"][i]
                    ])

            print(f"Đã cập nhật {len(times)} dòng")

        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    crawl_weather_incremental(start_date="2025-01-01")