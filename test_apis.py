import urllib.request
import urllib.parse
import json

ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjgzYTA3OTkwYWI3MjRmMjRhOGM4OTBlNDNiZDg5ZDIyIiwiaCI6Im11cm11cjY0In0="
OPENWEATHER_API_KEY = "cd71c2529b62ae38493a7c94832231a2"
OPENCHARGE_API_KEY = "14402eda-48ca-4832-b2e4-fce9aa6e40b8"

def test_ors():
    print("Testing OpenRouteService...")
    url = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
    data = json.dumps({"coordinates": [[-122.4194, 37.7749], [-122.4194, 37.8049]]}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={
        "Authorization": ORS_API_KEY,
        "Content-Type": "application/json"
    }, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            if response.status == 200:
                print("✅ ORS API: SUCCESS")
            else:
                print(f"❌ ORS API: FAILED with status {response.status}")
    except Exception as e:
        print(f"❌ ORS API: FAILED with exception {e}")

def test_elevation():
    print("Testing Open-Elevation...")
    url = "https://api.open-elevation.com/api/v1/lookup"
    data = json.dumps({"locations": [{"latitude": 37.7749, "longitude": -122.4194}]}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json",
        "Accept": "application/json"
    }, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            if response.status == 200 and 'results' in res_data:
                print("✅ Open-Elevation API: SUCCESS")
            else:
                print(f"❌ Open-Elevation API: FAILED with status {response.status}")
    except Exception as e:
        print(f"❌ Open-Elevation API: FAILED with exception {e}")

def test_weather():
    print("Testing OpenWeatherMap...")
    params = urllib.parse.urlencode({
        "lat": 37.7749, "lon": -122.4194, "units": "metric", "appid": OPENWEATHER_API_KEY
    })
    url = f"https://api.openweathermap.org/data/2.5/weather?{params}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                print("✅ OpenWeatherMap API: SUCCESS")
            else:
                print(f"❌ OpenWeatherMap API: FAILED with status {response.status}")
    except Exception as e:
        if hasattr(e, 'read'):
            print(f"❌ OpenWeatherMap API: FAILED with exception {e}, body: {e.read().decode('utf-8')}")
        else:
            print(f"❌ OpenWeatherMap API: FAILED with exception {e}")

def test_ocm():
    print("Testing Open Charge Map...")
    params = urllib.parse.urlencode({
        "latitude": 37.7749, "longitude": -122.4194, "distance": 10,
        "distanceunit": "KM", "maxresults": 5, "compact": "true",
        "verbose": "false", "output": "json"
    })
    url = f"https://api.openchargemap.io/v3/poi?{params}"
    req = urllib.request.Request(url, headers={
        "X-API-Key": OPENCHARGE_API_KEY,
        "User-Agent": "IES_EV_System/0.1.0"
    }, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            if response.status == 200:
                print("✅ Open Charge Map API: SUCCESS")
            else:
                print(f"❌ Open Charge Map API: FAILED with status {response.status}")
    except Exception as e:
        if hasattr(e, 'read'):
            print(f"❌ Open Charge Map API: FAILED with exception {e}, body: {e.read().decode('utf-8')}")
        else:
            print(f"❌ Open Charge Map API: FAILED with exception {e}")

if __name__ == "__main__":
    test_ors()
    test_elevation()
    test_weather()
    test_ocm()
