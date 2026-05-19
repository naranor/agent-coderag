import requests


# This is a class docstring for WeatherClient.
# It should be extracted by the parser.
class WeatherClient:
    """A client to fetch weather data."""

    # Retrieves temperature for a city.
    # Uses external weather API.
    def get_temperature(self, city: str) -> float:
        # Business logic to get weather
        response = requests.get(f"https://api.weather.com/{city}")
        return response.json()["temp"]


# Global function outside any class
# Performs weather retrieval
def get_temperature(city: str) -> float:
    return WeatherClient().get_temperature(city)
