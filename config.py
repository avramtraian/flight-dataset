from dataclasses import dataclass
from enum import Enum

import json

class Weather(Enum):
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAIN = "rain"
    FOG = "fog"
    SNOW = "snow"
    RAINSTORM = "rainstorm"
    THUNDERSTORM = "thunderstorm"
    INVALID = "<invalid>"

class Month(Enum):
    JANUARY = "January"
    FEBRUARY = "February"
    MARCH = "March"
    APRIL = "April"
    MAY = "May"
    JUNE = "June"
    JULY = "July"
    AUGUST = "August"
    SEPTEMBER = "September"
    OCTOBER = "October"
    NOVEMBER = "November"
    DECEMBER = "December"

@dataclass
class AirportWeather:
    probabilities: dict[Weather, float]

@dataclass
class AirportConfig:
    code: str
    name: str
    coord_x: float
    coord_y: float
    max_daily_flights: int
    congestion_factors: dict[Month, float]
    holiday_congestion_multiplier: float
    weather: dict[Month, AirportWeather]

def load_airport_congestion_factors(json_data) -> dict[Month, float]:
    raw_congestion_factors = list(json_data)
    congestion_factors: dict[Month, float] = {}

    for (month_index, month) in enumerate(Month):
        congestion_factors[month] = float(raw_congestion_factors[month_index])
    
    return congestion_factors

def load_airport_weather(json_data) -> dict[Month, AirportWeather]:
    raw_weather = list(json_data)
    weather: dict[Month, AirportWeather] = {}

    for (month_index, month) in enumerate(Month):
        raw_month_weather = dict(raw_weather[month_index])
        month_weather = AirportWeather(
            probabilities = dict()
        )

        for condition in Weather:
            if condition == Weather.INVALID:
                continue
            month_weather.probabilities[condition] = float(raw_month_weather[condition.value])
        weather[month] = month_weather
    
    return weather

def load_airport_config(json_data) -> AirportConfig:
    congestion_factors = load_airport_congestion_factors(json_data["congestion_factors"])
    weather = load_airport_weather(json_data["weather"])

    return AirportConfig(
        code = str(json_data["code"]),
        name = str(json_data["name"]),
        coord_x = float(json_data["coord_x"]),
        coord_y = float(json_data["coord_y"]),
        max_daily_flights = int(json_data["max_daily_flights"]),
        congestion_factors = congestion_factors,
        holiday_congestion_multiplier = float(json_data["holiday_congestion_multiplier"]),
        weather = weather
    )

def load_airports(file_path) -> dict[str, AirportConfig]:
    # Load the JSON file from disk.
    with open(file_path, "r") as file:
        json_data = json.load(file)

    # Parse each airport in the config file.
    airports: dict[str, AirportConfig] = {}
    for airport_json_data in json_data:
        airport_config = load_airport_config(airport_json_data)
        airports[airport_config.code] = airport_config

    return airports

@dataclass
class AirlineConfig:
    code: str
    name: str
    market_share: float
    ticket_cost: int
    base_delay_probability: float
    holiday_delay_multiplier: float

def load_airlines(file_path) -> dict[str, AirlineConfig]:
    # Load the JSON file from disk.
    with open(file_path, "r") as file:
        json_data = json.load(file)

    # Parse each airline in the config file.
    airlines: dict[str, AirlineConfig] = {}
    for airline_json_data in json_data:
        code = str(airline_json_data["code"])
        airlines[code] = AirlineConfig(
            code = code,
            name = str(airline_json_data["name"]),
            market_share = float(airline_json_data["market_share"]),
            ticket_cost = int(airline_json_data["ticket_cost"]),
            base_delay_probability = float(airline_json_data["base_delay_probability"]),
            holiday_delay_multiplier = float(airline_json_data["holiday_delay_multiplier"]),
        )

    return airlines
