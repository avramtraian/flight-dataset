from dataclasses import dataclass
import random
import config
import numpy as np
import pandas as pd

@dataclass
class Flight:
    flight_id: int
    departure_airport: str
    arrival_airport: str
    airline: str
    month: config.Month
    is_holiday: bool
    departure_weather: config.Weather
    arrival_weather: config.Weather
    departure_congestion: float
    arrival_congestion: float
    ticket_price: int
    is_delayed: bool

@dataclass
class OutlierStats:
    probability: float
    min_multiplier: float
    max_multiplier: float

class Context:
    airports: dict[str, config.AirportConfig]
    airlines: dict[str, config.AirlineConfig]
    is_holiday_probability: float
    weather_factor_map: dict[config.Weather, float]
    missing_probabilities: dict[str, float]
    outliers: dict[str, OutlierStats]

def init_context() -> Context:
    context = Context()
    context.airports = config.load_airports("airports.json")
    context.airlines = config.load_airlines("airlines.json")
    context.is_holiday_probability = 0.05
    context.weather_factor_map = {
        config.Weather.CLEAR: 0.0,
        config.Weather.CLOUDY: 0.1,
        config.Weather.RAIN: 0.2,
        config.Weather.FOG: 0.4,
        config.Weather.SNOW: 0.6,
        config.Weather.RAINSTORM: 0.8,
        config.Weather.THUNDERSTORM: 1.0,
    }
    context.missing_probabilities = {
        "airline": 0.01,
        "departure_weather": 0.04,
        "arrival_weather": 0.04,
        "departure_congestion": 0.02,
        "arrival_congestion": 0.03,
        "ticket_price": 0.05,
    }
    context.outliers = {
        "departure_congestion": OutlierStats(
            probability = 0.01,
            min_multiplier = 1.04,
            max_multiplier = 1.05,
        ),
        "arrival_congestion": OutlierStats(
            probability = 0.01,
            min_multiplier = 1.04,
            max_multiplier = 1.05,
        ),
        "ticket_price": OutlierStats(
            probability = 0.04,
            min_multiplier = 1.04,
            max_multiplier = 1.05,
        ),
        "is_delayed": OutlierStats(
            probability = 0.005,
            min_multiplier = 0.0,
            max_multiplier = 0.0,
        )
    }
    return context

def generate_weather_for_airport(
    context: Context,
    airport_code: str,
    month: config.Month,
    is_holiday: bool,
) -> config.Weather:
    airport = context.airports[airport_code]
    airport_weather = airport.weather[month]

    conditions = list(airport_weather.probabilities.keys())
    condition_weights = list(airport_weather.probabilities.values())
    condition = random.choices(conditions, weights=condition_weights, k=1)[0]
    return condition

def generate_congestion_factor_for_airport(
    context: Context,
    airport_code: str,
    month: config.Month,
    is_holiday: bool,
) -> float:
    airport = context.airports[airport_code]
    base_factor = airport.congestion_factors[month]
    holiday_multiplier = 1.0
    if is_holiday:
        holiday_multiplier = airport.holiday_congestion_multiplier
    return base_factor * holiday_multiplier

def get_flight_distance(context: Context, departure_code: str, arrival_code: str) -> float:
    departure = context.airports[departure_code]
    arrival = context.airports[arrival_code]
    
    # VERY simplified formula for calculating the distance from two points expressed in
    # longitude/latitude coordinates.
    dx = (arrival.coord_x - departure.coord_x) * 111
    dy = (arrival.coord_y - departure.coord_y) * 111
    return np.sqrt(dx**2 + dy**2)

# The implementation of the `P_delay(W_D, W_A, K_D, K_A, Q, BP)` mentioned in the documentation.
# Arguably, the most important and interesting function in the entire project!
def calculate_delayed_probability(
    context: Context,
    departure_weather: config.Weather,
    arrival_weather: config.Weather,
    departure_congestion_factor: float,
    arrival_congestion_factor: float,
    ticket_price: int,
    airline_base_delay_probability: float
) -> float:
    # Mathematical factors.
    W_D = context.weather_factor_map[departure_weather]
    W_A = context.weather_factor_map[arrival_weather]
    K_D = departure_congestion_factor
    K_A = arrival_congestion_factor
    Q   = float(ticket_price)
    BP  = airline_base_delay_probability

    # Linear coefficients.
    A_1 = 0.25
    A_2 = 0.15
    A_3 = 0.25
    A_4 = 0.15
    A_5 = 0.0
    A_6 = 1.0

    probability = (
        (A_1 * W_D) +
        (A_2 * W_A) +
        (A_3 * K_D) +
        (A_4 * K_A) +
        (A_5 * Q  ) +
        (A_6 * BP )
    )
    return min(1.0, max(0.0, probability))

def generate_flight(context: Context) -> Flight:
    # Randomly generate the departure and arrival airports.
    departure = random.choice(list(context.airports.keys()))
    arrival = random.choice(list(context.airports.keys()))
    if len(context.airports) >= 2:
        while departure == arrival:
            arrival = random.choice(list(context.airports.keys()))

    # Based on the market share, generate an airline.
    airline_codes = list(context.airlines.keys())
    airline_weights = list()
    for airline in context.airlines.values():
        airline_weights.append(airline.market_share)
    airline_code = random.choices(airline_codes, weights=airline_weights, k=1)[0]

    # Randomly generate the month when the flight is programmed.
    month = random.choice(list(config.Month))

    # Generate whether the flight is programmed on a holiday.
    is_holiday = random.random() < context.is_holiday_probability

    departure_weather = generate_weather_for_airport(context, departure, month, is_holiday)
    departure_congestion = generate_congestion_factor_for_airport(context, departure, month, is_holiday)
    arrival_weather = generate_weather_for_airport(context, arrival, month, is_holiday)
    arrival_congestion = generate_congestion_factor_for_airport(context, arrival, month, is_holiday)

    flight_distance = get_flight_distance(context, departure, arrival)
    airline_ticket_cost = context.airlines[airline_code].ticket_cost
    airline_base_delay_probability = context.airlines[airline_code].base_delay_probability
    ticket_price = int((flight_distance / 1000.0) * airline_ticket_cost)

    # Determine if the flight is delayed.
    delayed_probability = calculate_delayed_probability(
        context,
        departure_weather,
        arrival_weather,
        departure_congestion,
        arrival_congestion,
        ticket_price,
        airline_base_delay_probability
    )
    is_delayed = random.random() < delayed_probability

    return Flight(
        flight_id = 0,
        departure_airport = departure,
        arrival_airport = arrival,
        airline = airline_code,
        month = month,
        is_holiday = is_holiday,
        departure_weather = departure_weather,
        arrival_weather = arrival_weather,
        departure_congestion = departure_congestion,
        arrival_congestion = arrival_congestion,
        ticket_price = ticket_price,
        is_delayed = is_delayed
    )

def generate_flights(context: Context, number_of_flights: int) -> list[Flight]:
    flights = list()
    for i in range(number_of_flights):
        flights.append(generate_flight(context))
    return flights

INVALID_AIRPORT = "???"
INVALID_AIRLINE = "???"
INVALID_CONGESTION = -1.0
INVALID_TICKET_PRICE = 0

def insert_missing_characteristics(context: Context, flights: list[Flight]):
    probabilities = context.missing_probabilities
    for flight in flights:
        if random.random() < probabilities["airline"]:
            flight.airline = INVALID_AIRLINE
        if random.random() < probabilities["departure_weather"]:
            flight.departure_weather = config.Weather.INVALID
        if random.random() < probabilities["arrival_weather"]:
            flight.arrival_weather = config.Weather.INVALID
        if random.random() < probabilities["departure_congestion"]:
            flight.departure_congestion = INVALID_CONGESTION
        if random.random() < probabilities["arrival_congestion"]:
            flight.arrival_congestion = INVALID_CONGESTION
        if random.random() < probabilities["ticket_price"]:
            flight.ticket_price = INVALID_TICKET_PRICE

def insert_outliers(context: Context, flights: list[Flight]):
    outliers = context.outliers
    departure_congestion = outliers["departure_congestion"]
    arrival_congestion = outliers["arrival_congestion"]
    ticket_price = outliers["ticket_price"]
    is_delayed = outliers["is_delayed"]

    for flight in flights:
        # Departure congestion
        if random.random() < departure_congestion.probability:
            multiplier = random.uniform(departure_congestion.min_multiplier, departure_congestion.max_multiplier)
            flight.departure_congestion = flight.departure_congestion * multiplier

        # Arrival congestion
        if random.random() < arrival_congestion.probability:
            multiplier = random.uniform(arrival_congestion.min_multiplier, arrival_congestion.max_multiplier)
            flight.arrival_congestion = flight.arrival_congestion * multiplier

        # Ticket price
        if random.random() < ticket_price.probability:
            multiplier = random.uniform(ticket_price.min_multiplier, ticket_price.max_multiplier)
            flight.ticket_price = flight.ticket_price * multiplier

        # Is delayed
        if random.random() < is_delayed.probability:
            flight.is_delayed = not flight.is_delayed

def write_flights_to_csv(file_path: str, flights: list[Flight]):
    data_frame_rows = list()
    for flight in flights:
        departure_weather = None
        if flight.departure_weather != config.Weather.INVALID:
            departure_weather = flight.departure_weather.value

        arrival_weather = None
        if flight.arrival_weather != config.Weather.INVALID:
            arrival_weather = flight.arrival_weather.value

        data_frame_rows.append({
            "flight_id":            flight.flight_id,
            "departure_airport":    flight.departure_airport,
            "arrival_airport":      flight.arrival_airport,
            "airline":              None if flight.airline == INVALID_AIRLINE else flight.airline,
            "month":                flight.month.value,
            "is_holiday":           flight.is_holiday,
            "departure_weather":    departure_weather,
            "arrival_weather":      arrival_weather,
            "departure_congestion": None if flight.departure_congestion == INVALID_CONGESTION   else flight.departure_congestion,
            "arrival_congestion":   None if flight.arrival_congestion   == INVALID_CONGESTION   else flight.arrival_congestion,
            "ticket_price":         None if flight.ticket_price         == INVALID_TICKET_PRICE else flight.ticket_price,
            "is_delayed":           flight.is_delayed,
        })

    df = pd.DataFrame(data_frame_rows)
    df.to_csv(file_path, index = False)

def main():
    context = init_context()

    TRAIN_DATASET_SIZE = 2000
    train_flights = generate_flights(context, TRAIN_DATASET_SIZE)
    insert_missing_characteristics(context, train_flights)
    insert_outliers(context, train_flights)
    write_flights_to_csv("train.csv", train_flights)

    TEST_DATASET_SIZE = 1000
    test_flights = generate_flights(context, TEST_DATASET_SIZE)
    insert_missing_characteristics(context, test_flights)
    insert_outliers(context, test_flights)
    write_flights_to_csv("test.csv", test_flights)

if __name__ == "__main__":
    main()
