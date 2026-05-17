import tkinter as tk
import numpy as np
import train
from tkinter import ttk
from config import Month, Weather

def predict_delay():
    # These values are not used by the model, but it would be kinda werid to
    # predict if the flight will be delayed when the user doesn't even specify
    # the departure/arrival airports...
    departure_airport = departure_airport_entry.get()
    arrival_airport = arrival_airport_entry.get()
    airline = airline_entry.get()

    if not departure_airport or not arrival_airport or not airline:
        return

    departure_month = Month(month_var.get())
    departure_weather = Weather(departure_weather_var.get())
    arrival_weather = Weather(arrival_weather_var.get())

    is_holiday = holiday_var.get()
    try:
        departure_congestion = float(departure_congestion_entry.get())
        arrival_congestion = float(arrival_congestion_entry.get())
        ticket_price = int(ticket_price_entry.get())
    except ValueError:
        return

    flight = pd.DataFrame([{
        "is_holiday": int(is_holiday),
        "departure_congestion": departure_congestion,
        "arrival_congestion": arrival_congestion,
        "ticket_price": ticket_price,
    }])

    model = train.build_model("train.csv")
    is_delayed = model.predict(flight)[0]
    print(f"is_delayed={is_delayed}")

    result_label.config(
        text=(
            f"Departure Airport: {departure_airport}\n"
            f"Arrival Airport: {arrival_airport}\n"
            f"Airline: {airline}\n"
            f"Month: {departure_month.value}\n"
            f"Holiday: {'Yes' if is_holiday else 'No'}\n"
            f"Departure Weather: {departure_weather.value}\n"
            f"Arrival Weather: {arrival_weather.value}\n"
            f"Ticket Price: {ticket_price}\n\n"
            f"Is Delayed: {is_delayed}"
        )
    )

window = tk.Tk()
window.title("Flight Delay Predictor")
window.geometry("450x520")

month_values = [
    Month.JANUARY.value,
    Month.FEBRUARY.value,
    Month.MARCH.value,
    Month.APRIL.value,
    Month.MAY.value,
    Month.JUNE.value,
    Month.JULY.value,
    Month.AUGUST.value,
    Month.SEPTEMBER.value,
    Month.OCTOBER.value,
    Month.NOVEMBER.value,
    Month.DECEMBER.value,
]

weather_values = [
    Weather.CLEAR.value,
    Weather.RAIN.value,
    Weather.FOG.value,
    Weather.SNOW.value,
    Weather.RAINSTORM.value,
    Weather.THUNDERSTORM.value,
]

tk.Label(window, text="Departure Airport").pack()
departure_airport_entry = tk.Entry(window)
departure_airport_entry.pack()

tk.Label(window, text="Arrival Airport").pack()
arrival_airport_entry = tk.Entry(window)
arrival_airport_entry.pack()

tk.Label(window, text="Airline").pack()
airline_entry = tk.Entry(window)
airline_entry.pack()

tk.Label(window, text="Departure Month").pack()
month_var = tk.StringVar(value=Month.JANUARY.value)
month_dropdown = ttk.Combobox(
    window,
    textvariable=month_var,
    values=month_values,
    state="readonly"
)
month_dropdown.pack()

holiday_var = tk.BooleanVar()
holiday_checkbox = tk.Checkbutton(
    window,
    text="Is Holiday",
    variable=holiday_var
)
holiday_checkbox.pack(pady=10)

tk.Label(window, text="Departure Weather").pack()
departure_weather_var = tk.StringVar(value=Weather.CLEAR.value)
departure_weather_dropdown = ttk.Combobox(
    window,
    textvariable=departure_weather_var,
    values=weather_values,
    state="readonly"
)
departure_weather_dropdown.pack()

tk.Label(window, text="Arrival Weather").pack()
arrival_weather_var = tk.StringVar(value=Weather.CLEAR.value)
arrival_weather_dropdown = ttk.Combobox(
    window,
    textvariable=arrival_weather_var,
    values=weather_values,
    state="readonly"
)
arrival_weather_dropdown.pack()

tk.Label(window, text="Departure Congestion").pack()
departure_congestion_entry = tk.Entry(window)
departure_congestion_entry.pack()

tk.Label(window, text="Arrival Congestion").pack()
arrival_congestion_entry = tk.Entry(window)
arrival_congestion_entry.pack()

tk.Label(window, text="Ticket Price").pack()
ticket_price_entry = tk.Entry(window)
ticket_price_entry.pack()

predict_button = tk.Button(
    window,
    text="Predict Delay",
    command=predict_delay
)
predict_button.pack(pady=20)

result_label = tk.Label(window, text="Prediction will appear here", justify="left")
result_label.pack()

window.mainloop()
