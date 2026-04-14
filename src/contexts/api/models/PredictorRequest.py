from pydantic import BaseModel
from typing import Literal

EmailProvider = Literal[
    "yahoo", "sapo", "comcast", "jetbrains", "woodstock",
    "surfeu", "jubii", "embraer", "google", "riotur",
    "hotmail", "gmail", "aol", "rediff", "wp",
    "rogers", "shaw", "microsoft", "yachoo", "apple", "uol",
]

CountryProvider = Literal[
    "Argentina", "Spain", "Italy", "Hungary", "India", "Czech Republic",
    "Belgium", "Sweden", "Chile", "Norway", "France", "USA", "United Kingdom",
    "Netherlands", "Brazil", "Austria", "Poland", "Australia", "Ireland",
    "Germany", "Denmark", "Canada", "Finland", "Portugal"
]

CityProvider = Literal[
    "Porto", "Budapest", "Reno", "London", "Paris", "New York",
    "Salt Lake City", "Prague", "Dijon", "Rome", "Berlin", "Lyon",
    "Bangalore", "Ottawa", "Orlando", "Fort Worth", "Chicago", "Mountain View",
    "Stuttgart", "Delhi", "Stockholm", "Vienne", "Edinburgh",
    "Halifax", "São Paulo", "Madison", "Amsterdam", "Montréal",
    "Sidney", "Madrid", "Lisbon", "Warsaw", "Edmonton", "Vancouver",
    "Dublin", "Toronto", "Santiago", "Brasília", "Bordeaux",
    "Cupertino", "Brussels", "Redmond", "Buenos Aires",
    "São José dos Campos", "Yellowknife", "Boston", "Tucson",
    "Helsinki", "Winnipeg", "Copenhagen", "Oslo", "Frankfurt",
    "Rio de Janeiro"
]
    


class PredictorRequest(BaseModel):
  email: EmailProvider
  country: CountryProvider
  city: CityProvider
