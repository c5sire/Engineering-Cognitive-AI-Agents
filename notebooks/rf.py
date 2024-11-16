# Cookbook: OpenAI Beta Client + Pydantic Response Formatting

from datetime import datetime
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

T = TypeVar("T")


# 1. Basic Example with Simple Model
class User(BaseModel):
  name: str
  age: int


def get_basic_user() -> User | None:
  response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
      {
        "role": "user",
        "content": "Generate a random user",
      }
    ],
    response_format=User,
  )
  return response.choices[0].message.parsed


# 2. Nested Models Example
class Address(BaseModel):
  street: str
  city: str
  country: str


class DetailedUser(BaseModel):
  name: str
  age: int
  address: Address


def get_detailed_user() -> DetailedUser | None:
  response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
      {
        "role": "user",
        "content": "Generate a user with address details",
      }
    ],
    response_format=DetailedUser,
  )
  return response.choices[0].message.parsed


# 3. List of Items Example
class Product(BaseModel):
  name: str
  price: float
  in_stock: bool


class ProductList(BaseModel):
  products: list[Product]


def get_product_list() -> list[Product] | None:
  response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
      {
        "role": "user",
        "content": "Generate 3 random products",
      }
    ],
    response_format=ProductList,
  )
  parsed = response.choices[0].message.parsed
  return parsed.products if parsed else None


# 4. Complex Model with Optional Fields
class OrderItem(BaseModel):
  product_name: str
  quantity: int
  price_per_unit: float


class Order(BaseModel):
  order_id: str
  customer_name: str
  items: list[OrderItem]
  total_amount: float
  discount: float | None = None
  order_date: str


def get_order() -> Order | None:
  """Get a random order.

  Returns
  -------
  Order | None
      The generated order or None if parsing fails.
  """
  response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    max_tokens=8192,
    messages=[
      {
        "role": "user",
        "content": "Generate a random order with the date in ISO format (YYYY-MM-DD)",
      }
    ],
    response_format=Order,
  )
  return response.choices[0].message.parsed


# If you need to work with datetime objects, you can create a helper function:
def order_with_datetime(order: Order) -> Order:
  """Convert the order_date string to a datetime object.

  Parameters
  ----------
  order : Order
      The order with string date

  Returns
  -------
  Order
      The order with datetime date
  """
  if order:
    order.order_date = datetime.fromisoformat(
      order.order_date
    )
  return order


# 5. Example with System Message and User Context
class MovieRecommendation(BaseModel):
  title: str
  genre: str
  rating: float
  description: str


def get_movie_recommendation(
  preferred_genre: str,
) -> MovieRecommendation | None:
  response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    max_tokens=8192,
    messages=[
      {
        "role": "system",
        "content": "You are a movie recommendation expert.",
      },
      {
        "role": "user",
        "content": f"Suggest a {preferred_genre} movie",
      },
    ],
    response_format=MovieRecommendation,
  )
  return response.choices[0].message.parsed


# Usage Examples
if __name__ == "__main__":
  # Basic user
  user = get_basic_user()
  print(f"Basic User: {user}")

  # Detailed user with address
  detailed_user = get_detailed_user()
  print(f"Detailed User: {detailed_user}")

  # List of products
  products = get_product_list()
  print(f"Products: {products}")

  # Complex order
  order = get_order()
  if order:
    order = order_with_datetime(order)
  print(f"Order: {order}")

  # Movie recommendation
  movie = get_movie_recommendation("action")
  print(f"Movie Recommendation: {movie}")


# Error Handling Example
def safe_parse_response(
  model_class: type[T], prompt: str
) -> T | None:
  try:
    response = client.beta.chat.completions.parse(
      model="gpt-4o-mini",
      max_tokens=8192,
      messages=[{"role": "user", "content": prompt}],
      response_format=model_class,
    )
    return response.choices[0].message.parsed
  except Exception as e:
    print(f"Error parsing response: {e}")
    return None
