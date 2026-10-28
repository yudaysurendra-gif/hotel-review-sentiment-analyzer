"""Generates a small synthetic hotel review dataset (Review, Rating) so the
project runs without downloading anything. Replace with a real dataset
(e.g. Kaggle 'TripAdvisor Hotel Reviews') for better accuracy."""
import random
import pandas as pd

random.seed(42)

positive = [
    "The room was spotless and the bed was extremely comfortable",
    "Staff were friendly, helpful and always smiling",
    "Great location, walking distance to the beach and restaurants",
    "Breakfast buffet was delicious with lots of variety",
    "Excellent value for money, we will definitely come back",
    "The pool area was beautiful and very well maintained",
    "Check-in was quick and the receptionist was very welcoming",
    "Loved the sea view from our balcony, absolutely amazing",
    "Clean, modern and quiet room, we slept really well",
    "The service was outstanding and the food was fantastic",
]
negative = [
    "The room was dirty and smelled of smoke",
    "Staff were rude and unhelpful when we complained",
    "Terrible location, very noisy at night",
    "Breakfast was cold and the food was tasteless",
    "Overpriced for what you get, a complete waste of money",
    "The bathroom was filthy and the air conditioning was broken",
    "Check-in took over an hour and nobody apologised",
    "Bed bugs in the mattress, worst stay ever",
    "Walls are paper thin and the service was awful",
    "The hotel was run down and not worth the price",
]
neutral = [
    "The room was okay, nothing special but nothing bad either",
    "Average hotel, decent location but the staff were just fine",
    "Breakfast was acceptable, the room was a bit small",
    "It was an ordinary stay, the price was fair for the area",
    "Some things were good and some were disappointing",
    "The hotel is fine for a short stay, not memorable",
]
extras_pos = ["", " Highly recommended.", " Would stay again.", " Truly a lovely stay."]
extras_neg = ["", " Never again.", " Would not recommend.", " Very disappointed."]
extras_neu = ["", " Might stay again.", " It was alright."]

rows = []
for _ in range(400):
    rows.append((random.choice(positive) + random.choice(extras_pos), random.choice([4, 5])))
    rows.append((random.choice(negative) + random.choice(extras_neg), random.choice([1, 2])))
for _ in range(200):
    rows.append((random.choice(neutral) + random.choice(extras_neu), 3))

df = pd.DataFrame(rows, columns=["Review", "Rating"]).sample(frac=1, random_state=1)
df.to_csv("data/hotel_reviews.csv", index=False)
print(f"Saved {len(df)} reviews to data/hotel_reviews.csv")
