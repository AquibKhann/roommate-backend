# main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

# Allow frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class UserProfile(BaseModel):
    name: str
    gender: str
    age: int
    location: str
    cleanliness: float  # normalized 0–1
    smoking: str
    sleep_schedule: str
    pets: str
    social_pref: str
    extraversion_score: float
    neuroticism_score: float
    agreeableness_score: float
    conscientiousness_score: float
    openness_score: float


# Feature weights for cosine similarity
weights = {
    "extraversion_score": 2.0,
    "neuroticism_score": 2.0,
    "agreeableness_score": 2.0,
    "conscientiousness_score": 2.0,
    "openness_score": 2.0,
    "cleanliness": 1.5,
    "smoking_Yes": 1.2,
    "social_pref": 1.0,
    "sleep_schedule": 1.0,
    "pets": 1.0,
    "location": 0.5,
}

@app.post("/match")
async def match_user(user: UserProfile):
    try:
        # Fetch all users from Supabase
        response = supabase.table("users").select("*").execute()
        all_users = pd.DataFrame(response.data)

        if all_users.empty:
            raise HTTPException(status_code=404, detail="No users found in Supabase.")

        # Drop the same user if exists (e.g., matching against self)
        all_users = all_users[all_users["name"] != user.name]

        # Create input profile DataFrame
        input_df = pd.DataFrame([user.dict()])

        # Merge and align features
        all_users = all_users.dropna()
        all_users = all_users.reset_index(drop=True)
        features = [
            "extraversion_score", "neuroticism_score", "agreeableness_score",
            "conscientiousness_score", "openness_score", "cleanliness"
        ]

        for lifestyle in ["smoking", "pets", "social_pref", "sleep_schedule"]:
            for value in all_users[lifestyle].unique():
                col = f"{lifestyle}_{value}"
                all_users[col] = (all_users[lifestyle] == value).astype(int)
                input_df[col] = int(getattr(user, lifestyle) == value)
                if col not in weights:
                    weights[col] = 1.0  # Default weight for one-hot lifestyle

        # Location rule (same state bonus)
        input_df["location"] = user.location
        all_users["location"] = all_users["location"]
        all_users["location_match"] = (all_users["location"] == user.location).astype(int)
        input_df["location_match"] = 1  # always 1 for self

        features += [f for f in input_df.columns if f.startswith(("smoking_", "pets_", "social_pref_", "sleep_schedule_"))]
        features.append("location_match")

        # Normalize and weight features
        def apply_weights(df):
            for f in features:
                if f in weights:
                    df[f] = df[f] * weights[f]
            return df

        X = apply_weights(all_users[features])
        y = apply_weights(input_df[features])

        # Compute cosine similarity
        sims = cosine_similarity(X, y)[..., 0]
        all_users["similarity"] = sims

        # Apply business logic adjustments
        def apply_business_rules(row):
            score = row["similarity"]
            if row["location"] == user.location:
                score += 0.05
            if row["cleanliness"] >= 0.75 and user.cleanliness <= 0.25:
                score -= 0.08
            if row["smoking"] != user.smoking:
                score -= 0.15
            if row["social_pref"] == user.social_pref:
                score += 0.008
            if row["sleep_schedule"] != user.sleep_schedule:
                score -= 0.05
            if row["pets"] == user.pets:
                score += 0.03
            return score

        all_users["final_score"] = all_users.apply(apply_business_rules, axis=1)

        top_matches = all_users.sort_values("final_score", ascending=False).head(5)

        return {
            "top_matches": top_matches[[
                "name", "gender", "location", "cleanliness", "smoking",
                "sleep_schedule", "pets", "social_pref", "final_score"
            ]].rename(columns={"final_score": "similarity"}).to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
