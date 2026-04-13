# 🏎️ F1 Machine Learning Predictor (Advanced Simulator)

Welcome to the **F1 Racing Predictor**, a custom-built, deeply engineered AI simulator designed to predict Formula 1 race winners. Unlike basic prediction models that only look at points or grid positions, this project uses a powerful `RandomForestClassifier` trained on **4 full years (2021–2024)** of millimeter-accurate racing telemetry to understand the actual physics, team capabilities, and historical dominance behind every circuit.

---

## 🧠 How The Simulator Works

The core strength of this project lies in taking raw F1 racing data and translating it into real-world racing factors that actually decide who wins.

* **Massive History:** The script naturally downloads speed, timing, and weather data for almost **90 historical races** spanning 4 full years.
* **Weather Tracking:** It automatically looks up the exact track temperature for each circuit to teach the AI how heat affects the cars differently.
* **Smart Storage:** The code organizes all these learned patterns into tiny, optimized dictionary files so that predictions load instantly.

### Custom Racing Features
We built incredibly smart logic into the model to represent complex racing variables:
1. **Car Specs (Mechanical Advantage):** We evaluate engine and aero capabilities by dynamically extracting the 90th percentile of `Speed Traps (SpeedST)` per team across different tracks.
2. **Support Team Performance:** We quantify pit-crew efficiency by calculating the exact duration between `PitInTime` and `PitOutTime` globally for each constructor.
3. **Historical Track Dominance:** The model tracks `AvgTrackPosition` to evaluate a driver's exact skill level and dominance on a per-circuit basis (e.g., Max Verstappen at Suzuka).
4. **Recent Form (Momentum):** Tracks the immediate 2024 win-ratios of drivers and teams for momentum influence.

---

## 📈 Model Performance

When trained over the immense 4-year data set, the Random Forest model achieves an incredible **96.6% accuracy** rating on the test data! 

The algorithm has dynamically learned that:
1. **Driver Track Preference** (25%) and **Grid Position** (21%) are the most critical factors.
2. **Weather / Track Temp** (19%) acts as a massive conditional modifier that can swing win probabilities between drivers like Verstappen and Piastri depending on how hot the asphalt gets.
3. Because top teams have identical 2-second pitstops in the modern era, **Pit Times** are generally negligible in predicting the outright winner over a 60-lap race.

---

## 🚀 How to Run the Simulator

You can use the resulting Machine Learning model to simulate "What-If" scenarios or predict upcoming races.

1. Open `predict.py` in your code editor.
2. Change the `RACE` variable to specify the circuit you want to test (e.g., `"Miami Grand Prix"`, `"Japanese Grand Prix"`, `"Monaco Grand Prix"`).
3. (Optional) Tweak the `race_entry` list if you want to simulate a custom Grid Lineup or move drivers to different teams!
4. Run the prediction script:

```bash
python predict.py
```

*Note: The script will automatically fetch and apply the highly specific historical average Track Temperature for the `RACE` you selected, ensuring top-tier accuracy!*
