import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Player Difficulty Recommender", page_icon="🎮", layout="centered")

@st.cache_resource(show_spinner=False)
def load_artifacts():
    path = Path("artifacts/model_artifacts.joblib")
    if not path.exists():
        raise FileNotFoundError(
            "artifacts/model_artifacts.joblib not found. Please run the notebook export cell to save trained artifacts."
        )
    return joblib.load(path)

art = load_artifacts()
cluster_features = art["cluster_features"]
model_features = art["model_features"]
cluster_to_group = art["cluster_to_group"]
scaler_cluster = art["scaler_cluster"]
kmeans = art["kmeans"]
logreg_pipe = art["logreg_pipe"]
rf_clf = art["rf_clf"]

def recommend(group: str) -> str:
    if group == "Struggling":
        return "Decrease Difficulty"
    if group == "Balanced":
        return "Keep Difficulty Same"
    return "Increase Difficulty"

st.title("🎮 Player Difficulty Recommender")
st.write("Enter a player's stats to get a difficulty recommendation.")

tab1, tab2 = st.tabs(["Single Player", "Batch Upload (CSV)"])

# =========================
# TAB 1 — SINGLE PLAYER
# =========================
with tab1:
    with st.form("player_form"):
        col1, col2 = st.columns(2)

        with col1:
            kdRatio = st.number_input("K/D Ratio (kdRatio)", min_value=0.0, value=1.0, step=0.01)
            scorePerMinute = st.number_input("Score Per Minute", min_value=0.0, value=250.0, step=1.0)
            wins = st.number_input("Wins", min_value=0, value=10, step=1)
            losses = st.number_input("Losses", min_value=0, value=10, step=1)
            timePlayed = st.number_input("Time Played (minutes)", min_value=0.0, value=500.0, step=10.0)
            gamesPlayed = st.number_input("Games Played", min_value=0, value=50, step=1)

        with col2:
            xp = st.number_input("XP", min_value=0, value=20000, step=100)
            level = st.number_input("Level", min_value=0, value=20, step=1)
            prestige = st.number_input("Prestige", min_value=0, value=0, step=1)
            headshots = st.number_input("Headshots", min_value=0, value=50, step=1)
            assists = st.number_input("Assists", min_value=0, value=30, step=1)
            shots = st.number_input("Shots", min_value=0, value=1000, step=10)

        hits = st.number_input("Hits (for accuracy calc: hits/shots)", min_value=0, value=int(shots * 0.25), step=1)
        deaths = st.number_input("Deaths", min_value=0, value=gamesPlayed * 5, step=1)

        submitted = st.form_submit_button("Predict Difficulty")

    #  Render result OUTSIDE the form so it doesn't leak into other tabs
    if submitted:
        accuracy = 0.0 if shots == 0 else (hits / shots)

        row = {
            "kdRatio": kdRatio,
            "scorePerMinute": scorePerMinute,
            "accuracy": accuracy,
            "wins": wins,
            "losses": losses,
            "timePlayed": timePlayed,
            "gamesPlayed": gamesPlayed,
            "xp": xp,
            "level": level,
            "prestige": prestige,
            "headshots": headshots,
            "assists": assists,
            "shots": shots,
            "deaths": deaths,
        }

        # Fill missing model features with 0 (safety)
        for feat in model_features:
            if feat not in row:
                row[feat] = 0

        X_one = pd.DataFrame([row], columns=model_features)

        # KMeans
        X_cluster = X_one[cluster_features].copy()
        Z = scaler_cluster.transform(X_cluster)
        clus = int(kmeans.predict(Z)[0])
        group_kmeans = cluster_to_group.get(clus, "Balanced")

        # Supervised
        pred_lr = str(logreg_pipe.predict(X_one)[0])
        pred_rf = str(rf_clf.predict(X_one)[0])

        # Majority vote
        votes = [group_kmeans, pred_lr, pred_rf]
        final_group = pd.Series(votes).mode().iloc[0]
        difficulty = recommend(final_group)

        st.subheader("Result")
        st.metric("Final Group", final_group)
        st.metric("Difficulty Recommendation", difficulty)



        with st.expander("Input features used"):
            st.dataframe(X_one, use_container_width=True)


# =========================
# TAB 2 — BATCH UPLOAD
# =========================
with tab2:
    st.subheader("Upload CSV for Predictions")
    st.write("Upload a CSV file with player stats. ")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df_upload)} players from CSV")

            possible_id_cols = ["Player_ID", "player_id", "PlayerId", "playerId", "PLAYER_ID", "id", "ID", "username", "Username"]
            id_col = next((c for c in possible_id_cols if c in df_upload.columns), None)


            results_list = []

            for idx, row_data in df_upload.iterrows():
                try:
                    shots_val = row_data.get("shots", 0)
                    hits_val = row_data.get("hits", 0)

                    try:
                        shots_val = float(shots_val)
                        hits_val = float(hits_val)
                    except:
                        shots_val = 0.0
                        hits_val = 0.0

                    acc = 0.0 if shots_val == 0 else (hits_val / shots_val)

                    X_row = {}
                    for feat in model_features:
                        if feat == "accuracy":
                            X_row[feat] = acc
                        else:
                            X_row[feat] = row_data.get(feat, 0)

                    X_batch = pd.DataFrame([X_row], columns=model_features)

                    # KMeans
                    X_clus = X_batch[cluster_features].copy()
                    Z = scaler_cluster.transform(X_clus)
                    clus_id = int(kmeans.predict(Z)[0])
                    g_kmeans = cluster_to_group.get(clus_id, "Balanced")

                    # Supervised
                    p_lr = str(logreg_pipe.predict(X_batch)[0])
                    p_rf = str(rf_clf.predict(X_batch)[0])

                    # Majority vote
                    votes_list = [g_kmeans, p_lr, p_rf]
                    f_group = pd.Series(votes_list).mode().iloc[0]
                    diff = recommend(f_group)

                    #player_id = row_data.get("Player_ID", idx)
                    player_id = row_data.get(id_col, idx) if id_col else idx

                    results_list.append({
                        "Player_ID": player_id,
                        "Final_Group": f_group,
                        "Difficulty_Recommendation": diff,
                    })

                except Exception as e:
                    st.warning(f"Row {idx} error: {str(e)}")
                    continue

            if results_list:
                df_results = pd.DataFrame(results_list)

                st.subheader("Predictions Preview")
                st.dataframe(df_results, use_container_width=True)

                csv_output = df_results.to_csv(index=False)
                st.download_button(
                    label=" Download Predictions as CSV",
                    data=csv_output,
                    file_name="player_difficulty_predictions.csv",
                    mime="text/csv",
                )

                st.subheader("Summary Stats")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Total Players", len(df_results))
                with c2:
                    st.metric("Most Common Group", df_results["Final_Group"].mode().values[0])
                with c3:
                    st.metric("Most Common Recommendation", df_results["Difficulty_Recommendation"].mode().values[0])

                c4, c5 = st.columns(2)
                with c4:
                    st.write("**Group Distribution**")
                    st.bar_chart(df_results["Final_Group"].value_counts())
                with c5:
                    st.write("**Difficulty Recommendation Distribution**")
                    st.bar_chart(df_results["Difficulty_Recommendation"].value_counts())
            else:
                st.error("No valid predictions could be made. Check your CSV data.")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
