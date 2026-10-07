import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="NFL 2026 Game Predictor", page_icon="🏈", layout="wide")

ACCENT = "#2a78d6"
MUTED = "#898781"
GRID = "rgba(137,135,129,0.25)"


@st.cache_data
def load(data_stamp):
    games = pd.read_csv("nfl_app/games_2026.csv", parse_dates=["gameday"])
    teams = pd.read_csv("nfl_app/teams.csv")
    elo_hist = pd.read_csv("nfl_app/elo_history.csv", parse_dates=["gameday"])
    report = pd.read_csv("nfl_app/report.csv")
    exam = pd.read_csv("nfl_app/exam_2024_2025.csv")
    pick_log = pd.read_csv("nfl_app/pick_log.csv") if os.path.exists("nfl_app/pick_log.csv") else None
    return games, teams, elo_hist, report, exam, pick_log


games, teams, elo_hist, report, exam, pick_log = load(max(os.path.getmtime(f"nfl_app/{f}") for f in os.listdir("nfl_app") if f.endswith(".csv")))

# Show the pick that was saved before kickoff whenever one exists
games["locked"] = False
if pick_log is not None:
    locked = pick_log.drop_duplicates("game_id", keep="last").set_index("game_id")
    is_locked = games["game_id"].isin(locked.index)
    for col in ["home_win_prob", "pick", "confidence"]:
        games.loc[is_locked, col] = games.loc[is_locked, "game_id"].map(locked[col])
    games["locked"] = is_locked
    is_final = games["status"] == "Final"
    games.loc[is_final, "correct"] = (games.loc[is_final, "pick"] == games.loc[is_final, "winner"]).astype(float)
final = games[games["status"] == "Final"]
upcoming = games[games["status"] == "Upcoming"]
last_week = int(final["week"].max()) if len(final) else 0
next_week = int(upcoming["week"].min()) if len(upcoming) else last_week


def style(fig, height=360):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8), showlegend=False,
        hoverlabel=dict(font_size=13), bargap=0.35,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def result_label(row):
    if row["status"] == "Upcoming":
        return ""
    return "✅ Correct" if row["correct"] == 1 else "❌ Missed"


# ---------- Header ----------
st.title("🏈 NFL 2026 Game Predictor")
st.caption(f"Logistic regression on nflverse data since 2003. Results through Week {last_week}. "
           "Rerun the notebook each week to refresh.")

model_row = report.set_index("method").loc["My model"]
vegas_right = ((final["vegas_prob"] >= 0.5) == (final["home_score"] > final["away_score"])).sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("2026 picks correct", f"{int(final['correct'].sum())} of {len(final)}  ({final['correct'].mean():.0%})")
k2.metric("Vegas favorite, 2026", f"{vegas_right} of {len(final)}  ({vegas_right / max(len(final), 1):.0%})")
k3.metric("Test accuracy, 2024 to 2025", f"{model_row['accuracy']:.1%}",
          help=f"Brier score {model_row['brier']:.3f} on 570 games the model never trained on")
k4.metric("Games left to predict", len(upcoming))

tab_week, tab_team, tab_power, tab_report = st.tabs(
    ["Weekly picks", "Team outlook", "Power rankings", "Model report card"])

# ---------- Weekly picks ----------
with tab_week:
    weeks = sorted(games["week"].unique())
    week = st.selectbox("Week", weeks, index=weeks.index(next_week), key="week")
    wk = games[games["week"] == week].copy()
    wk["Matchup"] = wk["away_team"] + " @ " + wk["home_team"] + wk["neutral"].map({1: " (neutral)", 0: ""})
    wk["Win chance"] = wk["confidence"] * 100
    wk["Vegas on pick"] = (wk["vegas_prob"].where(wk["pick"] == wk["home_team"], 1 - wk["vegas_prob"])) * 100
    wk["Score"] = wk.apply(lambda r: "" if r["status"] == "Upcoming"
                           else f"{r['away_team']} {int(r['away_score'])}, {r['home_team']} {int(r['home_score'])}", axis=1)
    wk["Result"] = wk.apply(result_label, axis=1)
    wk["Date"] = wk["gameday"].dt.strftime("%a %b %d")
    wk["Pick"] = wk["pick"]

    if (wk["status"] == "Final").any():
        done = wk[wk["status"] == "Final"]
        note = " (picks locked before kickoff)" if done["locked"].all() else ""
        st.markdown(f"**Week {week}: {int(done['correct'].sum())} of {len(done)} correct**{note}")
    else:
        st.markdown(f"**Week {week}: {len(wk)} games.** Picks further out are pulled toward 50% "
                    "because a lot can change before then.")

    st.dataframe(
        wk[["Date", "Matchup", "Pick", "Win chance", "Vegas on pick", "Score", "Result"]],
        hide_index=True, width="stretch",
        column_config={
            "Win chance": st.column_config.ProgressColumn("Win chance", min_value=50, max_value=100, format="%.0f%%"),
            "Vegas on pick": st.column_config.NumberColumn("Vegas on pick", format="%.0f%%"),
        },
    )

# ---------- Team outlook ----------
with tab_team:
    team_list = sorted(teams["team"])
    team = st.selectbox("Team", team_list, index=team_list.index(teams.iloc[0]["team"]), key="team")
    t = teams.set_index("team").loc[team]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Record", f"{int(t['wins'])} and {int(t['losses'])}")
    c2.metric("Projected wins", f"{t['projected_wins']:.1f}")
    c3.metric(f"Power rank (Elo {t['elo']:.0f})", f"#{int(t['rank'])} of 32")
    c4.metric(f"Starting QB (rating {t['qb_rating']:+.3f})", t["qb"])

    sched = games[(games["home_team"] == team) | (games["away_team"] == team)].copy()
    is_home = sched["home_team"] == team
    sched["Opponent"] = sched["away_team"].where(is_home, sched["home_team"])
    sched["Where"] = is_home.map({True: "Home", False: "Away"}).where(sched["neutral"] == 0, "Neutral")
    sched["win_prob"] = sched["home_win_prob"].where(is_home, 1 - sched["home_win_prob"])
    sched["Result"] = sched.apply(
        lambda r: "" if r["status"] == "Upcoming" else ("W" if r["winner"] == team else ("T" if r["winner"] == "TIE" else "L")), axis=1)

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**{team} win chance, game by game**")
        fig = go.Figure(go.Bar(
            x=sched["week"], y=sched["win_prob"] * 100,
            marker_color=[ACCENT if s == "Upcoming" else MUTED for s in sched["status"]],
            marker_cornerradius=4,
            customdata=sched[["Opponent", "Where", "Result"]],
            hovertemplate="<b>%{y:.0f}%</b> win chance<br>Week %{x} vs %{customdata[0]} (%{customdata[1]})"
                          "<br>%{customdata[2]}<extra></extra>",
        ))
        fig.add_hline(y=50, line_dash="dot", line_color=MUTED)
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
        fig.update_xaxes(title="Week", dtick=1)
        st.plotly_chart(style(fig), width="stretch", theme="streamlit")
        st.caption("Gray bars are games already played, shown with the pregame win chance. Blue bars are upcoming. The dotted line is 50%.")
    with right:
        st.markdown("**Schedule**")
        st.dataframe(
            sched.assign(**{"Win chance": sched["win_prob"] * 100})[["week", "Opponent", "Where", "Win chance", "Result"]]
                 .rename(columns={"week": "Wk"}),
            hide_index=True, width="stretch", height=380,
            column_config={"Win chance": st.column_config.NumberColumn(format="%.0f%%")},
        )

    st.markdown(f"**{team} Elo rating since 2016**")
    h = elo_hist[elo_hist["team"] == team]
    fig = go.Figure(go.Scatter(
        x=h["gameday"], y=h["elo"], mode="lines", line=dict(color=ACCENT, width=2),
        hovertemplate="<b>%{y:.0f}</b><br>%{x|%b %d, %Y}<extra></extra>",
    ))
    fig.add_hline(y=1500, line_dash="dot", line_color=MUTED,
                  annotation_text="League average 1500", annotation_position="bottom left")
    st.plotly_chart(style(fig, 300), width="stretch", theme="streamlit")

# ---------- Power rankings ----------
with tab_power:
    st.markdown("**Projected 2026 wins** (wins so far plus the win chance of every game left)")
    ranked = teams.sort_values("projected_wins")
    fig = go.Figure(go.Bar(
        x=ranked["projected_wins"], y=ranked["team"], orientation="h",
        marker_color=ACCENT, marker_cornerradius=4,
        customdata=ranked[["wins", "losses", "qb"]],
        hovertemplate="<b>%{x:.1f}</b> projected wins<br>%{y}: %{customdata[0]} and %{customdata[1]} so far"
                      "<br>QB %{customdata[2]}<extra></extra>",
    ))
    fig.add_vline(x=8.5, line_dash="dot", line_color=MUTED)
    fig.update_xaxes(range=[0, 17], title="Projected wins", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(style(fig, 720), width="stretch", theme="streamlit")
    st.caption("The dotted line at 8.5 wins is a .500 season.")

    st.markdown("**Full table**, sorted by Elo")
    table = teams.assign(Record=teams["wins"].astype(str) + " and " + teams["losses"].astype(str))
    st.dataframe(
        table[["rank", "team", "elo", "qb", "qb_rating", "def_epa", "Record", "projected_wins"]].rename(columns={
            "rank": "Rank", "team": "Team", "elo": "Elo", "qb": "QB", "qb_rating": "QB rating",
            "def_epa": "Defense EPA allowed", "projected_wins": "Projected wins"}),
        hide_index=True, width="stretch", height=600,
        column_config={
            "Elo": st.column_config.NumberColumn(format="%.0f"),
            "QB rating": st.column_config.NumberColumn(format="%+.3f"),
            "Defense EPA allowed": st.column_config.NumberColumn(format="%+.3f", help="Lower is better"),
            "Projected wins": st.column_config.ProgressColumn(min_value=0, max_value=17, format="%.1f"),
        },
    )

# ---------- Model report card ----------
with tab_report:
    left, right = st.columns(2)
    with left:
        st.markdown("**Final exam: 2024 and 2025 seasons (570 games)**")
        r = report.copy()
        fig = go.Figure(go.Bar(
            x=r["method"], y=r["accuracy"] * 100,
            marker_color=[ACCENT if m == "My model" else MUTED for m in r["method"]],
            marker_cornerradius=4,
            text=[f"{a:.1%}" for a in r["accuracy"]], textposition="outside",
            customdata=r["brier"],
            hovertemplate="<b>%{y:.1f}%</b> accuracy<br>%{x}<br>Brier %{customdata:.4f}<extra></extra>",
        ))
        fig.update_yaxes(range=[0, 80], ticksuffix="%", title="Games picked correctly")
        st.plotly_chart(style(fig), width="stretch", theme="streamlit")
        st.dataframe(r.assign(accuracy=r["accuracy"] * 100).rename(
            columns={"method": "Method", "accuracy": "Accuracy", "brier": "Brier (lower is better)"}),
            hide_index=True, width="stretch",
            column_config={"Accuracy": st.column_config.NumberColumn(format="%.1f%%"),
                           "Brier (lower is better)": st.column_config.NumberColumn(format="%.4f")})

    with right:
        st.markdown("**Calibration: when the model says X%, does it happen X% of the time?**")
        exam["bucket"] = pd.cut(exam["model_prob"], [0, .3, .4, .5, .6, .7, 1])
        cal = exam.groupby("bucket", observed=True).agg(
            predicted=("model_prob", "mean"), actual=("home_win", "mean"), games=("home_win", "size")).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", hoverinfo="skip",
                                 line=dict(color=MUTED, dash="dot", width=1)))
        fig.add_trace(go.Scatter(
            x=cal["predicted"] * 100, y=cal["actual"] * 100, mode="lines+markers",
            line=dict(color=ACCENT, width=2), marker=dict(size=10, color=ACCENT, line=dict(width=2, color="white")),
            customdata=cal["games"],
            hovertemplate="Model said <b>%{x:.0f}%</b><br>Home team won <b>%{y:.0f}%</b><br>%{customdata} games<extra></extra>",
        ))
        fig.update_xaxes(range=[0, 100], ticksuffix="%", title="Model's home win chance")
        fig.update_yaxes(range=[0, 100], ticksuffix="%", title="How often the home team won")
        st.plotly_chart(style(fig), width="stretch", theme="streamlit")
        st.caption("Dots near the dotted line mean the probabilities can be trusted.")

    st.markdown("**2026 week by week**")
    weekly = final.assign(vegas_right=(final["vegas_prob"] >= 0.5) == (final["home_score"] > final["away_score"]))
    weekly = weekly.groupby("week").agg(Games=("correct", "size"), Model=("correct", "sum"),
                                        Vegas=("vegas_right", "sum"), locked=("locked", "all")).reset_index()
    weekly["Picks"] = weekly["locked"].map({True: "Locked before kickoff", False: "Rebuilt from earlier data"})
    weekly = weekly.rename(columns={"week": "Week"})[["Week", "Games", "Model", "Vegas", "Picks"]]
    st.dataframe(weekly.astype({"Week": int, "Games": int, "Model": int, "Vegas": int}), hide_index=True, width="stretch")
    st.caption("Weeks 1 to 3 happened before I started saving picks, so they are rebuilt using only data from before each week. "
               "From Week 4 on, the record uses the picks saved before kickoff.")
