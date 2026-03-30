from __future__ import annotations

import matplotlib.pyplot as plt
import math
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai


load_dotenv()


@st.cache_data
def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Ex: "20141013T000000" -> datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["price_per_sqft"] = df["price"] / df["sqft_living"].replace({0: pd.NA})
    df["age"] = df["date"].dt.year - df["yr_built"]
    df["is_renovated"] = df["yr_renovated"] > 0
    df["has_basement"] = df["sqft_basement"] > 0

    return df


def build_market_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtres interactifs")

    min_price = int(df["price"].min())
    max_price = int(df["price"].max())
    price_range = st.sidebar.slider(
        "Fourchette de prix",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
        step=10_000,
    )

    bedrooms_values = sorted(df["bedrooms"].dropna().unique().tolist())
    selected_bedrooms = st.sidebar.multiselect(
        "Nombre de chambres",
        options=bedrooms_values,
        default=bedrooms_values,
    )

    zipcode_values = sorted(df["zipcode"].dropna().unique().tolist())
    selected_zipcodes = st.sidebar.multiselect(
        "Code postal (zipcode)",
        options=zipcode_values,
        default=zipcode_values[:10] if len(zipcode_values) > 10 else zipcode_values,
    )

    min_grade = int(df["grade"].min())
    max_grade = int(df["grade"].max())
    grade_range = st.sidebar.slider(
        "Grade de construction",
        min_value=min_grade,
        max_value=max_grade,
        value=(min_grade, max_grade),
    )

    min_year = int(df["yr_built"].min())
    max_year = int(df["yr_built"].max())
    year_built_range = st.sidebar.slider(
        "Annee de construction",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
    )

    waterfront_only = st.sidebar.checkbox("Front de mer uniquement", value=False)

    filtered_df = df.copy()
    filtered_df = filtered_df[
        (filtered_df["price"] >= price_range[0])
        & (filtered_df["price"] <= price_range[1])
        & (filtered_df["grade"] >= grade_range[0])
        & (filtered_df["grade"] <= grade_range[1])
        & (filtered_df["yr_built"] >= year_built_range[0])
        & (filtered_df["yr_built"] <= year_built_range[1])
    ]

    if selected_bedrooms:
        filtered_df = filtered_df[filtered_df["bedrooms"].isin(selected_bedrooms)]
    else:
        filtered_df = filtered_df.iloc[0:0]

    if selected_zipcodes:
        filtered_df = filtered_df[filtered_df["zipcode"].isin(selected_zipcodes)]
    else:
        filtered_df = filtered_df.iloc[0:0]

    if waterfront_only:
        filtered_df = filtered_df[filtered_df["waterfront"] == 1]

    return filtered_df


def render_market_tab(filtered_df: pd.DataFrame) -> None:
    st.subheader("Exploration du marche")
    st.caption("Les metriques et visualisations se mettent a jour selon les filtres.")

    if filtered_df.empty:
        st.warning("Aucun resultat pour les filtres actuels. Ajustez les criteres.")
        return

    st.markdown("### B. Metriques cles (KPIs)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("N - Nombre de proprietes", f"{len(filtered_df):,}")
    col2.metric("$̄ - Prix moyen", f"${filtered_df['price'].mean():,.0f}")
    col3.metric("$̃ - Prix median", f"${filtered_df['price'].median():,.0f}")
    col4.metric("$/pi2 - Prix moyen par pied carre", f"${filtered_df['price_per_sqft'].mean():,.0f}")

    st.markdown("### C. Visualisations (matplotlib)")

    # 1) Histogramme de la distribution des prix
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.hist(filtered_df["price"], bins=30, color="steelblue", edgecolor="white")
    ax1.set_title("Distribution des prix des proprietes")
    ax1.set_xlabel("Prix ($)")
    ax1.set_ylabel("Nombre de proprietes")
    ax1.grid(axis="y", alpha=0.25)
    st.pyplot(fig1)
    plt.close(fig1)

    # 2) Nuage de points: prix vs superficie, colore par grade
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    scatter = ax2.scatter(
        filtered_df["sqft_living"],
        filtered_df["price"],
        c=filtered_df["grade"],
        cmap="viridis",
        alpha=0.7,
        s=20,
    )
    ax2.set_title("Prix vs superficie habitable (couleur = grade)")
    ax2.set_xlabel("Superficie habitable (sqft_living)")
    ax2.set_ylabel("Prix ($)")
    cbar = fig2.colorbar(scatter, ax=ax2)
    cbar.set_label("Grade")
    ax2.grid(alpha=0.2)
    st.pyplot(fig2)
    plt.close(fig2)

    # 3) Matrice de correlation entre variables numeriques pertinentes
    corr_cols = [
        "price",
        "sqft_living",
        "bedrooms",
        "bathrooms",
        "grade",
        "yr_built",
        "price_per_sqft",
    ]
    corr_df = filtered_df[corr_cols].corr(numeric_only=True)

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    im = ax3.imshow(corr_df, cmap="coolwarm", vmin=-1, vmax=1)
    ax3.set_title("Matrice de correlation (variables numeriques)")
    ax3.set_xticks(range(len(corr_cols)))
    ax3.set_yticks(range(len(corr_cols)))
    ax3.set_xticklabels(corr_cols, rotation=45, ha="right")
    ax3.set_yticklabels(corr_cols)
    fig3.colorbar(im, ax=ax3, label="Correlation")
    st.pyplot(fig3)
    plt.close(fig3)

    # 4) Diagramme en barres: prix moyen par zipcode (top 10)
    top10_zip = (
        filtered_df.groupby("zipcode", as_index=False)["price"]
        .mean()
        .sort_values("price", ascending=False)
        .head(10)
    )
    fig4, ax4 = plt.subplots(figsize=(9, 4))
    ax4.bar(top10_zip["zipcode"].astype(str), top10_zip["price"], color="teal")
    ax4.set_title("Top 10 zipcodes par prix moyen")
    ax4.set_xlabel("Zipcode")
    ax4.set_ylabel("Prix moyen ($)")
    ax4.tick_params(axis="x", rotation=45)
    ax4.grid(axis="y", alpha=0.25)
    st.pyplot(fig4)
    plt.close(fig4)

    st.markdown("### Donnees filtrees")
    st.dataframe(filtered_df.head(200), width="stretch")

    st.markdown("### D. Resume genere par LLM")
    if st.button("Generer un resume du marche", type="primary"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error(
                "Aucune cle API detectee. Ajoute `GOOGLE_API_KEY=...` dans ton fichier `.env`, puis relance Streamlit."
            )
            return

        n = int(len(filtered_df))
        mean_price = float(filtered_df["price"].mean())
        median_price = float(filtered_df["price"].median())
        min_price = float(filtered_df["price"].min())
        max_price = float(filtered_df["price"].max())
        mean_price_sqft = float(filtered_df["price_per_sqft"].mean())

        grade_counts = filtered_df["grade"].value_counts().sort_index()
        grade_distribution = ", ".join([f"{int(k)}: {int(v)}" for k, v in grade_counts.items()])

        pct_waterfront = float((filtered_df["waterfront"] == 1).mean() * 100)

        prompt = f"""Tu es un analyste immobilier senior. Voici les statistiques d'un segment
du marche immobilier du comte de King (Seattle) :

- Nombre de proprietes : {n}
- Prix moyen : {mean_price:,.0f} $
- Prix median : {median_price:,.0f} $
- Prix min / max : {min_price:,.0f} $ / {max_price:,.0f} $
- Prix moyen par pi2 : {mean_price_sqft:,.0f} $
- Repartition par grade : {grade_distribution}
- % front de mer : {pct_waterfront:.1f}%

Redige un resume executif de ce segment en 3-4 paragraphes.
Identifie les tendances cles et les opportunites d'investissement.
Contraintes de sortie:
1) Le resume doit mentionner explicitement toutes les statistiques ci-dessus.
2) Le ton doit etre professionnel et analytique.
3) Ne pas inventer de donnees non fournies.
"""

        with st.spinner("Generation du resume en cours..."):
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=prompt,
            )

        st.markdown(response.text)


def render_property_tab(df_all: pd.DataFrame) -> None:
    st.subheader("Analyse d'une propriete")
    st.markdown("### A. Selection de la propriete")

    if df_all.empty:
        st.warning("Aucune donnee disponible.")
        return

    mode = st.radio(
        "Mode de selection",
        options=["Selectbox filtrable (ID + prix)", "Filtres progressifs (zipcode -> chambres -> selection)"],
        horizontal=True,
    )

    selected_row = None

    if mode == "Selectbox filtrable (ID + prix)":
        options_df = df_all[["id", "price"]].copy()
        options_df["id_str"] = options_df["id"].astype("int64").astype(str)
        options = options_df["id_str"].tolist()
        id_to_price = dict(zip(options_df["id_str"], options_df["price"].astype(float)))
        selected_id = st.selectbox(
            "Choisir une maison",
            options=options,
            format_func=lambda x: f"ID {x} - ${id_to_price[x]:,.0f}",
        )
        selected_row = df_all[df_all["id"].astype("int64").astype(str) == selected_id].iloc[0]
    else:
        zipcodes = sorted(df_all["zipcode"].dropna().astype(int).unique().tolist())
        selected_zip = st.selectbox("1) Zipcode", options=zipcodes)

        step_df = df_all[df_all["zipcode"] == selected_zip]
        bedrooms = sorted(step_df["bedrooms"].dropna().unique().tolist())
        selected_bedrooms = st.selectbox("2) Nombre de chambres", options=bedrooms)

        step_df = step_df[step_df["bedrooms"] == selected_bedrooms]
        step_options_df = step_df[["id", "price"]].copy()
        step_options_df["id_str"] = step_options_df["id"].astype("int64").astype(str)
        step_options = step_options_df["id_str"].tolist()
        step_prices = dict(zip(step_options_df["id_str"], step_options_df["price"].astype(float)))
        selected_id = st.selectbox(
            "3) Selection de la propriete",
            options=step_options,
            format_func=lambda x: f"ID {x} - ${step_prices[x]:,.0f}",
        )
        selected_row = step_df[step_df["id"].astype("int64").astype(str) == selected_id].iloc[0]

    st.success(
        f"Propriete selectionnee: ID {int(selected_row['id'])} | Zipcode {int(selected_row['zipcode'])} | "
        f"{int(selected_row['bedrooms'])} chambres | ${float(selected_row['price']):,.0f}"
    )

    st.markdown("### B. Fiche descriptive")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Prix", f"${float(selected_row['price']):,.0f}")
    k2.metric("Surface habitable", f"{float(selected_row['sqft_living']):,.0f} pi2")
    k3.metric("Chambres", f"{int(selected_row['bedrooms'])}")
    k4.metric("Salles de bain", f"{float(selected_row['bathrooms']):.1f}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grade", f"{int(selected_row['grade'])}")
    c2.metric("Annee de construction", f"{int(selected_row['yr_built'])}")
    c3.metric("Prix / pi2", f"${float(selected_row['price_per_sqft']):,.0f}")
    c4.metric("Age de la maison", f"{int(selected_row['age'])} ans")

    flags_col1, flags_col2, flags_col3 = st.columns(3)
    flags_col1.metric("Renovee", "Oui" if bool(selected_row["is_renovated"]) else "Non")
    flags_col2.metric("Sous-sol", "Oui" if bool(selected_row["has_basement"]) else "Non")
    flags_col3.metric("Front de mer", "Oui" if bool(selected_row["waterfront"] == 1) else "Non")

    fiche_df = pd.DataFrame(
        [
            {"Champ": "ID", "Valeur": int(selected_row["id"])},
            {"Champ": "Date de vente", "Valeur": str(selected_row["date"])},
            {"Champ": "Zipcode", "Valeur": int(selected_row["zipcode"])},
            {"Champ": "Prix", "Valeur": f"${float(selected_row['price']):,.0f}"},
            {"Champ": "Sqft living", "Valeur": f"{float(selected_row['sqft_living']):,.0f}"},
            {"Champ": "Sqft lot", "Valeur": f"{float(selected_row['sqft_lot']):,.0f}"},
            {"Champ": "Floors", "Valeur": float(selected_row["floors"])},
            {"Champ": "Condition", "Valeur": int(selected_row["condition"])},
            {"Champ": "Grade", "Valeur": int(selected_row["grade"])},
        ]
    )
    st.dataframe(fiche_df, width="stretch", hide_index=True)

    st.markdown("### C. Recherche de comparables")
    sqft_ref = float(selected_row["sqft_living"])
    sqft_min = sqft_ref * 0.8
    sqft_max = sqft_ref * 1.2

    comps_df = df_all[
        (df_all["zipcode"] == selected_row["zipcode"])
        & (df_all["bedrooms"] == selected_row["bedrooms"])
        & (df_all["sqft_living"] >= sqft_min)
        & (df_all["sqft_living"] <= sqft_max)
        & (df_all["id"] != selected_row["id"])
    ].copy()

    st.caption(
        "Criteres utilises: meme zipcode, meme nombre de chambres, "
        "superficie habitable entre -20% et +20%."
    )
    st.metric("Nombre de comparables trouves", f"{len(comps_df):,}")

    if comps_df.empty:
        st.warning("Aucun comparable trouve avec ces criteres.")
    else:
        # Proximite basee sur l'ecart de superficie habitable
        comps_df["distance_sqft"] = (comps_df["sqft_living"] - sqft_ref).abs()
        comps_df = comps_df.sort_values("distance_sqft").head(10)

        selected_price = float(selected_row["price"])
        comps_mean_price = float(comps_df["price"].mean())
        price_gap = selected_price - comps_mean_price
        price_gap_pct = (price_gap / comps_mean_price * 100) if comps_mean_price != 0 else 0.0

        m1, m2, m3 = st.columns(3)
        m1.metric("Prix moyen des comparables", f"${comps_mean_price:,.0f}")
        m2.metric("Ecart en $ (maison - comparables)", f"${price_gap:,.0f}")
        m3.metric("Ecart en %", f"{price_gap_pct:+.1f}%")

        if price_gap > 0:
            st.warning("Diagnostic: maison en surcote par rapport au marche local des comparables.")
        elif price_gap < 0:
            st.success("Diagnostic: maison en decote par rapport au marche local des comparables.")
        else:
            st.info("Diagnostic: maison au prix du marche local des comparables.")

        st.markdown("### D. Visualisation comparative (matplotlib)")
        metrics_labels = ["Prix ($)", "Prix / pi2 ($)", "Surface (pi2)", "Grade"]
        selected_values = [
            float(selected_row["price"]),
            float(selected_row["price_per_sqft"]),
            float(selected_row["sqft_living"]),
            float(selected_row["grade"]),
        ]
        comps_values = [
            float(comps_df["price"].mean()),
            float(comps_df["price_per_sqft"].mean()),
            float(comps_df["sqft_living"].mean()),
            float(comps_df["grade"].mean()),
        ]

        fig_comp, ax_comp = plt.subplots(figsize=(10, 5))
        x = range(len(metrics_labels))
        width = 0.36
        bars_comps = ax_comp.bar(
            [i - width / 2 for i in x],
            comps_values,
            width=width,
            color="steelblue",
            label="Moyenne comparables",
        )
        bars_selected = ax_comp.bar(
            [i + width / 2 for i in x],
            selected_values,
            width=width,
            color="crimson",
            label="Propriete selectionnee",
        )
        ax_comp.set_title("Comparaison: propriete selectionnee vs moyenne des comparables")
        ax_comp.set_xlabel("Dimensions comparees")
        ax_comp.set_ylabel("Valeur")
        ax_comp.set_xticks(list(x))
        ax_comp.set_xticklabels(metrics_labels)
        ax_comp.legend()
        ax_comp.grid(axis="y", alpha=0.25)

        # Annotation visuelle de la propriete selectionnee
        for bar in bars_selected:
            height = bar.get_height()
            ax_comp.annotate(
                "Selectionnee",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="crimson",
            )

        st.pyplot(fig_comp)
        plt.close(fig_comp)

        # Graphique radar pour comparaison multidimensionnelle
        st.markdown("#### Graphique radar")
        radar_labels = ["Prix", "Prix/pi2", "Surface", "Grade", "Sdb"]
        selected_raw = [
            float(selected_row["price"]),
            float(selected_row["price_per_sqft"]),
            float(selected_row["sqft_living"]),
            float(selected_row["grade"]),
            float(selected_row["bathrooms"]),
        ]
        comps_raw = [
            float(comps_df["price"].mean()),
            float(comps_df["price_per_sqft"].mean()),
            float(comps_df["sqft_living"].mean()),
            float(comps_df["grade"].mean()),
            float(comps_df["bathrooms"].mean()),
        ]

        # Normalisation simple pour rendre les dimensions comparables (0-1)
        max_values = [max(a, b, 1e-9) for a, b in zip(selected_raw, comps_raw)]
        selected_norm = [v / m for v, m in zip(selected_raw, max_values)]
        comps_norm = [v / m for v, m in zip(comps_raw, max_values)]

        angles = [n / float(len(radar_labels)) * 2 * math.pi for n in range(len(radar_labels))]
        angles += angles[:1]
        selected_norm += selected_norm[:1]
        comps_norm += comps_norm[:1]

        fig_radar, ax_radar = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
        ax_radar.plot(angles, comps_norm, color="steelblue", linewidth=2, label="Moyenne comparables")
        ax_radar.fill(angles, comps_norm, color="steelblue", alpha=0.20)
        ax_radar.plot(angles, selected_norm, color="crimson", linewidth=2, label="Propriete selectionnee")
        ax_radar.fill(angles, selected_norm, color="crimson", alpha=0.15)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(radar_labels)
        ax_radar.set_yticklabels([])
        ax_radar.set_title("Radar: propriete selectionnee vs comparables (normalise)")
        ax_radar.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2)
        ax_radar.annotate(
            "Propriete selectionnee",
            xy=(angles[0], selected_norm[0]),
            xytext=(15, 15),
            textcoords="offset points",
            color="crimson",
            fontsize=9,
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": "crimson", "lw": 1.5},
        )
        st.pyplot(fig_radar)
        plt.close(fig_radar)

        comps_view = comps_df[
            ["id", "price", "bedrooms", "bathrooms", "sqft_living", "distance_sqft", "grade", "yr_built", "zipcode"]
        ]
        st.caption("Top 10 comparables les plus proches (selon la superficie habitable).")
        st.dataframe(comps_view, width="stretch", hide_index=True)

        st.markdown("### E. Recommandation generee par LLM")
        if st.button("Generer une recommandation", key="btn_reco_property", type="primary"):
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                st.error(
                    "Aucune cle API detectee. Ajoute `GOOGLE_API_KEY=...` dans ton fichier `.env`, puis relance Streamlit."
                )
                return

            if price_gap > 0:
                valuation_status = "surcote"
            elif price_gap < 0:
                valuation_status = "decote"
            else:
                valuation_status = "au prix du marche"

            comps_min_price = float(comps_df["price"].min())
            comps_max_price = float(comps_df["price"].max())
            comps_median_price = float(comps_df["price"].median())
            comps_mean_ppsf = float(comps_df["price_per_sqft"].mean())
            comps_mean_grade = float(comps_df["grade"].mean())
            comps_mean_sqft = float(comps_df["sqft_living"].mean())

            renovated_text = "Oui" if bool(selected_row["is_renovated"]) else "Non"
            waterfront_text = "Oui" if bool(selected_row["waterfront"] == 1) else "Non"
            status_text = (
                "Surcote"
                if valuation_status == "surcote"
                else ("Decote" if valuation_status == "decote" else "Au prix du marche")
            )

            prompt = f"""Tu es un analyste immobilier senior. Analyse uniquement les donnees suivantes:

PROPRIETE ANALYSEE :
- Prix : {float(selected_row['price']):,.0f} $
- Chambres : {int(selected_row['bedrooms'])} | Salles de bain : {float(selected_row['bathrooms']):.1f}
- Superficie : {float(selected_row['sqft_living']):,.0f} pi2 | Terrain : {float(selected_row['sqft_lot']):,.0f} pi2
- Grade : {int(selected_row['grade'])}/13 | Condition : {int(selected_row['condition'])}/5
- Annee de construction : {int(selected_row['yr_built'])} | Renovee : {renovated_text}
- Front de mer : {waterfront_text} | Vue : {int(selected_row['view'])}/4

ANALYSE COMPARATIVE :
- Nombre de comparables trouves : {len(comps_df)}
- Prix moyen des comparables : {comps_mean_price:,.0f} $
- Ecart vs comparables : {price_gap:+,.0f} $ ({price_gap_pct:+.1f}%)
- Statut : {status_text}

Redige une recommandation d'investissement en 3-4 paragraphes.
Inclus obligatoirement :
- evaluation du prix
- forces et faiblesses
- verdict final (Acheter / A surveiller / Eviter) avec justification

Contraintes :
- Reponse en francais, ton professionnel
- Ne pas inventer de donnees
"""

            with st.spinner("Generation de la recommandation en cours..."):
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model="gemini-3.1-flash-lite-preview",
                    contents=prompt,
                )
            st.markdown(response.text)


def main() -> None:
    st.set_page_config(page_title="Analyseur immobilier", layout="wide")
    st.title("Analyseur immobilier")

    df = load_and_prepare_data("kc_house_data.csv")
    filtered_df = build_market_filters(df)

    tab1, tab2 = st.tabs(["Exploration du marche", "Analyse d'une propriete"])

    with tab1:
        render_market_tab(filtered_df)

    with tab2:
        render_property_tab(df_all=df)


if __name__ == "__main__":
    main()

