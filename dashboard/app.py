# Barakaldoko airearen kalitatearen aginte-panela
# Denbora errealean erakusten ditu (2026ko datuekin simulatuta) ereduen iragarpenak,ingurumen-egoera (K-means) eta sistema adituaren alertak ziurtasun-faktoreekin.
# dashboard.bat fitxategia exekutatzean abiarazten da

import os
import sys
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

#Jetson Nanoren kode bera erabiltzen dugu
KARPETA = os.path.dirname(os.path.abspath(__file__))
JETSON_KARPETA = os.path.normpath(os.path.join(KARPETA, "..", "jetson"))
sys.path.append(JETSON_KARPETA)
import sistema

ICA_KOLOREAK = {"Oso ona": "#2b9348", "Ona": "#80b918", "Erregularra": "#f9c74f", "Txarra": "#f3722c"}
MAILA_KOLOREAK = {"NORMALA": "#2b9348", "ABISUA": "#f9c74f", "ERTAINA": "#f8961e", "ALTUA": "#d62828"}
EGOERA_KOLOREAK = {"Aire garbia haizearekin": "#4cc9f0", "Gaueko bareta": "#4361ee",
                   "Trafikoko puntako ordua": "#e63946", "Partikula-episodioa": "#9d4edd"}
# Arauak id-aren arabera bilatzeko hiztegia
ARAUAK = {}
for araua in sistema.ARAUAK:
    ARAUAK[araua["id"]] = araua

st.set_page_config(page_title="Airearen kalitatea · Barakaldo", page_icon="🌫️", layout="wide")


# Datuak
@st.cache_resource
def kargatu():
    # Ereduak eta simulazioko datuak behin bakarrik kargatzen ditu
    sis = sistema.sistema_kargatu(os.path.join(JETSON_KARPETA, "ereduak"))
    datuak = sistema.datuak_irakurri(os.path.join(JETSON_KARPETA, "datuak", "simulazioa_2026.csv"))
    return sis, datuak


@st.cache_data
def simulazioa_kalkulatu():
    # 2026ko ordu guztiak sistematik pasatzen ditu (segundo batzuk behar ditu lehen aldian)
    sis, datuak = kargatu()
    errenkadak = []
    for errenkada in datuak:
        emaitza, alertak = sistema.ordua_prozesatu(errenkada, sis)
        errenkadak.append(emaitza)
    return prestatu(pd.DataFrame(errenkadak))


def jetson_emaitzak_irakurri(bidea):
    # Jetsonen sistema_exekutatu.py-k idazten duen CSVa irakurtzen du
    return prestatu(pd.read_csv(bidea, keep_default_na=False, na_values=["", "nan"]))

def prestatu(df):
    df["data"] = pd.to_datetime(df["data"])
    df["arauak"] = df["arauak"].fillna("").astype(str)
    return df.reset_index(drop=True)


def formatua(balioa, hamartarrak=1, unitatea=""):
    if balioa is None or balioa != balioa:
        return "–"
    return str(round(float(balioa), hamartarrak)) + unitatea


def kutxa(testua, kolorea, testu_txikia=""):
    # HTMLz egindako kolore-koadroa
    return ("<div style='background:" + kolorea + ";padding:14px 18px;border-radius:10px;color:#111;'>"
            "<div style='font-size:13px;opacity:.8'>" + testu_txikia + "</div>"
            "<div style='font-size:24px;font-weight:700'>" + testua + "</div></div>")


# Alboko barra: simulazioaren kontrolak
st.sidebar.title("🌫️ Smart Environment AI")
st.sidebar.caption("E2T2 · Barakaldoko estazioa")

iturria = st.sidebar.radio("Datu-iturria", ["Simulazioa (2026ko datuak)", "Jetsoneko emaitzak (CSV)"])

if iturria == "Simulazioa (2026ko datuak)":
    emaitzak = simulazioa_kalkulatu()
else:
    jetson_bidea = st.sidebar.text_input("Jetsoneko fitxategia",
                                         os.path.join(JETSON_KARPETA, "emaitzak", "jetson_emaitzak.csv"))
    if not os.path.exists(jetson_bidea):
        st.warning("Ez da fitxategia aurkitzen. Lehenik exekutatu `python3 jetson/sistema_exekutatu.py` edo kopiatu hona Jetsonen sortutako CSVa.")
        st.stop()
    emaitzak = jetson_emaitzak_irakurri(jetson_bidea)

if "posizioa" not in st.session_state:
    st.session_state.posizioa = 0
    st.session_state.erreproduzitzen = False
    st.session_state.iturria = iturria
if st.session_state.iturria != iturria:
    st.session_state.iturria = iturria
    st.session_state.posizioa = 0

lehena = emaitzak["data"].min().date()
azkena = emaitzak["data"].max().date()
hasiera = st.sidebar.date_input("Hasiera-data", value=max(lehena, min(azkena, pd.Timestamp("2026-03-03").date())),
                                min_value=lehena, max_value=azkena)
abiadura = st.sidebar.slider("Segundoak simulatutako ordu bakoitzeko", 0.2, 5.0, 1.0, 0.1)

zutabea1, zutabea2 = st.sidebar.columns(2)
if zutabea1.button("▶ Abiatu", use_container_width=True):
    st.session_state.erreproduzitzen = True
if zutabea2.button("⏸ Pausa", use_container_width=True):
    st.session_state.erreproduzitzen = False
zutabea3, zutabea4 = st.sidebar.columns(2)
if zutabea3.button("⏭ +1 ordu", use_container_width=True):
    st.session_state.posizioa += 1
if zutabea4.button("⏮ Hasierara", use_container_width=True):
    st.session_state.posizioa = 0
    st.session_state.erreproduzitzen = False

# Erakusten ari den ordua
hasiera_indizea = int(emaitzak.index[emaitzak["data"] >= pd.Timestamp(hasiera)][0])
indizea = min(hasiera_indizea + st.session_state.posizioa, len(emaitzak) - 1)
unekoa = emaitzak.iloc[indizea]
historia = emaitzak.iloc[max(0, indizea - 47): indizea + 1]

st.sidebar.markdown("---")
st.sidebar.caption("Simulatutako ordua (GMT)")
st.sidebar.subheader(unekoa["data"].strftime("%Y/%m/%d %H:%M"))
if st.session_state.erreproduzitzen:
    st.sidebar.caption("Egoera: ▶ erreproduzitzen")
else:
    st.sidebar.caption("Egoera: ⏸ pausan")


# Goiburua eta alerta
st.title("Airearen kalitatea Barakaldon")
st.caption("Denbora errealeko monitorizazioa (2026ko orduko datuekin simulatua) · NexAI Innovations, Bizkaiko Foru Aldundiarentzat")

maila = unekoa["alerta_maila"]
idak = []
for arau_id in unekoa["arauak"].split():
    if arau_id in ARAUAK:
        idak.append(arau_id)
if maila == "NORMALA":
    st.markdown(kutxa("Alertarik ez: airearen kalitate normala", MAILA_KOLOREAK["NORMALA"], "SISTEMA ADITUNA"),
                unsafe_allow_html=True)
else:
    lerroak = []
    for arau_id in idak:
        lerroa = ("<b>[" + arau_id + "]</b> " + ARAUAK[arau_id]["mezua"] + " (ZF " + str(ARAUAK[arau_id]["zf"]) + ") — "
                  + ARAUAK[arau_id]["gomendioa"])
        lerroak.append(lerroa)
    testua = "<br>".join(lerroak)
    st.markdown("<div style='background:" + MAILA_KOLOREAK[maila] + ";padding:14px 18px;border-radius:10px;color:#111;'>"
                "<div style='font-size:13px;opacity:.8'>SISTEMA ADITUNA</div>"
                "<div style='font-size:24px;font-weight:700'>" + maila + " ALERTA · ziurtasuna " + str(round(float(unekoa.get("ziurtasuna", 0)), 1)) + " %</div>"
                "<div style='font-size:15px;margin-top:6px'>" + testua + "</div></div>", unsafe_allow_html=True)

st.write("")

# Uneko orduko adierazleak
# Aurreko ordua (aldaketa kalkulatzeko)
if indizea > 0:
    aurrekoa = emaitzak.iloc[indizea - 1]
else:
    aurrekoa = unekoa

# NO2aren aldaketa aurreko orduarekiko (aurrekoa NaN bada, ez da erakusten)
if aurrekoa["NO2"] != aurrekoa["NO2"]:
    no2_aldaketa = None
else:
    no2_aldaketa = formatua(unekoa["NO2"] - aurrekoa["NO2"], 0)

k1, k2, k3, k4 = st.columns(4)
k1.metric("NO2 orain", formatua(unekoa["NO2"], 0, " µg/m³"),
          delta=no2_aldaketa,
          delta_color="inverse")
k2.metric("Hurrengo orduko NO2 iragarria (MLP)", formatua(unekoa["NO2_iragarria"], 1, " µg/m³"),
          delta=formatua(unekoa["NO2_iragarria"] - unekoa["NO2"], 1), delta_color="inverse")
k3.markdown(kutxa(unekoa["ICA_iragarria"], ICA_KOLOREAK.get(unekoa["ICA_iragarria"], "#ccc"),
                  "Hurrengo orduko kalitate iragarria (zuhaitza)"), unsafe_allow_html=True)
k4.markdown(kutxa(unekoa["egoera"], EGOERA_KOLOREAK.get(unekoa["egoera"], "#ccc"),
                  "Ingurumen-egoera (K-means)"), unsafe_allow_html=True)

st.write("")
k5, k6, k7, k8 = st.columns(4)
k5.metric("PM10 24 orduko batez bestekoa", formatua(unekoa["PM10_batez_24h"], 1, " µg/m³"))
k6.metric("PM2.5 24 orduko batez bestekoa", formatua(unekoa["PM25_batez_24h"], 1, " µg/m³"))
k7.metric("Haizea", formatua(unekoa["haize_abiadura"], 2, " m/s"))
k8.metric("Tenperatura", formatua(unekoa["tenperatura"], 1, " ºC"))

ziurtasuna = unekoa.get("ziurtasuna", float("nan"))
if ziurtasuna == ziurtasuna:
    st.progress(min(1.0, max(0.0, float(ziurtasuna) / 100)),
                text="Hurrengo orduan kutsadura-arazoa egoteko ziurtasuna (arauen ziurtasun-faktoreak konbinatuta): "
                     + str(round(float(ziurtasuna), 1)) + " %")

# Azken 48 orduetako grafikoak
st.subheader("Azken 48 orduak")
ezkerra, eskuina = st.columns(2)

no2 = historia[["data", "NO2_benetako_hurrengoa", "NO2_iragarria"]].rename(
    columns={"NO2_benetako_hurrengoa": "Benetako NO2a (hurrengo ordua)", "NO2_iragarria": "NO2 iragarria (MLP)"})
# Azken orduak ez du oraindik benetako baliorik: kendu egiten dugu
if len(no2) > 1:
    no2 = no2.iloc[:-1]
no2 = no2.melt("data", var_name="seriea", value_name="µg/m³")
no2_grafikoa = alt.Chart(no2).mark_line(point=True).encode(
    x=alt.X("data:T", title=None, axis=alt.Axis(format="%m/%d %H:%M")), y=alt.Y("µg/m³:Q"),
    color=alt.Color("seriea:N", legend=alt.Legend(orient="bottom", title=None)),
    tooltip=["data:T", "seriea:N", alt.Tooltip("µg/m³:Q", format=".1f")],
).properties(height=260, title="NO2: iragarria vs benetakoa")
ezkerra.altair_chart(no2_grafikoa, use_container_width=True)

pm = historia[["data", "PM10_batez_24h", "PM25_batez_24h"]].rename(
    columns={"PM10_batez_24h": "PM10 24 h batez bestekoa", "PM25_batez_24h": "PM2.5 24 h batez bestekoa"})
pm = pm.melt("data", var_name="seriea", value_name="µg/m³")
pm_lerroak = alt.Chart(pm).mark_line().encode(
    x=alt.X("data:T", title=None, axis=alt.Axis(format="%m/%d %H:%M")), y=alt.Y("µg/m³:Q"),
    color=alt.Color("seriea:N", legend=alt.Legend(orient="bottom", title=None)),
    tooltip=["data:T", "seriea:N", alt.Tooltip("µg/m³:Q", format=".1f")])
mugak = alt.Chart(pd.DataFrame({"y": [35, 50], "muga": ["Erregularra (PM10)", "Txarra (PM10)"]})).mark_rule(
    strokeDash=[5, 4]).encode(y=alt.Y("y:Q", title="µg/m³"), color=alt.value("#d62828"))
eskuina.altair_chart((pm_lerroak + mugak).properties(height=260, title="Partikulak (PM10en ICA mugak gorriz)"),
                     use_container_width=True)

zerrendak = historia[["data", "egoera", "alerta_maila", "ICA_iragarria"]].copy()
zerrendak["amaiera"] = zerrendak["data"] + pd.Timedelta(hours=1)
egoera_zerrenda = alt.Chart(zerrendak).mark_rect().encode(
    x=alt.X("data:T", title=None, axis=alt.Axis(format="%m/%d %H:%M")), x2="amaiera:T",
    color=alt.Color("egoera:N", scale=alt.Scale(domain=list(EGOERA_KOLOREAK.keys()),
                                                range=list(EGOERA_KOLOREAK.values())),
                    legend=alt.Legend(orient="bottom", title=None)),
    tooltip=["data:T", "egoera:N"]).properties(height=45, title="Ingurumen-egoera (K-means)")
ica_zerrenda = alt.Chart(zerrendak).mark_rect().encode(
    x=alt.X("data:T", title=None, axis=alt.Axis(format="%m/%d %H:%M")), x2="amaiera:T",
    color=alt.Color("ICA_iragarria:N", scale=alt.Scale(domain=list(ICA_KOLOREAK.keys()), range=list(ICA_KOLOREAK.values())),
                    legend=alt.Legend(orient="bottom", title=None)),
    tooltip=["data:T", "ICA_iragarria:N"]).properties(height=45, title="Kalitate iragarria (zuhaitza)")
alerta_zerrenda = alt.Chart(zerrendak).mark_rect().encode(
    x=alt.X("data:T", title=None, axis=alt.Axis(format="%m/%d %H:%M")), x2="amaiera:T",
    color=alt.Color("alerta_maila:N", scale=alt.Scale(domain=list(MAILA_KOLOREAK.keys()), range=list(MAILA_KOLOREAK.values())),
                    legend=alt.Legend(orient="bottom", title=None)),
    tooltip=["data:T", "alerta_maila:N"]).properties(height=45, title="Alerta-maila (sistema adituna)")
st.altair_chart(alt.vconcat(egoera_zerrenda, ica_zerrenda, alerta_zerrenda).resolve_scale(color="independent"),
                use_container_width=True)

# Alerten historia eta xehetasunak
st.subheader("Alerten historia (azken 48 orduak)")
alertadunak = historia[historia["alerta_maila"] != "NORMALA"].iloc[::-1]
if len(alertadunak) == 0:
    st.info("Ez da alertarik egon azken 48 orduetan.")
else:
    zutabeak = ["data", "alerta_maila", "arauak", "mezuak", "ICA_iragarria", "PM10_batez_24h", "haize_abiadura"]
    if "ziurtasuna" in alertadunak.columns:
        zutabeak.insert(3, "ziurtasuna")
    taula = alertadunak[zutabeak].round(1)
    st.dataframe(taula.rename(columns={"alerta_maila": "maila", "ICA_iragarria": "ICA iragarria",
                                       "PM10_batez_24h": "PM10 24 h", "haize_abiadura": "haizea"}),
                 use_container_width=True, hide_index=True)

with st.expander("Zergatik iragartzen du zuhaitzak kategoria hau?"):
    sis, datuak = kargatu()
    gakoa = unekoa["data"].strftime("%Y-%m-%d %H:%M")
    errenkadak = []
    for e in datuak:
        if e["data"] == gakoa:
            errenkadak.append(e)
    if len(errenkadak) > 0:
        x = sistema.aurreprozesatu(errenkadak[0], sis)
        st.write("Erabaki-zuhaitzean jarraitutako bidea:")
        st.code(sistema.zuhaitzaren_araua(x, sis).replace(" ETA ", "\nETA "))

with st.expander("Sistemaren errendimendua"):
    zutabeak = ["aurreprozesamendua_ms", "mlp_ms", "zuhaitza_ms", "kmeans_ms", "adituna_ms", "guztira_ms"]
    denborak = emaitzak[zutabeak].mean().round(4).to_frame("batez bestekoa (ms orduko)")
    st.dataframe(denborak, use_container_width=True)
    benetakoak = emaitzak["NO2_benetako_hurrengoa"].astype(float)
    baliozkoak = benetakoak.notna()
    st.write("Iragarritako NO2aren MAE aldi osoan:",
             round(float(np.abs(benetakoak[baliozkoak] - emaitzak.loc[baliozkoak, "NO2_iragarria"]).mean()), 2), "µg/m³")
    st.write("ICA kategoriaren asmatzeak:",
             round(100 * float((emaitzak["ICA_iragarria"] == emaitzak["ICA_benetako_hurrengoa"]).mean()), 1), "%")

st.caption("Datu-iturria: Open Data Euskadi — Eusko Jaurlaritzaren airearen kalitatea kontrolatzeko sarea (Barakaldoko estazioa). "
           "Orduak GMTn.")

# Aurrerapen automatikoa
if st.session_state.erreproduzitzen and indizea < len(emaitzak) - 1:
    time.sleep(abiadura)
    st.session_state.posizioa += 1
    st.rerun()