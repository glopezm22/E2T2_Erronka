#Garbiketa, aldagai berriak sortzea eta aurreprozesamenduko pipeline-a

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

# Kutsatzaileen zutabeak (balio negatiboak garbitzeko)
KUTSATZAILEAK = ["CO", "CO_8h", "NO", "NOX", "NO2", "PM10", "PM25", "SO2"]

# Ereduek erabiliko dituzten aldagaiak
ALDAGAI_NUMERIKOAK = [
    # kutsatzaileak uneko orduan
    "CO", "NO", "NO2", "PM10", "PM25", "SO2",
    # meteorologia
    "tenperatura", "hezetasuna", "haize_abiadura",
    # azken orduetako historia
    "NO2_duela_1h", "NO2_duela_24h", "NO2_batez_24h",
    "PM10_batez_24h", "PM25_batez_24h",
]
ALDAGAI_KATEGORIKOAK = ["ordua", "aste_eguna", "hilabetea", "haize_sektorea"]

# Iragarri nahi duguna
ERREGRESIO_HELBURUA = "NO2_hurrengoa"     # hurrengo orduko NO2
SAILKAPEN_HELBURUA = "ICA_hurrengoa"      # hurrengo orduko ICA kategoria

# Kutsatzaile bakoitzaren erregresio-helburua (hurrengo orduko balioa)
KUTSATZAILE_HELBURUAK = {}
KUTSATZAILE_HELBURUAK["NO2"] = "NO2_hurrengoa"
KUTSATZAILE_HELBURUAK["PM10"] = "PM10_hurrengoa"
KUTSATZAILE_HELBURUAK["PM25"] = "PM25_hurrengoa"

def balio_ezinezkoak_kendu(df):
    # Fisikoki egon ezin diren balioak -> nulu bihurtzen ditugu
    df = df.copy()
    for zutabea in KUTSATZAILEAK + ["haize_abiadura"]:
        df.loc[df[zutabea] < 0, zutabea] = np.nan
    # Hezetasuna 0 eta 100 artean
    gaizki = (df["hezetasuna"] < 0) | (df["hezetasuna"] > 100)
    df.loc[gaizki, "hezetasuna"] = np.nan
    # Haizearen norabidea 0 eta 360 artean
    gaizki = (df["haize_norabidea"] < 0) | (df["haize_norabidea"] > 360)
    df.loc[gaizki, "haize_norabidea"] = np.nan
    # Tenperatura -20 eta 50 ºC artean
    gaizki = (df["tenperatura"] < -20) | (df["tenperatura"] > 50)
    df.loc[gaizki, "tenperatura"] = np.nan
    return df


def outlierrak_zenbatu_iqr(df, zutabeak):
    # Outlierrak kuartil arteko heinaren (IQR) metodoarekin zenbatzen ditu
    errenkadak = []
    for zutabea in zutabeak:
        q1 = df[zutabea].quantile(0.25)
        q3 = df[zutabea].quantile(0.75)
        iqr = q3 - q1
        beheko_muga = q1 - 1.5 * iqr
        goiko_muga = q3 + 1.5 * iqr
        txikiegiak = df[zutabea] < beheko_muga
        handiegiak = df[zutabea] > goiko_muga
        n_outlier = (txikiegiak | handiegiak).sum()
        datudun_orduak = df[zutabea].notna().sum()
        ehunekoa = 100 * n_outlier / datudun_orduak
        errenkadak.append([zutabea, round(beheko_muga, 2), round(goiko_muga, 2), n_outlier, round(ehunekoa, 2)])
    zutabe_izenak = ["aldagaia", "beheko_muga", "goiko_muga", "n_outlier", "ehunekoa"]
    return pd.DataFrame(errenkadak, columns=zutabe_izenak)


def outlierrak_zenbatu_zscore(df, zutabeak, muga=3):
    # Outlierrak z-score-arekin zenbatzen ditu: batez bestekotik "muga" desbideratze baino urrunago dauden balioak
    errenkadak = []
    for zutabea in zutabeak:
        batez_bestekoa = df[zutabea].mean()
        desbideratzea = df[zutabea].std()
        z = (df[zutabea] - batez_bestekoa) / desbideratzea
        n_outlier = (z.abs() > muga).sum()
        goiko_muga = batez_bestekoa + muga * desbideratzea
        ehunekoa = 100 * n_outlier / df[zutabea].notna().sum()
        errenkadak.append([zutabea, round(goiko_muga, 2), n_outlier, round(ehunekoa, 2)])
    zutabe_izenak = ["aldagaia", "goiko_muga", "n_outlier", "ehunekoa"]
    return pd.DataFrame(errenkadak, columns=zutabe_izenak)


def aldagaiak_sortu(df):
    # Neurketetatik abiatuta aldagai berriak sortzen ditu
    df = df.copy()
    # Egutegia tokiko orduan
    tokiko_data = df.index.tz_localize("UTC").tz_convert("Europe/Madrid")
    df["ordua"] = tokiko_data.hour
    df["aste_eguna"] = tokiko_data.dayofweek      # 0 = astelehena, 6 = igandea
    df["hilabetea"] = tokiko_data.month

    # Haizearen norabidea diskretizatu: 8 sektore (0 = iparra, 1 = ipar-ekialdea, ... 7 = ipar-mendebaldea)
    # 22,5 gradu mugitzen dugu iparra 337,5 eta 22,5 graduen artean gera dadin
    mugitua = (df["haize_norabidea"] + 22.5) % 360
    df["haize_sektorea"] = pd.cut(mugitua, bins=range(0, 361, 45), labels=range(8), right=False).astype(float)

    # Azken orduetako historia (iraganeko datuak bakarrik)
    df["NO2_duela_1h"] = df["NO2"].shift(1)
    df["NO2_duela_24h"] = df["NO2"].shift(24)
    df["NO2_batez_24h"] = df["NO2"].rolling(24, min_periods=18).mean()
    df["PM10_batez_24h"] = df["PM10"].rolling(24, min_periods=18).mean()
    df["PM25_batez_24h"] = df["PM25"].rolling(24, min_periods=18).mean()

    # Helburuak: HURRENGO ORDUKO balioa
    df[ERREGRESIO_HELBURUA] = df["NO2"].shift(-1)
    df["PM10_hurrengoa"] = df["PM10"].shift(-1)
    df["PM25_hurrengoa"] = df["PM25"].shift(-1)
    df[SAILKAPEN_HELBURUA] = df["ICA_estazioa"].shift(-1)
    return df


def aurreprozesadorea_sortu(numerikoak, kategorikoak, pca_erabili=False, pca_bariantza=0.95):
    # Aldagai numerikoen urratsak: nuluak medianarekin bete eta estandarizatu
    urrats_numerikoak = Pipeline([
        ("inputatu", SimpleImputer(strategy="median")),
        ("eskalatu", StandardScaler()),
    ])

    # Kategorikoen urratsak: maizen agertzen den balioarekin bete eta one-hot
    urrats_kategorikoak = Pipeline([
        ("inputatu", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Urrats-talde bakoitza bere zutabeei aplikatu
    eraldatzailea = ColumnTransformer([
        ("numerikoak", urrats_numerikoak, numerikoak),
        ("kategorikoak", urrats_kategorikoak, kategorikoak),
    ])
    # PCA dimentsioak murrizteko
    if pca_erabili:
        return Pipeline([
            ("eraldatu", eraldatzailea),
            ("pca", PCA(n_components=pca_bariantza)),
        ])
    return Pipeline([("eraldatu", eraldatzailea)])
