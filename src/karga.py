# Barakaldoko estazioaren datuak kargatzeko funtzioak Iturria Open Data Euskadi

import pandas as pd

# Zutabeen izen berriak (laburragoak eta ikur arraroik gabe)
ZUTABE_IZENAK = {
    "CO (mg/m3)": "CO",
    "CO 8h (mg/m3)": "CO_8h",
    "NO (µg/m3)": "NO",
    "NOX (µg/m3)": "NOX",
    "NO2 (µg/m3)": "NO2",
    "PM10 (µg/m3)": "PM10",
    "PM2,5 (µg/m3)": "PM25",
    "SO2 (µg/m3)": "SO2",
    "D.vien (grados)": "haize_norabidea",
    "H (%)": "hezetasuna",
    "Tº (ºC)": "tenperatura",
    "V.vien (m/s)": "haize_abiadura",
}

# Indize-fitxategiko zutabe interesgarriak (ICA kategoriak)
ICA_ZUTABEAK = {
    "NO2 - ICA": "ICA_NO2",
    "PM10 - ICA": "ICA_PM10",
    "PM2,5 - ICA": "ICA_PM25",
    "SO2 - ICA": "ICA_SO2",
    "ICA Estación": "ICA_estazioa",
}

# ICA etiketak urte batetik bestera aldatzen dira (Muy bueno / Muy buena, Malo / Txarra...)
# Hiztegi honek guztiak berdin uzten ditu, euskaraz
ICA_ETIKETAK = {
    "Muy bueno": "Oso ona",
    "Muy buena": "Oso ona",
    "Bueno": "Ona",
    "Buena": "Ona",
    "Regular": "Erregularra",
    "Malo": "Txarra",
    "Mala": "Txarra",
    "Muy malo": "Oso txarra",
    "Muy mala": "Oso txarra",
    "Extremadamente malo": "Muturreko txarra",
    "Extremadamente mala": "Muturreko txarra",
}

# Kategorien ordena logikoa (onenetik txarrenera)
ICA_ORDENA = ["Oso ona", "Ona", "Erregularra", "Txarra", "Oso txarra", "Muturreko txarra"]


def csv_irakurri(bidea):
    # Open Data Euskadiko CSVek ; bereizlea eta koma hamartarra erabiltzen dute
    df = pd.read_csv(bidea, sep=";", decimal=",", encoding="latin-1")

    # Ordua testu gisa dator "01:00" ... "24:00"
    # Orduaren zenbakia hartu eta datari gehitzen diogu
    # Horrela "24:00" hurrengo eguneko 00:00 bihurtzen da
    ordua = df["Hour  (GMT)"].str[:2].astype(int)
    eguna = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df["data"] = eguna + pd.to_timedelta(ordua, unit="h")

    # Jatorrizko testu-zutabeak kendu eta data indize gisa jarri
    df = df.drop(columns=["Date", "Hour  (GMT)"])
    df = df.set_index("data")
    # Fitxategiak berrienetik zaharrenera datoz: ordenatu egiten ditugu
    df = df.sort_index()
    return df


def ica_etiketa_garbitu(testua):
    # "Muy buena / Oso Ona" -> "Oso ona"
    # "Sin datos / Daturik gabe" -> None (nulu)
    if pd.isna(testua):
        return None
    gaztelaniaz = testua.split("/")[0].strip()
    if gaztelaniaz == "Sin datos":
        return None
    if gaztelaniaz in ICA_ETIKETAK:
        return ICA_ETIKETAK[gaztelaniaz]
    return None


def urtea_kargatu(orduko_bidea, indize_bidea):
    # Urte bat kargatzen du: orduko datuak + indize-fitxategiko ICA kategoriak

    # 1) Orduko datuak (jatorrizko neurketak)
    orduko = csv_irakurri(orduko_bidea)
    # Izenaren arabera hautatzen ditugu (2023an zutabeen ordena desberdina da)
    orduko = orduko[list(ZUTABE_IZENAK.keys())]
    orduko = orduko.rename(columns=ZUTABE_IZENAK)

    # 2) Indize-fitxategia: ICA zutabeak bakarrik
    indizea = csv_irakurri(indize_bidea)
    indizea = indizea[list(ICA_ZUTABEAK.keys())]
    indizea = indizea.rename(columns=ICA_ZUTABEAK)
    for zutabea in indizea.columns:
        indizea[zutabea] = indizea[zutabea].apply(ica_etiketa_garbitu)

    # 3) Bi taulak dataren arabera elkartu
    df = orduko.join(indizea, how="left")
    return df


def estazioa_kargatu(orduko_bidea, indize_bidea):
    # Edozein estazioren NO2, PM10, PM2.5 eta ICA bakarrik kargatzen ditu
    # (Barakaldo beste estazio batzuekin alderatzeko; ez dute denek gauza bera neurtzen)
    orduko = csv_irakurri(orduko_bidea)
    orduko = orduko[["NO2 (µg/m3)", "PM10 (µg/m3)", "PM2,5 (µg/m3)"]]
    orduko.columns = ["NO2", "PM10", "PM25"]
    indizea = csv_irakurri(indize_bidea)
    orduko["ICA_estazioa"] = indizea["ICA Estación"].apply(ica_etiketa_garbitu)
    return orduko


def datuak_kargatu(karpeta, urteak):
    # Hainbat urte kargatu eta taula bakarrean elkartzen ditu
    taulak = []
    for urtea in urteak:
        orduko_bidea = karpeta + "/BARAKALDO_" + str(urtea) + ".csv"
        indize_bidea = karpeta + "/BARAKALDO_INDIZEA_" + str(urtea) + ".csv"
        taula = urtea_kargatu(orduko_bidea, indize_bidea)
        # Fitxategiaren urtea gorde (entrenamendua eta proba bereizteko erabilgarria)
        taula["urtea"] = urtea
        print("Urtea", urtea, "->", len(taula), "errenkada")
        taulak.append(taula)

    df = pd.concat(taulak)
    df = df.sort_index()

    # Badaezpada, errepikatutako datak kendu
    errepikatuak = df.index.duplicated(keep="first")
    df = df[~errepikatuak]
    # Aurtengo urtea oraindik argitaratu gabeko orduekin amaitzen daberaz kendu egiten ditugu
    neurketak = list(ZUTABE_IZENAK.values())
    datudun_azken_ordua = df[neurketak].dropna(how="all").index.max()
    df = df.loc[:datudun_azken_ordua]
    return df
