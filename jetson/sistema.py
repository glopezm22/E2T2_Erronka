# Airearen kalitatea monitorizatzeko sistema (Barakaldo)
# Jetson Nanon exekutatzeko pentsatua (Python 3.6), numpy BAKARRIK erabiliz.
#
# Edukia:
#   1) Aurreprozesamendua (PIAko Pipeline-aren berdina)
#   2) Erregresioko MLPa -> hurrengo orduko NO2a (numpy edo TensorFlow Lite)
#   3) Erabaki-zuhaitza -> hurrengo orduko ICA kategoria
#   4) K-means -> ingurumen-egoera
#   5) Arauetan oinarritutako sistema adituna -> alertak, ziurtasun-faktoreekin
#
# Ereduen parametroak PCtik esportatzen dira 06 eta 07 koadernoekin.

import csv
import json
import math
import os
import time

# Jetson Nano: pip bidez instalatutako numpy-ren OpenBLASak Illegal instruction errorea ematen du aldagai hau gabe.  numpy inportatu AURRETIK jarri behar da.
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

import numpy as np

# Ereduak kargatzea
def sistema_kargatu(ereduen_karpeta, motorra="numpy", tflite_fitxategia="mlp_erregresioa_NO2.tflite"):
    # parametroak.json irakurri (aurreprozesamendua, zuhaitza eta K-means)
    json_bidea = os.path.join(ereduen_karpeta, "parametroak.json")
    fitxategia = open(json_bidea, encoding="utf-8")
    parametroak = json.load(fitxategia)
    fitxategia.close()

    # MLParen pisuak irakurri: W0, b0, W1, b1, ...
    pisuak = np.load(os.path.join(ereduen_karpeta, "mlp_erregresioa_NO2_pisuak.npz"))
    geruza_kopurua = len(pisuak.files) // 2
    geruzak = []
    for i in range(geruza_kopurua):
        W = pisuak["W" + str(i)].astype(np.float32)
        b = pisuak["b" + str(i)].astype(np.float32)
        geruzak.append((W, b))

    # Sistema hiztegi batean gordetzen dugu
    sistema = {}
    sistema["parametroak"] = parametroak
    sistema["mlp_geruzak"] = geruzak
    sistema["motorra"] = "numpy"
    sistema["interpretatzailea"] = None

    # TensorFlow Lite eskatzen bada eta instalatuta badago, hura erabiltzen dugu
    if motorra == "tflite":
        tflite_bidea = os.path.join(ereduen_karpeta, tflite_fitxategia)
        interpretatzailea = tflite_interpretatzailea_sortu(tflite_bidea)
        if interpretatzailea is None:
            print("OHARRA: TensorFlow Lite ez dago instalatuta, numpy erabiltzen da")
        else:
            sistema["motorra"] = "tflite"
            sistema["interpretatzailea"] = interpretatzailea

    return sistema


def tflite_interpretatzailea_sortu(bidea):
    # Bi aukera probatzen ditugu, ordenan: tflite_runtime (Jetson) eta TensorFlow osoa (PCa)
    Interpreter = None
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        pass

    if Interpreter is None:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        except ImportError:
            return None

    interpretatzailea = Interpreter(model_path=bidea)
    interpretatzailea.allocate_tensors()
    return interpretatzailea


# Zenbakiak ez diren zutabeak
TESTU_ZUTABEAK = ["data", "ICA_estazioa", "ICA_hurrengoa"]
# Datuak irakurtzea
def datuak_irakurri(csv_bidea):
    # Simulazioko CSVa irakurtzen du. Ordu bakoitza hiztegi bat da.
    errenkadak = []
    fitxategia = open(csv_bidea, encoding="utf-8")
    irakurlea = csv.DictReader(fitxategia)

    for errenkada in irakurlea:
        berria = {}
        for zutabea in errenkada:
            balioa = errenkada[zutabea]
            if zutabea in TESTU_ZUTABEAK:
                berria[zutabea] = balioa
            elif balioa == "":
                berria[zutabea] = float("nan")   # datu falta
            else:
                berria[zutabea] = float(balioa)
        errenkadak.append(berria)

    fitxategia.close()
    return errenkadak


def balioa_hartu(errenkada, aldagaia):
    # Aldagaiaren balioa itzultzen du, edo NaN ez badago
    if aldagaia in errenkada:
        return errenkada[aldagaia]
    return float("nan")


# 1) Aurreprozesamendua
def aurreprozesatu(errenkada, sistema):
    # Ordu bat ereduek espero duten zenbaki-bektore bihurtzen du (65 zutabe)
    p = sistema["parametroak"]["aurreprozesamendua"]
    balioak = []

    # Numerikoak: nuluak medianarekin bete eta estandarizatu
    for i in range(len(p["numerikoak"])):
        balioa = balioa_hartu(errenkada, p["numerikoak"][i])
        if math.isnan(balioa):
            balioa = p["medianak"][i]
        estandarizatua = (balioa - p["batez_bestekoak"][i]) / p["eskalak"][i]
        balioak.append(estandarizatua)

    # Kategorikoak (ordua, asteko eguna, hilabetea eta haize-sektorea): one-hot
    for i in range(len(p["kategorikoak"])):
        balioa = balioa_hartu(errenkada, p["kategorikoak"][i])
        if math.isnan(balioa):
            balioa = p["maizenak"][i]
        for kategoria in p["kategoriak"][i]:
            if balioa == kategoria:
                balioak.append(1.0)
            else:
                balioak.append(0.0)

    return np.array(balioak, dtype=np.float32)


# 2) Erregresioko MLPa: hurrengo orduko NO2a
def no2_iragarri(x, sistema):
    # TensorFlow Lite-rekin
    if sistema["motorra"] == "tflite":
        interpretatzailea = sistema["interpretatzailea"]
        sarrera = interpretatzailea.get_input_details()[0]
        irteera = interpretatzailea.get_output_details()[0]
        x_tflite = x.reshape(1, -1).astype(np.float32)
        interpretatzailea.set_tensor(sarrera["index"], x_tflite)
        interpretatzailea.invoke()
        emaitza = interpretatzailea.get_tensor(irteera["index"])
        return float(emaitza[0, 0])

    # Numpy-rekin: geruzaz geruza, pisuekin biderkatu eta alborapena gehitu
    geruzak = sistema["mlp_geruzak"]
    azken_geruza = len(geruzak) - 1
    for i in range(len(geruzak)):
        W = geruzak[i][0]
        b = geruzak[i][1]
        x = np.dot(x, W) + b
        # ReLU geruza ezkutuetan bakarrik (ez irteeran)
        if i < azken_geruza:
            x = np.maximum(x, 0)
    return float(x[0])


# 3) Erabaki-zuhaitza: hurrengo orduko ICA kategoria
def ica_sailkatu(x, sistema):
    zuhaitza = sistema["parametroak"]["zuhaitza"]
    x32 = x.astype(np.float32)   # scikit-learn-ek float32n alderatzen du

    # Erroko nodotik hasi eta hosto batera iritsi arte jaitsi (hostoetan ezkerra = -1)
    nodoa = 0
    while zuhaitza["ezkerra"][nodoa] != -1:
        aldagaia = zuhaitza["aldagaia"][nodoa]
        atalasea = zuhaitza["atalasea"][nodoa]
        if x32[aldagaia] <= atalasea:
            nodoa = zuhaitza["ezkerra"][nodoa]
        else:
            nodoa = zuhaitza["eskuina"][nodoa]

    return zuhaitza["klasea"][nodoa]


def zuhaitzaren_araua(x, sistema):
    # Zuhaitzak jarraitutako bidea testuz itzultzen du (iragarpena azaltzeko)
    zuhaitza = sistema["parametroak"]["zuhaitza"]
    p = sistema["parametroak"]["aurreprozesamendua"]
    x32 = x.astype(np.float32)
    numeriko_kopurua = len(p["numerikoak"])

    urratsak = []
    nodoa = 0
    while zuhaitza["ezkerra"][nodoa] != -1:
        i = zuhaitza["aldagaia"][nodoa]
        atalasea = zuhaitza["atalasea"][nodoa]

        # Atalasea jatorrizko unitateetara pasatzen dugu (estandarizazioa desegin)
        if i < numeriko_kopurua:
            izena = p["numerikoak"][i]
            benetako_atalasea = atalasea * p["eskalak"][i] + p["batez_bestekoak"][i]
        else:
            izena = zuhaitza["zutabe_izenak"][i]
            benetako_atalasea = atalasea

        if x32[i] <= atalasea:
            urratsak.append(izena + " <= " + str(round(benetako_atalasea, 1)))
            nodoa = zuhaitza["ezkerra"][nodoa]
        else:
            urratsak.append(izena + " > " + str(round(benetako_atalasea, 1)))
            nodoa = zuhaitza["eskuina"][nodoa]

    return " ETA ".join(urratsak)


# 4) K-means: ingurumen-egoera
def ingurumen_egoera(errenkada, sistema):
    k = sistema["parametroak"]["kmeans"]

    # Ordu hau estandarizatu (K-meansen prestaketa bera)
    balioak = []
    for i in range(len(k["aldagaiak"])):
        balioa = balioa_hartu(errenkada, k["aldagaiak"][i])
        if math.isnan(balioa):
            balioa = k["medianak"][i]
        balioak.append((balioa - k["batez_bestekoak"][i]) / k["eskalak"][i])
    balioak = np.array(balioak)

    # Zentroide hurbilena bilatu
    hurbilena = 0
    distantzia_txikiena = None
    for c in range(len(k["zentroideak"])):
        zentroidea = np.array(k["zentroideak"][c])
        distantzia = np.sum((zentroidea - balioak) ** 2)
        if distantzia_txikiena is None or distantzia < distantzia_txikiena:
            distantzia_txikiena = distantzia
            hurbilena = c

    return k["izenak"][str(hurbilena)]


# 5) Arauetan oinarritutako sistema adituna
# Atalaseak (aldatu egin daitezke haien eragina aztertzeko)
ATALASEAK = {}
ATALASEAK["PM10_txarra"] = 50 # PM10en ICA "Txarra" muga (24 orduko batez bestekoa)
ATALASEAK["PM25_txarra"] = 25  # PM2.5en ICA "Txarra" muga (24 orduko batez bestekoa)
ATALASEAK["PM10_erregularra"] = 35 # PM10en ICA "Erregularra" muga
ATALASEAK["NO2_altua"] = 40  # NO2aren ICA "Oso ona" muga (eta EBren urteko muga)
ATALASEAK["haize_bareta"] = 1.0 # m/s: bareta, trafikoaren NO2a metatu egiten da
ATALASEAK["haize_sakabanatzea"] = 2.0 # m/s: horren azpitik partikulek igotzen jarraitzen dute
ATALASEAK["tenperatura_hotza"] = 10 # ºC: goiz hotzetan alderantzizko termikoa errazagoa da


def arau_1(e):
    # Kalitate txarra: ereduak dio edo muga ofiziala gainditzen da
    if e["ICA_iragarria"] == "Txarra":
        return True
    if e["PM10_batez_24h"] > ATALASEAK["PM10_txarra"]:
        return True
    if e["PM25_batez_24h"] > ATALASEAK["PM25_txarra"]:
        return True
    return False


def arau_2(e):
    # Partikula altuak eta sakabanatzeko haize gutxi
    partikula_altuak = (e["ICA_iragarria"] == "Erregularra"
                        or e["PM10_batez_24h"] > ATALASEAK["PM10_erregularra"])
    haize_gutxi = e["haize_abiadura"] < ATALASEAK["haize_sakabanatzea"]
    return partikula_altuak and haize_gutxi


def arau_3(e):
    # NO2 altua iragarrita haize-baretarekin (enuntziatuaren araua)
    no2_altua = e["NO2_iragarria"] > ATALASEAK["NO2_altua"]
    bareta = e["haize_abiadura"] < ATALASEAK["haize_bareta"]
    return no2_altua and bareta


def arau_4(e):
    # K-meansek detektatutako partikula-episodioa eta kalitate Erregularra edo Txarra
    episodioa = e["egoera"] == "Partikula-episodioa"
    kalitate_txarra = e["ICA_iragarria"] == "Erregularra" or e["ICA_iragarria"] == "Txarra"
    return episodioa and kalitate_txarra


def arau_5(e):
    # Puntako ordua goiz hotz eta haizerik gabean: alderantzizko termiko posiblea
    puntako_ordua = e["egoera"] == "Trafikoko puntako ordua"
    hotza = e["tenperatura"] < ATALASEAK["tenperatura_hotza"]
    bareta = e["haize_abiadura"] < ATALASEAK["haize_bareta"]
    return puntako_ordua and hotza and bareta


#Ezagutza-basea: arauen zerrenda
# Arau bakoitzak ziurtasun-faktore bat du, MYCIN sisteman bezala: araua betetzen denean hurrengo orduan kutsadura-arazoa egongo dela (ICA Erregularra edo Txarra, edo NO2 > 40)
# zenbateko konfiantza dugun, 0tik 1era. 2023-2025eko datuekin kalkulatu dira (08koadernoa).
ARAUAK = []
ARAUAK.append({"id": "R1", "maila": "ALTUA", "zf": 0.97, "baldintza": arau_1,
               "mezua": "Airearen kalitate TXARRA iragarrita",
               "gomendioa": "Kanpoko jarduera fisikoa saihestu. Ikastetxeei, osasun-zentroei eta pertsona sentikorrei abisatu."})
ARAUAK.append({"id": "R2", "maila": "ERTAINA", "zf": 0.92, "baldintza": arau_2,
               "mezua": "Partikula altuak eta sakabanatzeko haize gutxi",
               "gomendioa": "Litekeena da okertzea. Pertsona sentikorrak: kalean ariketa bizia murriztu."})
ARAUAK.append({"id": "R3", "maila": "ERTAINA", "zf": 0.81, "baldintza": arau_3,
               "mezua": "NO2 altua iragarrita haize-baretarekin",
               "gomendioa": "Garraio publikoa gomendatu eta eremuko trafikoa murriztu."})
ARAUAK.append({"id": "R4", "maila": "ABISUA", "zf": 0.97, "baldintza": arau_4,
               "mezua": "Partikula-episodioa martxan",
               "gomendioa": "Hurrengo orduetan PM10 eta PM2.5en bilakaera zaindu."})
ARAUAK.append({"id": "R5", "maila": "ABISUA", "zf": 0.48, "baldintza": arau_5,
               "mezua": "Alderantzizko termiko posiblea puntako orduan",
               "gomendioa": "Trafikoaren kutsadura metatu egiten da: NO2a zaindu goizean zehar."})


def arauak_aplikatu(egitateak):
    # Inferentzia-motorra: arau guztiak egiaztatu eta betetzen direnak itzultzen ditu,
    # larrienetik arinenera ordenatuta
    aktibatuak = []
    for maila in ["ALTUA", "ERTAINA", "ABISUA"]:
        for araua in ARAUAK:
            if araua["maila"] == maila and araua["baldintza"](egitateak):
                alerta = {}
                alerta["id"] = araua["id"]
                alerta["maila"] = araua["maila"]
                alerta["mezua"] = araua["mezua"]
                alerta["gomendioa"] = araua["gomendioa"]
                alerta["zf"] = araua["zf"]
                aktibatuak.append(alerta)
    return aktibatuak


def ziurtasuna_konbinatu(aktibatuak):
    # Aktibatutako arauen ziurtasun-faktoreak konbinatzen ditu (MYCIN):
    # ZF = ZF1 + ZF2 * (1 - ZF1). Arau bakoitzak konfiantza gehitzen du, baina inoiz ez da 1era iristen.
    zf = 0.0
    for alerta in aktibatuak:
        zf = zf + alerta["zf"] * (1 - zf)
    return zf


def alerta_maila(aktibatuak):
    # Azken maila arau larrienarena da (lehenengoa, ordenatuta daudelako)
    if len(aktibatuak) == 0:
        return "NORMALA"
    return aktibatuak[0]["maila"]


# Dena batera: ordu bat prozesatu
def ordua_prozesatu(errenkada, sistema):
    denborak = {}

    # 1) Aurreprozesamendua
    hasiera = time.perf_counter()
    x = aurreprozesatu(errenkada, sistema)
    denborak["aurreprozesamendua_ms"] = (time.perf_counter() - hasiera) * 1000

    # 2) MLPa
    hasiera = time.perf_counter()
    no2_iragarria = no2_iragarri(x, sistema)
    denborak["mlp_ms"] = (time.perf_counter() - hasiera) * 1000

    # 3) Zuhaitza
    hasiera = time.perf_counter()
    ica_iragarria = ica_sailkatu(x, sistema)
    denborak["zuhaitza_ms"] = (time.perf_counter() - hasiera) * 1000

    # 4) K-means
    hasiera = time.perf_counter()
    egoera = ingurumen_egoera(errenkada, sistema)
    denborak["kmeans_ms"] = (time.perf_counter() - hasiera) * 1000

    # 5) Sistema adituna
    hasiera = time.perf_counter()
    egitateak = {}
    egitateak["NO2_iragarria"] = no2_iragarria
    egitateak["ICA_iragarria"] = ica_iragarria
    egitateak["PM10_batez_24h"] = errenkada["PM10_batez_24h"]
    egitateak["PM25_batez_24h"] = errenkada["PM25_batez_24h"]
    egitateak["haize_abiadura"] = errenkada["haize_abiadura"]
    egitateak["tenperatura"] = errenkada["tenperatura"]
    egitateak["egoera"] = egoera

    alertak = arauak_aplikatu(egitateak)
    ziurtasuna = ziurtasuna_konbinatu(alertak)
    denborak["adituna_ms"] = (time.perf_counter() - hasiera) * 1000

    denborak["guztira_ms"] = (denborak["aurreprozesamendua_ms"] + denborak["mlp_ms"] + denborak["zuhaitza_ms"]
                              + denborak["kmeans_ms"] + denborak["adituna_ms"])

    # Aktibatutako arauen idak eta mezuak testu bakarrean
    idak = []
    mezuak = []
    for alerta in alertak:
        idak.append(alerta["id"])
        mezuak.append(alerta["mezua"])

    # Orduaren emaitza
    emaitza = {}
    emaitza["data"] = errenkada.get("data", "")
    emaitza["NO2"] = errenkada["NO2"]
    emaitza["NO2_iragarria"] = round(no2_iragarria, 2)
    emaitza["NO2_benetako_hurrengoa"] = errenkada.get("NO2_hurrengoa", float("nan"))
    emaitza["ICA_unekoa"] = errenkada.get("ICA_estazioa", "")
    emaitza["ICA_iragarria"] = ica_iragarria
    emaitza["ICA_benetako_hurrengoa"] = errenkada.get("ICA_hurrengoa", "")
    emaitza["PM10_batez_24h"] = errenkada["PM10_batez_24h"]
    emaitza["PM25_batez_24h"] = errenkada["PM25_batez_24h"]
    emaitza["haize_abiadura"] = errenkada["haize_abiadura"]
    emaitza["tenperatura"] = errenkada["tenperatura"]
    emaitza["egoera"] = egoera
    # Alertaren ziurtasuna (%). Konbinazioak ez du inoiz 1 ematen, baina biribiltzean 100 ager daiteke;
    # horregatik 99,9an mugatzen dugu
    emaitza["ziurtasuna"] = round(min(ziurtasuna * 100, 99.9), 1)
    emaitza["alerta_maila"] = alerta_maila(alertak)
    emaitza["arauak"] = " ".join(idak)
    emaitza["mezuak"] = " | ".join(mezuak)

    # Denborak ere gehitzen ditugu
    for gakoa in denborak:
        emaitza[gakoa] = denborak[gakoa]

    return emaitza, alertak
