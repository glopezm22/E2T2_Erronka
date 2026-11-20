# Sistemaren zati bakoitzaren inferentzia-denborak neurtzen ditu
# Jetson Nanon eta PCan exekutatu alderatzeko::
# python3 denborak_neurtu.py
# python3 denborak_neurtu.py --errepikapenak 2000

import argparse
import csv
import os
import platform
import socket
import sys
import time

#Jetosn Nannon pip bidez instalatutako numpy-ren OpenBLASak Illegal instruction" errorea ematen du aldagai hau gabe. numpy inportatu aurretik jarri behar da.
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

import numpy as np

import sistema
KARPETA = os.path.dirname(os.path.abspath(__file__))
EREDUEN_KARPETA = os.path.join(KARPETA, "ereduak")

def neurtu(zatia, errenkada, x, sis, errepikapenak):
    # "zatia"k esaten du sistemaren zein zati neurtu behar den.
    # Hainbat aldiz exekutatzen da eta denborak milisegundotan itzultzen dira.
    denborak = []
    for i in range(20 + errepikapenak):
        hasiera = time.perf_counter()
        if zatia == "aurreprozesamendua":
            sistema.aurreprozesatu(errenkada, sis)
        elif zatia == "mlp":
            sistema.no2_iragarri(x, sis)
        elif zatia == "zuhaitza":
            sistema.ica_sailkatu(x, sis)
        elif zatia == "kmeans":
            sistema.ingurumen_egoera(errenkada, sis)
        elif zatia == "osoa":
            sistema.ordua_prozesatu(errenkada, sis)
        amaiera = time.perf_counter()

        # Lehen 20ak "berotzeko" dira eta ez ditugu kontuan hartzen
        if i >= 20:
            denborak.append((amaiera - hasiera) * 1000)

    denborak = np.array(denborak)
    batez_bestekoa = round(denborak.mean(), 4)
    p95 = round(np.percentile(denborak, 95), 4)
    maximoa = round(denborak.max(), 4)
    return [batez_bestekoa, p95, maximoa]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--errepikapenak", type=int, default=1000)
    args = parser.parse_args()

    print("Gailua:", socket.gethostname(), "|", platform.machine(), "|", platform.platform())
    print("Python", sys.version.split()[0], "| numpy", np.__version__)
    print()

    datuak = sistema.datuak_irakurri(os.path.join(KARPETA, "datuak", "simulazioa_2026.csv"))
    errenkada = datuak[100]
    taula = []

    # 1) Numpy motorra: zati guztiak
    sis = sistema.sistema_kargatu(EREDUEN_KARPETA, motorra="numpy")
    x = sistema.aurreprozesatu(errenkada, sis)

    taula.append(["Aurreprozesamendua"] + neurtu("aurreprozesamendua", errenkada, x, sis, args.errepikapenak))
    taula.append(["NO2 erregresioko MLPa (numpy)"] + neurtu("mlp", errenkada, x, sis, args.errepikapenak))
    taula.append(["ICA erabaki-zuhaitza"] + neurtu("zuhaitza", errenkada, x, sis, args.errepikapenak))
    taula.append(["K-means egoera"] + neurtu("kmeans", errenkada, x, sis, args.errepikapenak))
    taula.append(["Sistema osoa, ordu 1 (numpy)"] + neurtu("osoa", errenkada, x, sis, args.errepikapenak))

    # 2) TensorFlow Lite motorra, instalatuta badago: MLPa eta sistema osoa
    tflite_dago = sistema.tflite_interpretatzailea_sortu(os.path.join(EREDUEN_KARPETA, "mlp_erregresioa_NO2.tflite"))
    if tflite_dago is None:
        print("TensorFlow Lite ez dago instalatuta: numpy bakarrik neurtzen da")
    else:
        sis_tflite = sistema.sistema_kargatu(EREDUEN_KARPETA, motorra="tflite")
        taula.append(["NO2 erregresioko MLPa (tflite)"] + neurtu("mlp", errenkada, x, sis_tflite, args.errepikapenak))
        taula.append(["Sistema osoa, ordu 1 (tflite)"] + neurtu("osoa", errenkada, x, sis_tflite, args.errepikapenak))
        if os.path.exists(os.path.join(EREDUEN_KARPETA, "mlp_erregresioa_NO2_int8.tflite")):
            # tflite_runtime bertsio zaharrek ez dituzte eredu kuantizatu guztiak irakurtzen; orduan saltatu egiten da
            try:
                sis_int8 = sistema.sistema_kargatu(EREDUEN_KARPETA, motorra="tflite", tflite_fitxategia="mlp_erregresioa_NO2_int8.tflite")
                taula.append(["NO2 erregresioko MLPa (tflite int8)"] + neurtu("mlp", errenkada, x, sis_int8, args.errepikapenak))
            except ValueError as errorea:
                print("OHARRA: int8 eredua ezin da kargatu tflite_runtime honekin:", str(errorea).splitlines()[0])

    # 3) Urte oso bat, itxaron gabe
    hasiera = time.perf_counter()
    for e in datuak:
        sistema.ordua_prozesatu(e, sis)
    segundoak = time.perf_counter() - hasiera
    izena = "Simulazioko datu guztiak (" + str(len(datuak)) + " ordu)"
    taula.append([izena, round(segundoak * 1000, 1), "", ""])

    # Taula pantailan erakutsi
    print("{:<48} {:>16} {:>12} {:>12}".format("Osagaia", "batez best. (ms)", "p95 (ms)", "max (ms)"))
    print("-" * 92)
    for errenkada_taula in taula:
        print("{:<48} {:>16} {:>12} {:>12}".format(errenkada_taula[0], errenkada_taula[1], errenkada_taula[2], errenkada_taula[3]))
    print()
    print("Segundoko prozesatutako orduak:", round(len(datuak) / segundoak))

    # Emaitzak CSV batean gorde
    emaitzen_karpeta = os.path.join(KARPETA, "emaitzak")
    if not os.path.exists(emaitzen_karpeta):
        os.makedirs(emaitzen_karpeta)
    bidea = os.path.join(emaitzen_karpeta, "denborak_" + socket.gethostname() + ".csv")

    fitxategia = open(bidea, "w", newline="", encoding="utf-8")
    idazlea = csv.writer(fitxategia)
    idazlea.writerow(["gailua", "osagaia", "batez_bestekoa_ms", "p95_ms", "max_ms"])
    for errenkada_taula in taula:
        idazlea.writerow([socket.gethostname()] + errenkada_taula)
    fitxategia.close()
    print("Hemen gordeta:", bidea)


if __name__ == "__main__":
    main()
