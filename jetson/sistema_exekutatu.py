# Monitorizazio-sistema denbora errealean (simulatua) exekutatzen du Jetson Nanon
#
# 2026ko orduak banan-banan irakurtzen ditu, Barakaldoko estaziotik iritsiko balira bezala,
# eta ordu bakoitzean: NO2a iragarri, ICA sailkatu, egoera detektatu,
# sistema adituna aplikatu eta emaitza CSV batean gordetzen du.
#
# --pantaila aukerarekin, ordu bakoitzaren laburpena Waveshare plakako OLED pantailan ere
# erakusten du (SH1106, I2C 0x3C) eta gelako tenperatura eta hezetasuna BME280 sentsorearekin
# neurtzen ditu (0x76). Pantailaren RST hanka 18. pinera (BOARD) doa: behean badago pantailak
# ez du erantzuten, beraz programak goian jartzen du hasieran (--rst-pina).
#
# Adibideak:
# python3 sistema_exekutatu.py
# python3 sistema_exekutatu.py --hasiera 2026-03-03 --orduak 72 --tartea 0.5
# python3 sistema_exekutatu.py --motorra tflite --tartea 0 --orduak 9000
# python3 sistema_exekutatu.py --pantaila          # OLED pantailan eta sentsorearekin
# python3 sistema_exekutatu.py --bilatu            # I2C gailuak zerrendatu eta irten

import argparse
import csv
import os
import struct
import time

# Jetson Nanon pip bidez instalatutako numpy-ren OpenBLASak Illegal instruction errorea ematen du aldagai hau gabe. numpy inportatu AURRETIK jarri behar da.
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

import numpy as np
import sistema

KARPETA = os.path.dirname(os.path.abspath(__file__))

# Terminaleko koloreak (ondo ikusten dira demoaren bideoan)
KOLOREAK = {}
KOLOREAK["ALTUA"] = "\033[41m\033[97m"     # atzealde gorria
KOLOREAK["ERTAINA"] = "\033[43m\033[30m"   # atzealde horia
KOLOREAK["ABISUA"] = "\033[33m"            # letra horia
KOLOREAK["NORMALA"] = "\033[32m"           # letra berdea
KOLORE_AMAIERA = "\033[0m"


# OLED pantaila eta gelako sentsorea (--pantaila aukerarekin bakarrik)
PANTAILA_HELBIDEA = 0x3C
BME_HELBIDEA = 0x76
WAVESHARE_GAILUAK = {
    0x28: "TSL25911 argi-sentsorea",
    0x29: "TSL25911 argi-sentsorea",
    0x3C: "OLED pantaila (SH1106)",
    0x53: "LTR390 UV sentsorea",
    0x59: "SGP40 VOC sentsorea",
    0x68: "ICM20948 9-DOF sentsorea",
    0x76: "BME280 tenperatura eta hezetasuna",
}


def pantaila_resetetik_atera(pina):
    # OLEDaren RST pina goian jartzen du (reset pultsu labur baten ondoren). Gabe, pantaila
    # reset egoeran geratzen da eta ez du I2C bidez erantzuten.
    if pina <= 0:
        return
    try:
        import Jetson.GPIO as GPIO
    except ImportError:
        print("OHARRA: Jetson.GPIO ez dago; ezin da pantailaren RST pina goian jarri")
        return
    try:
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(pina, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.output(pina, GPIO.LOW)
        time.sleep(0.05)
        GPIO.output(pina, GPIO.HIGH)
        time.sleep(0.2)
    except Exception as errorea:
        print("OHARRA: ezin da", pina, ". pina erabili (" + str(errorea) + "). Probatu sudo-rekin edo gpio taldean sartu.")

def busa_ireki(zenbakia):
    # smbus2 hobesten dugu; bestela sistemako smbus (apt) erabiltzen dugu
    try:
        from smbus2 import SMBus
        return SMBus(zenbakia), "smbus2"
    except ImportError:
        pass
    try:
        from smbus import SMBus
        return SMBus(zenbakia), "smbus"
    except ImportError:
        return None, None


def gailuak_bilatu(busa):
    # i2cdetect-en antzera: helbide bakoitzean byte bat irakurtzen saiatzen gara
    aurkituak = []
    for helbidea in range(0x03, 0x78):
        try:
            busa.read_byte(helbidea)
            aurkituak.append(helbidea)
        except Exception:
            pass
    return aurkituak


def gailuaren_izena(helbidea):
    return WAVESHARE_GAILUAK.get(helbidea, "ezezaguna")


class Pantaila:
    ZABALERA = 128
    ALTUERA = 64

    def __init__(self, busa, helbidea):
        self.busa = busa
        self.helbidea = helbidea
        # SH1106ak 132 zutabe ditu eta irudia 2 zutabe mugituta hasten da
        self.desplazamendua = 2

        hasieratzea = [
            0xAE,        # pantaila itzali
            0xD5, 0x80,  # erlojua
            0xA8, 0x3F,  # 64 lerro
            0xD3, 0x00,  # desplazamendu bertikalik ez
            0x40,        # hasierako lerroa 0
            0x8D, 0x14,  # karga-ponpa piztu (SSD1306)
            0x20, 0x02,  # orrialde bidezko helbideratzea
            0xA1,        # zutabeak ezkerretik eskuinera
            0xC8,        # lerroak goitik behera
            0xDA, 0x12,
            0x81, 0xCF,  # kontrastea
            0xD9, 0xF1,
            0xDB, 0x40,
            0xA4,        # RAMeko edukia erakutsi
            0xA6,        # kolore normalak (ez alderantzikatuak)
            0xAF,        # pantaila piztu
        ]
        for agindua in hasieratzea:
            self.agindua(agindua)

    def agindua(self, balioa):
        self.busa.write_byte_data(self.helbidea, 0x00, balioa)

    def erakutsi(self, irudia):
        # PIL irudia (1 bit, 128x64) pantailara bidaltzen du, 8 lerroko orrialdeka
        pixelak = irudia.load()
        for orrialdea in range(8):
            self.agindua(0xB0 + orrialdea)
            self.agindua(0x00 + (self.desplazamendua & 0x0F))
            self.agindua(0x10 + (self.desplazamendua >> 4))

            byteak = []
            for x in range(self.ZABALERA):
                bytea = 0
                for bita in range(8):
                    y = orrialdea * 8 + bita
                    if pixelak[x, y]:
                        bytea = bytea | (1 << bita)
                byteak.append(bytea)

            # I2C bidez gehienez 32 byteko zatiak
            for hasiera in range(0, len(byteak), 16):
                self.busa.write_i2c_block_data(self.helbidea, 0x40, byteak[hasiera:hasiera + 16])

    def itzali(self):
        try:
            self.agindua(0xAE)
        except Exception:
            pass

def zeinuduna(balioa, bitak):
    if balioa >= (1 << (bitak - 1)):
        balioa = balioa - (1 << bitak)
    return balioa


class BME280:
    # Bosch BME280 (tenperatura, hezetasuna, presioa) eta BMP280 (hezetasunik gabe)

    def __init__(self, busa, helbidea, txipa):
        self.busa = busa
        self.helbidea = helbidea
        self.hezetasuna_du = (txipa == 0x60)
        if self.hezetasuna_du:
            self.izena = "BME280"
        else:
            self.izena = "BMP280"

        # Kalibrazio-koefizienteak (datu-orriko izenak)
        k = bytes(self.busa.read_i2c_block_data(helbidea, 0x88, 24))
        balioak = struct.unpack("<HhhHhhhhhhhh", k)
        self.T1 = balioak[0]
        self.T2 = balioak[1]
        self.T3 = balioak[2]
        self.P = list(balioak[3:12])  # P1..P9

        if self.hezetasuna_du:
            self.H1 = self.busa.read_byte_data(helbidea, 0xA1)
            h = self.busa.read_i2c_block_data(helbidea, 0xE1, 7)
            self.H2 = zeinuduna(h[1] * 256 + h[0], 16)
            self.H3 = h[2]
            self.H4 = zeinuduna((h[3] << 4) | (h[4] & 0x0F), 12)
            self.H5 = zeinuduna((h[5] << 4) | (h[4] >> 4), 12)
            self.H6 = zeinuduna(h[6], 8)
            self.busa.write_byte_data(helbidea, 0xF2, 0x01)   # hezetasuna x1

        self.busa.write_byte_data(helbidea, 0xF5, 0xA0)       # 1 s-ro neurtu
        self.busa.write_byte_data(helbidea, 0xF4, 0x27)       # tenperatura eta presioa x1, modu normala
        time.sleep(0.1)

    def irakurri(self):
        d = self.busa.read_i2c_block_data(self.helbidea, 0xF7, 8)
        adc_p = (d[0] << 12) | (d[1] << 4) | (d[2] >> 4)
        adc_t = (d[3] << 12) | (d[4] << 4) | (d[5] >> 4)
        adc_h = (d[6] << 8) | d[7]

        # Tenperatura (datu-orriko formula)
        var1 = (adc_t / 16384.0 - self.T1 / 1024.0) * self.T2
        var2 = (adc_t / 131072.0 - self.T1 / 8192.0) ** 2 * self.T3
        t_fine = var1 + var2
        tenperatura = t_fine / 5120.0

        # Presioa
        P1, P2, P3, P4, P5, P6, P7, P8, P9 = self.P
        var1 = t_fine / 2.0 - 64000.0
        var2 = var1 * var1 * P6 / 32768.0
        var2 = var2 + var1 * P5 * 2.0
        var2 = var2 / 4.0 + P4 * 65536.0
        var1 = (P3 * var1 * var1 / 524288.0 + P2 * var1) / 524288.0
        var1 = (1.0 + var1 / 32768.0) * P1
        if var1 == 0:
            presioa = float("nan")
        else:
            p = 1048576.0 - adc_p
            p = (p - var2 / 4096.0) * 6250.0 / var1
            var1 = P9 * p * p / 2147483648.0
            var2 = p * P8 / 32768.0
            presioa = (p + (var1 + var2 + P7) / 16.0) / 100.0   # hPa

        # Hezetasuna
        hezetasuna = float("nan")
        if self.hezetasuna_du:
            h = t_fine - 76800.0
            h = (adc_h - (self.H4 * 64.0 + self.H5 / 16384.0 * h)) * (
                self.H2 / 65536.0 * (1.0 + self.H6 / 67108864.0 * h * (1.0 + self.H3 / 67108864.0 * h)))
            h = h * (1.0 - self.H1 * h / 524288.0)
            hezetasuna = min(100.0, max(0.0, h))

        neurketa = {}
        neurketa["tenperatura"] = tenperatura
        neurketa["hezetasuna"] = hezetasuna
        neurketa["presioa"] = presioa
        return neurketa


def sentsorea_hasieratu(busa, helbideak):
    # Plakako BME280a (edo BMP280) hasieratzen du; ez badago, None
    if BME_HELBIDEA not in helbideak:
        return None
    try:
        txipa = busa.read_byte_data(BME_HELBIDEA, 0xD0)
        if txipa == 0x60 or txipa == 0x58:
            return BME280(busa, BME_HELBIDEA, txipa)
    except Exception as errorea:
        print("Ezin izan da sentsorea hasieratu:", errorea)
    return None


def testua_zabalera(marrazkia, testua, letra):
    # Pillow zaharrak (Jetson) textsize du; berriak textbbox
    if hasattr(marrazkia, "textbbox"):
        koadroa = marrazkia.textbbox((0, 0), testua, font=letra)
        return koadroa[2] - koadroa[0]
    return marrazkia.textsize(testua, font=letra)[0]


def zenbaki_laburra(balioa, hamartarrak=0):
    if balioa is None or balioa != balioa:
        return "-"
    return str(round(balioa, hamartarrak)) if hamartarrak > 0 else str(int(round(balioa)))


EGOERA_LABURRAK = {
    "Aire garbia haizearekin": "Aire garbia",
    "Gaueko bareta": "Gaueko bareta",
    "Trafikoko puntako ordua": "Trafiko-punta",
    "Partikula-episodioa": "Partikulak",
}


def irudia_sortu(emaitza, neurketa):
    from PIL import Image, ImageDraw, ImageFont

    irudia = Image.new("1", (Pantaila.ZABALERA, Pantaila.ALTUERA), 0)
    marrazkia = ImageDraw.Draw(irudia)
    letra = ImageFont.load_default()

    # 1. lerroa: data eta ordua
    marrazkia.text((0, 0), emaitza["data"][5:16], font=letra, fill=1)

    # 2. lerroa: NO2 unekoa -> iragarria
    no2 = "NO2 " + zenbaki_laburra(emaitza["NO2"]) + " -> " + zenbaki_laburra(emaitza["NO2_iragarria"])
    marrazkia.text((0, 10), no2, font=letra, fill=1)

    # 3. lerroa: ICA iragarria
    marrazkia.text((0, 20), "ICA: " + str(emaitza["ICA_iragarria"]), font=letra, fill=1)

    # 4. lerroa: ingurumen-egoera
    egoera = EGOERA_LABURRAK.get(emaitza["egoera"], str(emaitza["egoera"]))
    marrazkia.text((0, 30), egoera, font=letra, fill=1)

    # 5. lerroa: gelako sentsorea (badago)
    if neurketa:
        gela = "Gela " + zenbaki_laburra(neurketa.get("tenperatura"), 1) + "C " + zenbaki_laburra(neurketa.get("hezetasuna")) + "%"
        marrazkia.text((0, 40), gela, font=letra, fill=1)

    # Alerta-maila: beheko laukia, NORMALA ez denean alderantzikatua
    maila = emaitza["alerta_maila"]
    zabalera = testua_zabalera(marrazkia, maila, letra)
    x = Pantaila.ZABALERA - zabalera - 4
    if maila == "NORMALA":
        marrazkia.rectangle((x - 2, 53, Pantaila.ZABALERA - 1, 63), outline=1, fill=0)
        marrazkia.text((x, 53), maila, font=letra, fill=1)
    else:
        marrazkia.rectangle((x - 2, 53, Pantaila.ZABALERA - 1, 63), outline=1, fill=1)
        marrazkia.text((x, 53), maila, font=letra, fill=0)

    # Alertaren ziurtasuna ezkerrean, behean
    marrazkia.text((0, 53), "ZF " + zenbaki_laburra(emaitza["ziurtasuna"]) + "%", font=letra, fill=1)
    return irudia

def hardwarea_prestatu(args):
    # Pantaila eta sentsorea prestatzen ditu. Ez badaude, (None, None) eta terminalean jarraitzen du.
    pantaila_resetetik_atera(args.rst_pina)
    busa, liburutegia = busa_ireki(args.busa)
    if busa is None:
        print("Ez dago smbus2 edo smbus liburutegirik. Exekutatu: bash instalatu.sh")
        return None, None

    helbideak = gailuak_bilatu(busa)
    print("I2C", args.busa, "busa (" + liburutegia + "). Aurkitutako gailuak:")
    if len(helbideak) == 0:
        print("  (bat ere ez) -> egiaztatu plaka ondo sartuta dagoela")
    for helbidea in helbideak:
        print("  " + hex(helbidea) + "  " + gailuaren_izena(helbidea))

    pantaila = None
    if PANTAILA_HELBIDEA in helbideak:
        try:
            pantaila = Pantaila(busa, PANTAILA_HELBIDEA)
            print("Pantaila: SH1106", hex(PANTAILA_HELBIDEA))
        except Exception as errorea:
            print("Ezin izan da pantaila hasieratu:", errorea)
    else:
        print("Pantailak ez du erantzuten: terminalean bakarrik erakutsiko da.")

    sentsorea = sentsorea_hasieratu(busa, helbideak)
    if sentsorea is not None:
        print("Sentsorea:", sentsorea.izena)
    return pantaila, sentsorea


def argumentuak_irakurri():
    parser = argparse.ArgumentParser(description="Airearen kalitate-sistema denbora errealean (simulatua)")
    parser.add_argument("--datuak", default=os.path.join(KARPETA, "datuak", "simulazioa_2026.csv"))
    parser.add_argument("--ereduak", default=os.path.join(KARPETA, "ereduak"))
    parser.add_argument("--hasiera", default="2026-03-03", help="hasiera-data (UUUU-HH-EE)")
    parser.add_argument("--orduak", type=int, default=72, help="prozesatu beharreko ordu kopurua")
    parser.add_argument("--tartea", type=float, default=1.0, help="orduen arteko segundoak (0 = itxaron gabe)")
    parser.add_argument("--motorra", default="numpy", choices=["numpy", "tflite"])
    parser.add_argument("--tflite", default="mlp_erregresioa_NO2.tflite", help=".tflite fitxategia")
    parser.add_argument("--irteera", default=os.path.join(KARPETA, "emaitzak", "jetson_emaitzak.csv"))
    parser.add_argument("--isilik", action="store_true", help="ordu bakoitza pantailan ez erakutsi")
    parser.add_argument("--pantaila", action="store_true", help="OLED pantaila eta gelako sentsorea erabili")
    parser.add_argument("--bilatu", action="store_true", help="I2C gailuak zerrendatu eta irten")
    parser.add_argument("--busa", type=int, default=1, help="I2C busa (Jetson Nanoko 3. eta 5. pinak = 1)")
    parser.add_argument("--rst-pina", type=int, default=18, help="OLEDaren RST pina (BOARD zenbakia); 0 = ez ukitu")
    return parser.parse_args()


def zenbakia(balioa, hamartarrak=1):
    # Zenbaki bat testu bihurtzen du pantailan erakusteko (NaN bada, marratxo bat)
    if balioa != balioa:
        return "  -  "
    return str(round(balioa, hamartarrak))


def ordua_erakutsi(emaitza, alertak):
    # Ordu baten lerroa pantailan idazten du, alerta-maila koloretan
    maila = emaitza["alerta_maila"]
    lerroa = emaitza["data"]
    lerroa = lerroa + " | NO2 " + zenbakia(emaitza["NO2"], 0) + " -> " + zenbakia(emaitza["NO2_iragarria"])
    lerroa = lerroa + " | ICA aurr.: " + emaitza["ICA_iragarria"].ljust(11)
    lerroa = lerroa + " | PM10 24h " + zenbakia(emaitza["PM10_batez_24h"])
    lerroa = lerroa + " | haizea " + zenbakia(emaitza["haize_abiadura"], 2)
    lerroa = lerroa + " | " + emaitza["egoera"].ljust(23)
    lerroa = lerroa + " | ziurtasuna " + (zenbakia(emaitza["ziurtasuna"], 1) + "%").rjust(6)
    lerroa = lerroa + " | " + KOLOREAK[maila] + " " + maila + " " + KOLORE_AMAIERA
    lerroa = lerroa + " | " + str(round(emaitza["guztira_ms"], 2)) + " ms"
    print(lerroa)

    for alerta in alertak:
        print("      [" + alerta["id"] + ", ZF " + str(alerta["zf"]) + "] " + alerta["mezua"] + ". " + alerta["gomendioa"])

def laburpena_erakutsi(emaitzak, motorra, irteera):
    if len(emaitzak) == 0:
        return
    guztira = []
    mlp = []
    for e in emaitzak:
        guztira.append(e["guztira_ms"])
        mlp.append(e["mlp_ms"])
    guztira = np.array(guztira)
    mlp = np.array(mlp)

    print()
    print("=" * 70)
    print("LABURPENA")
    print("Prozesatutako orduak:", len(emaitzak))
    print("Denbora orduko (sistema osoa): batez bestekoa", round(guztira.mean(), 3), "ms",
          "| p95", round(np.percentile(guztira, 95), 3), "ms",
          "| maximoa", round(guztira.max(), 3), "ms")
    print("MLParen denbora (" + motorra + "): batez bestekoa", round(mlp.mean(), 3), "ms")

    # NO2aren errorea (benetako balioa dagoen orduetan)
    erroreak = []
    for e in emaitzak:
        benetakoa = e["NO2_benetako_hurrengoa"]
        if benetakoa == benetakoa:
            erroreak.append(abs(benetakoa - e["NO2_iragarria"]))
    if len(erroreak) > 0:
        print("NO2aren MAE ordu hauetan:", round(np.mean(erroreak), 2), "µg/m³")

    # ICA kategoriaren asmatzeak
    asmatzeak = 0
    orduak = 0
    for e in emaitzak:
        if e["ICA_benetako_hurrengoa"] != "":
            orduak = orduak + 1
            if e["ICA_iragarria"] == e["ICA_benetako_hurrengoa"]:
                asmatzeak = asmatzeak + 1
    if orduak > 0:
        print("ICA kategoriaren asmatzeak:", round(100 * asmatzeak / orduak, 1), "%")

    # Maila bakoitzeko alerta kopurua
    for maila in ["ALTUA", "ERTAINA", "ABISUA"]:
        kopurua = 0
        for e in emaitzak:
            if e["alerta_maila"] == maila:
                kopurua = kopurua + 1
        print(maila + " alerta duten orduak:", kopurua)

    print("Emaitzak hemen gordeta:", irteera)


def main():
    args = argumentuak_irakurri()

    if args.bilatu:
        pantaila_resetetik_atera(args.rst_pina)
        busa, liburutegia = busa_ireki(args.busa)
        if busa is None:
            print("Ez dago smbus2 edo smbus liburutegirik. Exekutatu: bash instalatu.sh")
            return
        print("I2C", args.busa, "busa (" + liburutegia + "). Aurkitutako gailuak:")
        for helbidea in gailuak_bilatu(busa):
            print("  " + hex(helbidea) + "  " + gailuaren_izena(helbidea))
        return

    pantaila = None
    sentsorea = None
    if args.pantaila:
        pantaila, sentsorea = hardwarea_prestatu(args)

    print("Ereduak kargatzen...")
    sis = sistema.sistema_kargatu(args.ereduak, motorra=args.motorra, tflite_fitxategia=args.tflite)
    print("MLParen motorra:", sis["motorra"])

    # Datuak irakurri eta hasiera-datatik aurrerako orduak hartu
    datu_guztiak = sistema.datuak_irakurri(args.datuak)
    datuak = []
    for errenkada in datu_guztiak:
        if errenkada["data"] >= args.hasiera and len(datuak) < args.orduak:
            datuak.append(errenkada)
    print("Prozesatu beharreko orduak:", len(datuak), "noiztik:", datuak[0]["data"])
    print()

    # Irteerako karpeta sortu behar bada
    irteera_karpeta = os.path.dirname(args.irteera)
    if irteera_karpeta != "" and not os.path.exists(irteera_karpeta):
        os.makedirs(irteera_karpeta)

    fitxategia = open(args.irteera, "w", newline="", encoding="utf-8")
    idazlea = None
    emaitzak = []

    try:
        for errenkada in datuak:
            emaitza, alertak = sistema.ordua_prozesatu(errenkada, sis)

            # Gelako sentsorea (--pantaila aukerarekin)
            neurketa = {}
            if sentsorea is not None:
                try:
                    neurketa = sentsorea.irakurri()
                except Exception as errorea:
                    print("Sentsorearen irakurketak huts egin du:", errorea)
            if args.pantaila:
                emaitza["gela_tenperatura"] = neurketa.get("tenperatura", "")
                emaitza["gela_hezetasuna"] = neurketa.get("hezetasuna", "")
                emaitza["gela_presioa"] = neurketa.get("presioa", "")
            emaitzak.append(emaitza)

            # Ordu bakoitza CSVan gordetzen dugu (aginte-panelak bitartean irakur dezake)
            if idazlea is None:
                idazlea = csv.DictWriter(fitxategia, fieldnames=list(emaitza.keys()))
                idazlea.writeheader()
            idazlea.writerow(emaitza)
            fitxategia.flush()

            if not args.isilik:
                ordua_erakutsi(emaitza, alertak)
                if "tenperatura" in neurketa:
                    print("      gela: " + zenbaki_laburra(neurketa["tenperatura"], 1) + " C, "
                          + zenbaki_laburra(neurketa.get("hezetasuna")) + " %")

            if pantaila is not None:
                try:
                    pantaila.erakutsi(irudia_sortu(emaitza, neurketa))
                except Exception as errorea:
                    print("Pantailara idazteak huts egin du:", errorea)

            if args.tartea > 0:
                time.sleep(args.tartea)
    except KeyboardInterrupt:
        print("\nGeldituta.")
    finally:
        fitxategia.close()
        if pantaila is not None:
            pantaila.itzali()

    laburpena_erakutsi(emaitzak, sis["motorra"], args.irteera)


if __name__ == "__main__":
    main()
