# Jetson Nano

Karpeta hau osorik kopiatu behar da Jetsonera. Python 3.6 eta numpy-rekin funtzionatzen du (JetPack 4.6n badatoz); ez da scikit-learn ezta TensorFlow behar.

```
jetson/
├── sistema.py              aurreprozesamendua, MLP, zuhaitza, K-means eta arauak (ziurtasun-faktoreekin)
├── sistema_exekutatu.py    denbora errealeko demoa (2026ko datuekin), aukeran OLED pantailan
├── denborak_neurtu.py      inferentzia-denborak
├── instalatu.sh            Jetsona prestatzeko scripta
├── ereduak/                parametroak.json, MLParen pisuak (.npz) eta .tflite
├── datuak/                 simulazioa_2026.csv
└── emaitzak/               hemen gordetzen dira emaitzak
```

`ereduak/` eta `datuak/` 06 eta 07 koadernoek sortzen dituzte. Ereduak berriz entrenatzen badira, karpeta berriro kopiatu behar da.

## Lehen aldiz

PCko PowerShell-etik, proiektuaren karpetan (`erabiltzailea@IPa` Jetsonarena da; bertsio zaharra badago, lehenik ezabatu `~/airearen_kalitatea`):

```powershell
scp -r jetson erabiltzailea@IPa:~/airearen_kalitatea
```

Jetsonean, behin bakarrik:

```bash
cd ~/airearen_kalitatea
bash instalatu.sh
sudo reboot
```

`instalatu.sh`-k numpy, Pillow, smbus2 eta i2c-tools instalatzen ditu, tflite_runtime saiatzen da (aukerakoa) eta erabiltzailea i2c eta gpio taldeetan sartzen du. Berrabiarazi ondoren, sistema prest dago.

## Erabilera

```bash
cd ~/airearen_kalitatea
python3 denborak_neurtu.py
python3 sistema_exekutatu.py --hasiera 2026-03-03 --orduak 72 --tartea 0.5
```

`denborak_neurtu.py`-k zati bakoitzaren denbora erakusten du (milisegundotan) eta `emaitzak/denborak_<gailua>.csv` fitxategian gordetzen du. Balio horiek MIAko txostenaren 3. atalean daude.

`sistema_exekutatu.py`-k orduz ordu erakusten ditu NO2aren iragarpena, ICA kategoria, egoera, alertaren ziurtasuna eta alerta-maila koloretan. Beste aukera batzuk:

```bash
# 2026 osoa ahalik eta azkarren, laburpena bakarrik
python3 sistema_exekutatu.py --hasiera 2026-01-01 --orduak 9000 --tartea 0 --isilik

# MLPa TensorFlow Lite-rekin (tflite_runtime instalatuta badago)
python3 sistema_exekutatu.py --motorra tflite

# Waveshare plakako OLED pantailan ere bai, gelako tenperatura eta hezetasunarekin (BME280)
python3 sistema_exekutatu.py --pantaila

# Plakako I2C gailuak zerrendatu
python3 sistema_exekutatu.py --bilatu
```

Emaitzak `emaitzak/jetson_emaitzak.csv` fitxategian gordetzen dira. Fitxategi hori PCko `jetson/emaitzak/` karpetara kopiatuz gero, aginte-panelean ikus daiteke ("Jetsoneko emaitzak (CSV)" aukera).

## Arazoak

| Arazoa | Irtenbidea |
|---|---|
| `Illegal instruction (core dumped)` | `export OPENBLAS_CORETYPE=ARMV8` (scriptek bakarrik jartzen dute) |
| `ModuleNotFoundError: numpy` | `sudo apt-get install python3-numpy` |
| Errorea `.tflite` irekitzean | `--motorra numpy` erabili (lehenetsia da, emaitza berdinak) |
| Koloreak `\033[...` gisa | Terminalak ez ditu koloreak onartzen; ez du ezer hausten |
| OLED pantaila ez da pizten | Pantailaren RST pina (18) goian egon behar da; programak berak jartzen du. Baimen-errorea badago: `sudo python3 sistema_exekutatu.py --pantaila` |
