#!/bin/bash
# Jetson Nano prestatzeko scripta. jetson karpetaren barruan "bash instalatu.sh" behin bakarrik exekutatu.

cd "$(dirname "$0")"

echo "== 1/6 Paketeen zerrenda eguneratzen =="
sudo apt-get update

echo "== 2/6 Oinarrizko paketeak instalatzen =="
# numpy: ereduak exekutatzeko | pil eta smbus: pantaila eta sentsorea | i2c-tools: i2cdetect
sudo apt-get install -y python3-pip python3-numpy python3-pil python3-smbus i2c-tools

echo "== 3/6 pip eguneratzen (Python 3.6rako azken bertsioa) =="
python3 -m pip install --user --upgrade "pip<22"

echo "== 4/6 Python liburutegiak =="
python3 -m pip install --user "smbus2==0.4.3"
python3 -m pip install --user --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime \
    || echo "  (aukerakoa) tflite_runtime ez da instalatu: sistemak numpy-rekin funtzionatzen du"

echo "== 5/6 Baimenak eta karpetak =="
# Jetson Nano: pip-eko numpyk Illegal instruction ematen du aldagai hau gabe
grep -q OPENBLAS_CORETYPE ~/.bashrc || echo "export OPENBLAS_CORETYPE=ARMV8" >> ~/.bashrc
export OPENBLAS_CORETYPE=ARMV8
# I2C erabiltzeko sudo gabe (berrabiarazi ondoren aplikatzen da)
sudo usermod -aG i2c "$USER"
# Pantailaren RST pina (GPIO) sudo gabe erabiltzeko
sudo groupadd -f -r gpio
sudo usermod -aG gpio "$USER"
mkdir -p emaitzak
chmod +x *.py instalatu.sh 2>/dev/null

echo "== 6/6 Egiaztapena =="
python3 --version
python3 -c "import numpy; print('numpy', numpy.__version__)"
python3 -c "import sistema; s = sistema.sistema_kargatu('ereduak'); print('Ereduak kargatuta, motorra:', s['motorra'])"
python3 -c "import tflite_runtime; print('tflite_runtime OK')" 2>/dev/null || echo "tflite_runtime ez dago (ez da beharrezkoa)"
echo
echo "I2C gailuak 1. busean:"
sudo i2cdetect -y -r 1

echo
echo "Prest! Berrabiarazi Jetsona (sudo reboot) I2C baimenak aplikatzeko."
echo "Gero:  python3 sistema_exekutatu.py --hasiera 2026-03-03 --orduak 72 --tartea 0.5"
echo "       python3 sistema_exekutatu.py --pantaila"