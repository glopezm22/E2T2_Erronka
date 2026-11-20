# E2T2 Erronka · Smart Environment AI

Barakaldoko airearen kalitatea iragartzeko eta monitorizatzeko sistema adimenduna, Bizkaiko Foru Aldundiarentzat.

Taldea: Erlantz, Gorka, Iñigo eta Iraitz

## Entregagarriak

| # | Entregagarria | Non |
|---|---|---|
| 1 | EDA | koadernoak/01_karga_eta_EDA.ipynb |
| 2 | Aurreprozesamenduko pipeline-a eta dokumentazioa | koadernoak/02, src/, docs/aurreprozesamenduaren_dokumentazioa.docx |
| 3 | Ereduak eta txosten konparatiboa | koadernoak/03-05, docs/SAA_txostena.docx |
| 4 | Sare neuronala | koadernoak/06, docs/MIA_txostena.docx (2. atala) |
| 5 | Sistema Jetson Nanon | jetson/, koadernoak/07, docs/MIA_txostena.docx (3. atala) eta demoaren bideoa |
| 6 | Aginte-panela | dashboard/app.py, dashboard.bat, docs/MIA_txostena.docx (4. atala) |
| 7 | Sistema adituna | koadernoak/08, docs/MIA_txostena.docx (5. atala) |
| 8 | Aurkezpena | docs/aurkezpena/ |

## Datuak

Open Data Euskadiko airearen kalitate-sarea, Barakaldoko estazioa, 2023tik 2026ko azaroaren 12ra arteko orduko datuak. 2023-2025 entrenatzeko (26.039 ordu) eta 2026 probatzeko (7.253 ordu).

- data/jatorrizkoak/: jatorrizko CSVak (orduko datuak eta indizea urteka), ez dira inoiz aldatzen. beste_estazioak/ karpetakoak EDAn alderatzeko bakarrik dira.
- data/prozesatuak/: 02 koadernoak sortutako datu garbiak eta banaketak.

## Egitura

```
E2T2_Erronka/
├── data/            jatorrizko eta prozesatutako datuak
├── koadernoak/      01-08 koadernoak
├── src/             karga.py eta aurreprozesamendua.py
├── ereduak/         joblib eta keras ereduak
├── docs/            txostenak eta emaitza-taulak
├── dashboard/       Streamlit aginte-panela
├── dashboard.bat    aginte-panela abiarazteko
└── jetson/          Jetson Nanorako sistema
```

## Instalazioa eta exekuzioa (Windows)

Lehenik liburutegi hauek instalatu behar dira (PCan; Jetsonek numpy bakarrik behar du, eta instalatu.sh-k jartzen du):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter ipykernel tensorflow streamlit altair
jupyter notebook
```

Koadernoak ordenan exekutatu (01etik 08ra). 02ak data/prozesatuak eta ereduak karpetak betetzen ditu, eta 06ak eta 07ak jetson/ereduak.

Aginte-panela: klik bikoitza dashboard.bat fitxategian (edo `python -m streamlit run dashboard/app.py`). Nabigatzailean irekitzen da.

Jetson Nano: ikusi jetson/JETSON_README.md.

## Emaitza nagusiak

| Zeregina | Eredua | 2026ko proba | Erreferentzia (iraunkortasuna) |
|---|---|---|---|
| Hurrengo orduko NO2a | MLP [64, 32] Keras | MAE 4,12, R² 0,78 (Lasso: 4,24) | MAE 4,55, R² 0,71 |
| Hurrengo orduko ICA | Erabaki-zuhaitza (4 sakonera) | F1 makroa 0,89, "Txarra" recall 1,00 | F1 makroa 0,95, recall 0,93 |
| Ingurumen-egoera | K-means (K = 4) | Aire garbia, Gaueko bareta, Trafiko-punta, Partikula-episodioa | – |
| Alertak | Sistema adituna (5 arau + ziurtasun-faktoreak) | 2026ko 128 ordu "Txarra" guztiak detektatuta | – |
| Jetson Nanon hedapena | Sistema osoa numpy-rekin | 0,54 ms orduko, PCko emaitza berberak | PCan 0,04 ms |
