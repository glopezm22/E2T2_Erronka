# PIA Aurreprozesamenduaren dokumentazioa

Barakaldoko estazioa · entrenamendua 2023-2025 · proba 2026 (azaroaren 12ra arte)
Erlantz, Gorka, Iñigo eta Iraitz

Dokumentu honek Open Data Euskadiko jatorrizko CSVetatik entrenatzeko prest dagoen datu-multzora iristeko egindako urrats guztiak biltzen ditu. Kodea eta grafikoak 02 koadernoan daude.

## 1. Abiapuntuko datuak

| Fitxategia | Errenkadak | Edukia |
|---|---|---|
| BARAKALDO_2023.csv … _2026.csv | 8.760 / 8.784 / 8.760 / 7.584 | Orduko neurketak: CO, CO_8h, NO, NOX, NO2, PM10, PM2.5, SO2, haizea, hezetasuna eta tenperatura |
| BARAKALDO_INDIZEA_2023.csv … _2026.csv | Berdinak | ICA kategoriak, kutsatzaile bakoitzeko eta estaziokoa |
| beste_estazioak/ (Erandio, Sangroniz, Mazarredo, 2025) | 8.760 bakoitza | EDAn alderatzeko bakarrik |

2026ko datuak azaroaren 12ra arte iristen dira. Entrenamendua 2023-2025 da eta proba 2026.

## 2. Jatorrizko fitxategien arazoak

| Arazoa | Irtenbidea |
|---|---|
| ; bereizlea, koma hamartarra eta latin-1 | pd.read_csv-ren parametroak |
| Ordua 01:00etik 24:00ra (GMT) | 24:00 hurrengo eguneko 00:00 bihurtu |
| Errenkadak alderantzizko ordenan | sort_index() |
| 2023an zutabeen ordena desberdina | Zutabeak izenaren arabera hautatu |
| ICA etiketak urtearen arabera desberdin idatzita | Euskarara bateratu: Oso ona, Ona, Erregularra, Txarra |
| "Sin datos" testua | Nulu bihurtu |
| 2026ko azken orduak argitaratu gabe | Kargatzean kendu |

## 3. Urratsak

| # | Urratsa | Non | Emaitza |
|---|---|---|---|
| 1 | CSVak irakurri, datak konpondu eta lau urteak elkartu | karga.py | 33.874 ordu, bat ere falta gabe |
| 2 | Balio ezinezkoak nulu bihurtu | balio_ezinezkoak_kendu | Ez da bat ere agertu |
| 3 | Outlierrak IQR eta z-scorearekin aztertu | outlierrak_zenbatu_iqr / _zscore | Mantendu egiten dira |
| 4 | Aldagai berriak sortu | aldagaiak_sortu | Egutegia, haize-sektorea, historia eta helburuak |
| 5 | Helbururik gabeko errenkadak kendu | 02 koadernoa | 33.292 errenkada |
| 6 | Urteka banatu | 02 koadernoa | Entrenamendua 26.039 / proba 7.253 |
| 7 | Nuluak medianarekin bete (numerikoak) eta maizenarekin (kategorikoak) | Pipeline | Nulurik gabe |
| 8 | StandardScaler numerikoei | Pipeline | Batez bestekoa 0, desbideratzea 1 |
| 9 | One-hot kategoriei (ordua, asteko eguna, hilabetea, haize-sektorea) | Pipeline | 65 zutabe |
| 10 | PCA probatu | 02 koadernoa | Baztertua |
| 11 | Hiperparametroak Pipeline osoaren gainean doitu (GridSearchCV) | 03 eta 04 koadernoak | Aurreprozesamendua zatiketa bakoitzean berriz doitzen da |
| 12 | Pipeline-a joblib-ekin gorde eta datu berriekin probatu | ereduak/ | Jatorrizko CSVekin dabil |

## 4. Aldagaiak

Sarrerak (18):

- Kutsatzaileak: CO, NO, NO2, PM10, PM25, SO2
- Meteorologia: tenperatura, hezetasuna, haize_abiadura, haize_sektorea (norabidea 8 sektoretan)
- Egutegia: ordua, aste_eguna, hilabetea
- Historia: NO2_duela_1h, NO2_duela_24h, NO2_batez_24h, PM10_batez_24h, PM25_batez_24h

Helburuak: NO2_hurrengoa (erregresioa), ICA_hurrengoa (sailkapena) eta PM10_hurrengoa eta PM25_hurrengoa (SAAn beste bi kutsatzaileen erregresioa).

Ez ditugu erabiltzen NOX eta CO_8h (erredundanteak) ezta uneko ICA kategoriak ere.

## 5. Erabakiak

- Outlierrak mantentzen ditugu. IQRk orduen % 2-4 markatzen ditu NO2, PM10 eta PM2.5ean, eta z-scoreak erdia inguru (banaketa normala suposatzen du, eta gure datuek eskuinerako isats luzea dute). Gehienak episodio errealak dira, adibidez 2023ko Urteberriko su artifizialak (PM2.5 = 288 µg/m³), eta hain zuzen ere horiek detektatu nahi ditugu.
- Nuluak betetzeko apunteetako lau estrategiak alderatu ditugu Random Forest batekin (2023-2024 entrenatzeko, 2025 balioztatzeko): batez bestekoa (MAE 4,313), mediana (4,317), mediana + "_was_missing" (4,323) eta errenkadak ezabatu (4,371, eta errenkaden % 32 galtzen da). Aldeak % 1 baino txikiagoak dira, beraz sinpleena eta sendoena aukeratzen dugu: mediana.
- Haizearen norabidea 8 sektoretan diskretizatzen dugu (pd.cut), EDAn norabideak eragina zuela ikusi genuelako eta graduak zuzenean ez direlako zentzuzkoak (0º eta 360º gauza bera dira).
- Ordua, asteko eguna, hilabetea eta haize-sektorea kategoriak dira; one-hot bidez kodetzen ditugu.
- StandardScaler aukeratzen dugu MinMaxScaler-en ordez: azken horrekin PM10 gailurrek gainerako balioak 0tik gertu estutzen dituzte.
- PCAk zutabeak 65etik 39ra murrizten ditu (% 95eko bariantza), baina erregresio lineala apur bat okertzen du (MAE 4,26tik 4,29ra) eta ez du denbora nabarmen aurrezten. Ez dugu erabiltzen.
- Pipeline bera erabiltzen dute SAAko eredu guztiek, eta hiperparametroak Pipeline osoaren gainean doitzen dira GridSearchCV-rekin (adibidez eredua__alpha Ridgen). Horrela balioztapen-zatiketa bakoitzean mediana eta eskalatzea entrenamenduko zatiarekin bakarrik kalkulatzen dira, eta ez dago datu-ihesik.
- Datuak urteka banatzen ditugu, ez ausaz: denbora-serie batean ondoz ondoko orduak nahasteak metrika faltsuki onak emango lituzke.
- Hurrengo ordua iragartzen dugu, eta sarrerek iragana bakarrik erabiltzen dute (datu-ihesik ez).

## 6. Mugak

- 2023ko PM10ak % 42 nulu ditu; ordu horietan ereduek medianarekin lan egiten dute.
- 2023ko udan ICA PM10 gabe kalkulatu zen, beraz etiketa horiek baikorregiak izan daitezke.
- 2026an meteorologia orduen % 31n falta da, eta medianarekin betetzen da.
- 2026an airea okerragoa izan da (% 9 ordu Erregular, entrenamenduko % 5,5en aldean).
- "Txarra" klasea orduen % 1,5-1,8 bakarrik da.
