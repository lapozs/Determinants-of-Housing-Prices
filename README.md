# Determinants-of-Housing-Prices

Ez a repository a "Lakásárakat befolyásoló tényezők elemzése Budapest példáján" című szakdolgozathoz készült.
A projekt Python-alapú adatfeldolgozást, statisztikai elemzést és vizualizációkat tartalmaz, amelyek célja a budapesti ingatlanpiac árait befolyásoló főbb tényezők azonosítása és elemzése.

Struktúra:
Excel_base_FINAL-c1.xlsx : A kezdetleges adatstruktúra 
Excel_base_FINAL_ALT2-c1.xlsx : A kezdetleges adatstruktúrából szűrt (ADF, VIF alapján) végleges tényezők
adf.py : Augmented Dickey–Fuller (ADF) teszt az adatok stacionaritásának elemzésére
data_visu.py : Kezdetleges adatelemzést tartalmaz, vizualizációk
main.py : Az elemzések, modellek inicalizálása és eredményei, végeredmények mentése
vif_calc.py : Multikollinearitás vizsgálata, VIF-értékek

Készítette: Pozsgai Emil Csanád, gazdaságinformatikus alapszakos hallgató (Budapesti Corvinus Egyetem)
Kapcsolat: pozsgaicsanad@gmail.com / emil.pozsgai@stud.uni-corvinus.hu
