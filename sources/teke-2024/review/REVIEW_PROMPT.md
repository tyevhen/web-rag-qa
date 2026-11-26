# Review: text descriptions of medical diagrams (TEKE 2024, Estonian)

## Your task

You get 12 page images from *Traumahaige esmane käsitlus Eestis 2024* (TEKE), an Estonian trauma-care manual,
and for each a **draft**: an Estonian text description of the diagram on that page. The image file name gives the
page: `p017.png` belongs to draft "Page 17".

The drafts will be added to a search index, and an assistant will quote them as if the manual said them. So a draft
must say **exactly what the image shows, and nothing more**. Compare each draft with its image and report errors.

## What counts as an error (most serious first)

1. **A wrong number**: dose, percentage, rate, pressure, score, points, weight, age, time, threshold, or a wrong
   `<` / `>` / `≥` / range.
2. **A wrong branch or order** in an algorithm: a yes/no swapped, an arrow leading to a different step than in the
   image, steps out of order.
3. **Added content**: advice, explanation or conclusions that the image does not show (even if medically true).
4. **Missing content**: a box, step, row, column or label in the image that the draft leaves out. Decorative
   elements (logos, colours with no meaning) don't count.
5. **Wrong or misleading Estonian**: a term that differs from the image's wording in a way that changes meaning.
   Ignore style.

Do not judge whether the medicine is right; only whether the draft matches the image. Read small print carefully
(zoom into the image). If you cannot read something in the image, say so instead of guessing.

## Output format

For each page, in order, write exactly one of:

- `Page N: OK`
- `Page N: FIX`, then a list of errors, each as: what the draft says → what the image shows (quote both), with the
  error category number. Then the **complete corrected draft** in a fenced code block, ready to save as the file.
- `Page N: DROP`, with the reason, when the image is too unclear to describe reliably.

End with one line: `Summary: X OK, Y FIX, Z DROP`.

Keep the first line of every draft (the bold `**Joonis …**` / `**LISA …**` caption line) unchanged.

## Drafts

### Page 17 (image `p017.png`)

````markdown
**Joonis 4.1. Esmase triaaži juhend (kirjeldus)**

Esmane triaaž ehk sõelumine (triage sieve), sammud järjekorras:

1. Massiivne väline verejooks? JAH: žgutt, kannatanu on T1. EI: järgmine samm.
2. Kõnnib? JAH: T3. EI: järgmine samm.
3. Hingab? EI: ava hingamisteed ja kontrolli uuesti. Kui ei hinga ka pärast hingamisteede avamist: SURNUD. Kui hakkab hingama: püsiv külili asend, T1. JAH: järgmine samm.
4. Teadvus? P või U: keera stabiilsesse külili asendisse, T1. A või V: järgmine samm.
5. Hingamissagedus? Alla 10 x/minutis või üle 30 x/minutis: T1. 10–29 x/min: järgmine samm.
6. Südametöö (pulss/kapillaartäituvus)? Pulss randmelt puudub või üle 120x´ ja/või kapillaartäituvus 3 ja enam sek: T1. Pulss alla 120x´ ja/või kapillaartäituvus 1 või 2 sek: T2.
````

### Page 67 (image `p067.png`)

````markdown
**Joonis 9.4. FAST-i algoritm (kirjeldus)**

- FAST positiivne: konsulteeri kirurgiga. Laparotoomia või angiograafia.
- FAST negatiivne, kaks võimalust:
  - jälgi / korda FAST-i;
  - hemodünaamika häire: konsulteeri kirurgiga, lisauuringud või operatsioonituba.
````

### Page 84 (image `p084.png`)

````markdown
**Joonis 10.1. Jälgimisleht, mis kajastab peaajutraumaga patsiendi seisundi halvenemist (kirjeldus)**

Jälgimislehel on aja jooksul näha peaajutraumaga patsiendi seisundi halvenemine (Cushingi refleks):

- GKS väheneb;
- vererõhk tõuseb;
- pulsisagedus aeglustub;
- hingamissagedus aeglustub.
````

### Page 116 (image `p116.png`)

````markdown
**Joonis 12.1. Traumahaige elustamise algoritm (kirjeldus)**

1. Võimalik mittetraumaatiline südameseiskus? JAH: ALS. EI: jätka järgmiste sammudega.
2. Peata väline verejooks. Aseta vaagnalahas.
3. Kohene transport lähimasse traumakeskusesse.
4. Taga avatud hingamistee ja vajadusel normaalne ventilatsioon.
5. Pleura nõeldekompressioon bilateraalselt (välistab ventiilpingelise õhkrinna).
6. Südame tamponaadi dreneerimine (haiglas?).
7. Transfusioon ehk verekomponentide ülekanne.
8. ROSC? EI: tegevuse lõpp. JAH: operatsioon.
````

### Page 122 (image `p122.png`)

````markdown
**Joonis 13.4. Põletuspinna jaotus protsentides Lundi ja Browderi järgi (kirjeldus)**

A. Täiskasvanu (üheksate reegel):

| Kehapiirkond | Osa kehapinnast |
|---|---|
| Pea ja kael | 9% |
| Rindkere ja kõht | 18% |
| Käsi (kumbki) | 9% |
| Jalg (kumbki) | 18% |
| Lahkliha ja suguelundid | 1% |

B. Laps, eespind ja selg eraldi (keha mõõtmete protsendi (%BSA) muutus lapse kasvades). Kummalgi joonisel on näidatud:

- kael 1%;
- kere 13%;
- õlavars 2% (kumbki);
- küünarvars 1½% (kumbki);
- käelaba 1½% (kumbki);
- labajalg 1¾% (kumbki);
- eespinnal suguelundid 1%;
- seljal tuharad 2½% (kumbki).

Pea (a), reie (b) ja säär (c) on tähistatud tähtedega, sest nende osa muutub lapse vanusega.
````

### Page 156 (image `p156.png`)

The author was unsure here: Which side is upwind: check the wind arrow against the zones.

````markdown
**Joonis 14.1. Ohuala tähistus (kirjeldus)**

Ohualad on kujutatud kolme üksteise sees asuva ringina:

- keskel kuum tsoon (punane);
- selle ümber soe tsoon (kollane);
- välimine ring on külm tsoon (roheline).

Dekontaminatsiooniala asub soojas tsoonis kuuma tsooni serval, tuulepoolsel küljel: tuule suund on joonisel dekontaminatsioonialast kuuma tsooni poole.
````

### Page 171 (image `p171.png`)

The author was unsure here: Step 0 (cover mouth/nose) is a guess from the pictogram.

````markdown
**Joonis 14.4. Hollandi dekontaminatsiooni piktogrammiline juhis kannatanule (kirjeldus)**

Piktogrammid (GHOR Rotterdam-Rijnmond) näitavad dekontaminatsiooni samme tuuma-, bioloogilise ja keemilise saastatuse korral:

0. Tuumasaastatuse korral: hoia suu kaetud (piktogrammil inimene katab suu).
1. Lahti riietumine (undress): sisene, võta riided seljast, pane riided kotti ja märgista kott. Bioloogilise ja keemilise saastatuse korral järgneb kontroll.
2. Pesemine (wash): pese end pesuvahendiga määratud aja (x min), seejärel loputa duši all määratud aja (x min). Järgneb kontroll: korras, mine sammule 3; mitte korras, korda sammu 2.
3. Riietumine (dress): pane selga puhtad riided, järgneb kontroll ja vajadusel ravi, siis registreerimine ja väljumine.
````

### Page 174 (image `p174.png`)

````markdown
**Joonis 14.5. Mürgistuse tunnused (kirjeldus)**

| Tunnus | Närvimürk | Tsüaniidid |
|---|---|---|
| Pupillid | ahenenud | normaalsed või laienenud |
| Nahk | higine, tsüanoos, fastsikulatsioon | kuiv, roosa, tsüanootiline |
| Hingamine | bronhokonstriktsioon, kiire aeglustumine, vilistav hingamine | esmalt hingamissagedus kiireneb, seejärel aeglustub, sügav ja ebaregulaarne |
| Sekreedid | suurenenud süljeeritus/kopsusekreet | normaalne |
````

### Page 182 (image `p182.png`)

The author was unsure here: Two age labels were unreadable; the draft marks them '(silt loetamatu)'.

````markdown
**Joonis 16.2. Kehakaalu ja värvikoodi järgi grupeeritud laste tarvikute näidiskapp (kirjeldus)**

Kapi sahtlid on värvikoodiga ja märgistatud lapse kehakaalu ja vanuse järgi:

| Kehakaal | Vanus |
|---|---|
| 3–5 kg | (silt loetamatu) |
| 6–9 kg | (silt loetamatu) |
| 10–11 kg | 1-aastane |
| 12–14 kg | 2-aastane |
| 15–18 kg | 3–4-aastane |
| 19–23 kg | 5–6-aastane |
| 24–29 kg | 7–9-aastane |
| 30–36 kg | 10–11-aastane |
````

### Page 208 (image `p208.png`)

````markdown
**Joonis 20.1. Valuravi astmeline käsitlus (kirjeldus)**

| Valu tugevus | Ravimid |
|---|---|
| Nõrk valu | paratsetamool, MSPVA |
| Mõõdukas valu | paratsetamool, MSPVA, nõrk opiaat (tramadool, kodeiin) |
| Tugev valu | paratsetamool, MSPVA, tugev opiaat (morfiin, fentanüül, oksükodoon) |
````

### Page 225 (image `p225.png`)

The author was unsure here: Score table was read from a crop; check every points value and the P1/P2/P3 totals.

````markdown
**LISA 3. Teisese triaaži algoritmiga kaardi näidis (kirjeldus)**

Triaažikaart (õppevahend, ingliskeelne) koosneb värvilistest prioriteediosadest 1 (punane), 2 (kollane) ja 3 (roheline) ning väljadest:
- MIST: mehhanism (Mechanism), vigastused (Injury), tunnused (Signs), ravi (Treatment);
- AMPLE: allergiad, ravimid, varasemad haigused, viimane söögikord, sündmus;
- patsiendi andmed, prioriteet 1, 2, 3 või surnud (DEAD), transpordi aeg ja sihtkoht.

Teisese triaaži skoor kaardil: liidetakse kolm punktisummat.

| Näitaja | 4 p | 3 p | 2 p | 1 p | 0 p |
|---|---|---|---|---|---|
| Glasgow' kooma skaala summa | 13–15 | 9–12 | 6–8 | 4–5 | 3 |
| Hingamissagedus | 10–29 | üle 29 | 6–9 | 1–5 | 0 |
| Süstoolne vererõhk | 90 või rohkem | 76–89 | 50–75 | 1–49 | 0 |

Kokku 12 punkti: prioriteet 3. Kokku 11 punkti: prioriteet 2. Kokku 10 või vähem: prioriteet 1.
````

### Page 230 (image `p230.png`)

````markdown
**LISA 5. Hamba esmaabi: "Päästa oma hammas" (kirjeldus)**

Hambatrauma korral on jäävhammas enamasti päästetav, kui tead, mida teha.

Kui hammas on murdunud:
1. Otsi hambatükk üles.
2. Murdunud tüki võib hambaarst tagasi panna.
3. Selleks mine kiiresti hambaarsti juurde.

Kui hammas tuli koos juurega välja:
1. Otsi hammas üles.
2. Ära puutu hammast juure osast.
3. Kui hammas on must, siis loputa korraks veega.
4. Edasi vali sobivaim variant: a) pane hammas kohe oma kohale tagasi; b) pane hammas piima või füsioloogilise lahuse sisse; c) kui eelmised variandid ei toimi, siis hoia hammast põses, suu sees.
5. Pöördu hambaarsti poole järgneva 2 tunni jooksul, sest kiire tegutsemine aitab hammast päästa.
````
