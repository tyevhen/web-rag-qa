# Sources

Everything the corpus is built from. One folder per document source: `source.json` (what `POST /ingest/url`
with `mode: "pdf"` reads), reviewed `figures/pNNN.md` texts, and the downloaded original. The originals
(`*.pdf`, `*.txt`) are third-party works and are not committed: download them from the URL into their folder.

| Slug | Title | Publisher, year | Lang | Ingest | Status |
|---|---|---|---|---|---|
| `teke-2024` | Traumahaige esmane käsitlus Eestis | Kaitseväe Akadeemia et al., 2024 | et | `pdf` | ingested, 374 chunks |
| `haigekassa-esmaabi-2010` | Elupäästev esmaabi vigastuste korral | Eesti Haigekassa, 2010 | et | `pdf` | ingested, 49 chunks |
| `fm21-76` | FM 21-76 Survival Manual | US Army, 1992 | en | `single` (djvu.txt) | ingested, 361 chunks |
| `fm4-25-11` | FM 4-25.11 First Aid | US Army, 2002 | en | `single` (djvu.txt) | ingested, 199 chunks |
| — | TCCC Ukraine | tccc.org.ua/en | en | `crawl` | ingested, 4459 chunks |
| — | Ole valmis! | olevalmis.ee (et), olevalmis.ee/en | et, en | `crawl` | ingested, 220 + 296 chunks |

Original URLs:
- teke-2024: https://www.kvak.ee/files/2025/02/Traumahaige-esmane-kaesitlus-eestis-2024.pdf
- haigekassa-esmaabi-2010: https://tervisekassa.ee/sites/default/files/haigekassa_teatajad/hk_vigastused_esmaabi.pdf
- fm21-76: https://archive.org/download/Fm21-76SurvivalManual/FM21-76_SurvivalManual_djvu.txt
- fm4-25-11: https://archive.org/download/FM4-25.11/FM4-25.11_djvu.txt

PDFs are read with `pymupdf4llm` (AGPL-3.0). Figure texts describe diagrams whose content is only in the
image; they are drafted from the rendered page and reviewed by hand (`FIGURES_REVIEW.md`).
