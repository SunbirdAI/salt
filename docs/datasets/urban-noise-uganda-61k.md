# urban-noise-uganda-61k

## Description

The `urban-noise-uganda-61k` dataset consists of audio samples representing urban noise environments in Uganda, designed for tasks such as noise classification, audio tagging, and sound analysis. It was collected as part of a large-scale noise mapping initiative across Kampala (five divisions) and Entebbe (four wards), in collaboration with city authorities (Kampala Capital City Authority and Entebbe Municipal Council). Each recording is tagged with a noise category, sound pressure level (SPL), GPS coordinates, and a timestamp — making it, per its authors, the first large-scale urban sound dataset collected in an African city, curated for African urban noise categories.

The dataset is described in a peer-reviewed data descriptor published in *Scientific Data* (2026), and was featured as a Nature Africa Research Highlight in March 2026.

## Languages Covered

Not applicable — this is an environmental/noise audio dataset (traffic, sirens, vendors, construction, etc.), not speech.

## Size / Records

~62.7k clips total, split across two configurations:
- **large** — full dataset
- **small** — reduced-size configuration

## Coverage

| City | Areas |
|---|---|
| Kampala | Five divisions (Central, Kawempe, Makindye, Nakawa, Rubaga) |
| Entebbe | Four wards (Central, Kiwafu, Katabi, Kigungu) |

## Data Collection Methodology

- **Pipeline:** Field collectors used a custom mobile application to capture short audio recordings at specific georeferenced locations across residential, commercial, and mixed-use urban areas.
- **Noise sources captured:** Traffic, sirens, outdoor vendors, religious gatherings, nightlife/entertainment venues, construction sites, and others.
- **Annotations per sample:** Noise category, sound pressure level (SPL, dB), GPS coordinates (latitude, longitude, altitude, accuracy), and timestamp.
- **Audio format:** Single-channel (mono), 16 kHz.
- **Taxonomy:** Noise categories were developed in consultation with city authorities to reflect noise types relevant to urban planning and noise-complaint resolution in the Ugandan context.

### Label semantics

The dataset carries two parallel label columns with different origins:

| Column | Description |
|---|---|
| `class` / `class_id` | Ground-truth label assigned at collection time. Kampala samples were hand-labelled by annotators; Entebbe samples were collected without class labels (`class_id = 19`, unclassified). |
| `predicted_class` / `predicted_class_id` | Label assigned post-collection by the SunEcho urban sound classifier via model inference, applied to all samples including Entebbe. |

Kampala samples carry both human and model labels (enabling agreement analysis); Entebbe samples carry only model-predicted labels.

## Data Fields

| Field | Type | Description |
|---|---|---|
| `audio` | Audio (16 kHz) | Raw audio clip |
| `class` | string | Human-annotated sound class (collection time) |
| `class_id` | int64 | Integer ID for human-annotated class |
| `predicted_class` | string | SunEcho model-predicted sound class |
| `predicted_class_id` | int64 | Integer ID for model-predicted class |
| `noise_measurement` | float64 | Ambient noise level (SPL) at time of collection |
| `latitude` | float64 | GPS latitude (decimal degrees) |
| `longitude` | float64 | GPS longitude (decimal degrees) |
| `altitude` | float64 | Altitude above sea level (metres) |
| `accuracy` | float64 | GPS horizontal accuracy (metres) |
| `submitter_id` | int64 | Anonymised data collector ID |
| `region` | string | Collection region (Kampala or Entebbe) |
| `timestamp` | string | ISO 8601 collection timestamp |

## Intended Use Cases

- Noise classification and audio tagging
- Spatiotemporal urban noise mapping
- Urban planning and noise-complaint resolution research
- Sound analysis / environmental sensing ML applications

## Known Limitations

- Entebbe samples lack human-annotated ground-truth labels (model-predicted labels only).
- Coverage is limited to Kampala and Entebbe — not representative of other Ugandan cities or regions.
- Label taxonomy reflects categories relevant to Ugandan urban planning use cases specifically, which may not map cleanly onto other noise taxonomies (e.g. UrbanSound8K, SONYC-UST).

## Why This Dataset Is Unique

Unlike existing benchmarks such as UrbanSound8K and SONYC-UST, each sample includes precise geospatial metadata, timestamps, and sound pressure level measurements, enabling detailed spatiotemporal noise mapping. It is also the first large-scale urban sound dataset collected in an African city, curated for African urban noise categories.

## How to Load

```python
from datasets import load_dataset

# Load the large configuration
large_dataset = load_dataset("Sunbird/urban-noise-uganda-61k", "large")

# Load the small configuration
small_dataset = load_dataset("Sunbird/urban-noise-uganda-61k", "small")
```

## Licence

Check the "Files and versions" tab on the HuggingFace dataset page for the exact license — not explicitly stated in the dataset card text; please confirm before publishing.

## HuggingFace Link

[huggingface.co/datasets/Sunbird/urban-noise-uganda-61k](https://huggingface.co/datasets/Sunbird/urban-noise-uganda-61k)

## Related Resources

- **Publication:** Nsumba, S., Muhanguzi, T., Ouma, E.N. et al. "Noise mapping and ambient sound recordings of the urban environment in Uganda." *Scientific Data* 13, 345 (2026). https://doi.org/10.1038/s41597-026-06658-w
- **Nature Africa feature:** "Fungus fighting fruit flies, dinosaur tracks, and city soundscapes" (Research Highlight, 12 March 2026) — https://www.nature.com/articles/d44148-026-00056-5
- **Figshare (full dataset + raw files):** https://doi.org/10.6084/m9.figshare.30168901
- **Real-time noise map (Kampala/Entebbe):** https://noise.sunbird.ai
- **Sunbird AI Environmental Sensing:** https://sunbird.ai/portfolio/environmental-sensing/

## Citation

```bibtex
@article{nsumba2026urban,
  title     = {Noise mapping and ambient sound recordings of the urban environment in Uganda},
  author    = {Nsumba, Solomon and Muhanguzi, Tibabwetiza and Ouma, Evelyn Nafula and
               Sekalala, Imran and Bainomugisha, Engineer and Mwebaze, Ernest and Quinn, John},
  journal   = {Scientific Data},
  volume    = {13},
  pages     = {345},
  year      = {2026},
  publisher = {Springer Nature},
  doi       = {10.1038/s41597-026-06658-w}
}
```
