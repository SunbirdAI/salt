# SALT-31: A Machine Translation Benchmark Dataset for 31 Ugandan Languages


`Sunbird/salt-31`

## Description
A context-aware MT evaluation benchmark covering 31 Ugandan (and closely related regional) languages. Unlike sentence-level benchmarks (e.g. FLORES), SALT-31 uses five-sentence mini-dialogues built around realistic Ugandan communication scenarios, enabling evaluation of discourse-level phenomena — coreference, register consistency, cultural grounding — in addition to standard translation quality. Used to benchmark Sunbird's Sunflower model and other MT systems.

## Languages Covered
31 target languages across three families, plus English as the source language.

| Family | # Languages |
|---|---|
| Bantu | 17 |
| Nilotic | 9 |
| Central Sudanic | 5 |


| Code | Language | Family |
|---|---|---|
| eng | English | Source language |
| cgg | Rukiga | Bantu |
| gwr | Lugwere | Bantu |
| kin | Kinyarwanda | Bantu |
| koo | Rukonjo | Bantu |
| lsm | Samia | Bantu |
| lug | Luganda | Bantu |
| myx | Lumasaba | Bantu |
| nuj | Lunyole | Bantu |
| xog | Lusoga | Bantu |
| nyn | Runyankole | Bantu |
| nyo | Runyoro | Bantu |
| rub | Lugungu | Bantu |
| ruc | Ruruuli | Bantu |
| rwm | Kwamba | Bantu |
| swa | Swahili | Bantu |
| tlj | Lubwisi | Bantu |
| ttj | Rutooro | Bantu |
| ach | Acholi | Nilotic |
| adh | Dhopadhola | Nilotic |
| alz | Alur | Nilotic |
| kdi | Kumam | Nilotic |
| kdj | Karamojong | Nilotic |
| kpz | Kupsabiny | Nilotic |
| laj | Lango | Nilotic |
| pok | Pokot | Nilotic |
| teo | Ateso | Nilotic |
| bfa | Bari | Central Sudanic |
| keo | Kakwa | Central Sudanic |
| lgg | Lugbara | Central Sudanic |
| luc | Aringa | Central Sudanic |
| mhi | Ma'di | Central Sudanic |
 

## Size / Examples
- 20 communication scenarios × 5 sentences = **100 English source sentences**
- Each sentence translated into all **31 target languages**
- File Size- **262KB**
- Domains: health, market, family, school, transport, government, agriculture, and other everyday/formal Ugandan contexts

## Data Collection Methodology
- **Seed text:** Multiple LLMs (GPT-4.5, GPT-4o, DeepSeek R1, LLaMA 3.3 70B, Mistral Large, Gemini 2 Flash, Claude Sonnet) generated candidate English mini-sequences for 20 predefined scenarios; best candidates manually selected for cultural/contextual fit.
- **Translation:** Native speakers translated all sentences into the 31 target languages, prioritizing natural phrasing and cultural appropriateness over literal translation.
- **Verification:** Independently reviewed by linguists at Makerere University's Department of Linguistics for semantic fidelity, grammar, and cultural appropriateness; disagreements resolved by consensus.

## Intended Use Cases
- Benchmarking MT systems (both directions) across 31 Ugandan languages
- Diagnostic evaluation of discourse-level translation quality (coreference, register, cultural grounding) — not just sentence-level accuracy
- Comparing regionally specialized models vs. general-purpose multilingual models

## Known Limitation
- Small (100 sentences) — built for diagnostic evaluation, not model training or fine-grained benchmarking
- Doesn't cover all sociolinguistic registers, dialects, or code-switching across the 31 languages
- Text-only for now; speech modality planned for future work

## Last Updated
2026-02-02

## Citation
Nsumba, S., Akera, B., Ouma, E.N., Ssentanda, M., Kawalya, D., Bainomugisha, E., Mwebaze, E.T., & Quinn, J. (2026). *SALT-31: A Machine Translation Benchmark Dataset for 31 Ugandan Languages.* Proceedings of the 7th Workshop on African Natural Language Processing (AfricaNLP 2026), pages 211–216. ACL.

## HuggingFace Link
[hf.co/datasets/Sunbird/salt-31](https://hf.co/datasets/Sunbird/salt-31)

