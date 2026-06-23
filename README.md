# Clinical NLP Pipeline

A lightweight mock clinical NLP pipeline that reads office notes from text files, uses a local Ollama model to extract structured epilepsy-related information, and writes one row per note to a CSV file.

The pipeline is configured for:

- Ollama running locally
- `llama3.1:8b`
- one `.txt` office note per input file
- one combined CSV output file
- comma-separated values inside CSV cells, with standard CSV quoting

## What the pipeline extracts

The output includes

- visit type
- treatment context, including VNS, RNS, DBS, medication, dietary, and surgical treatment
- baseline seizure frequency
- seizure reduction and timepoint
- cognitive outcomes: memory, attention, processing speed, executive function
- psychiatric outcomes: depression, anxiety, ADHD, psychosis, substance use disorder
- requested SUDEP risk factors
- surgical, device, and medication adverse events
- exact evidence text from the note for each extracted finding

For status columns, the pipeline uses:

- `present`: explicitly documented for the patient
- `absent`: explicitly denied or ruled out
- `not_listed`: not documented in the note

Silence in a note is treated as `not_listed`, not `absent`.

## Project structure

```text
clinical-nlp/
├── config/
│   ├── csv_columns.json
│   ├── prompt.txt
│   ├── settings.yaml
│   └── vocabulary.yaml
│
├── data/
│   ├── input_notes/
│   │   ├── note_001.txt
│   │   └── note_002.txt
│   │
│   └── output_csv/
│       └── extractions.csv
│
├── logs/
│   ├── failed_notes.log
│   ├── pipeline_errors.log
│   └── raw_responses/
│
├── scripts/
│   ├── main.py
│   ├── extract.py
│   ├── validate.py
│   └── utils.py
│
├── requirements.txt
└── README.md
```

## Running the pipeline

Before running the pipeline, make sure Ollama is running and the model is installed:

```bash
ollama serve
ollama pull llama3.1:8b

python scripts/main.py
```

