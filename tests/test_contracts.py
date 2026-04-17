import json
from pathlib import Path


def test_amplitude_schema_id_matches_filename():
    schema_path = Path(__file__).resolve().parents[1] / 'schemas' / 'amplitude_experiment_assigned.schema.json'
    payload = json.loads(schema_path.read_text(encoding='utf-8'))
    assert payload['$id'].endswith('amplitude_experiment_assigned.schema.json')


def test_exposure_batch_id_type_aligned_between_dwh_templates():
    root = Path(__file__).resolve().parents[1]
    ch = (root / 'sql' / 'clickhouse' / '002_fact_experiment_exposure.sql').read_text(encoding='utf-8')
    sf = (root / 'sql' / 'snowflake' / '002_fact_experiment_exposure.sql').read_text(encoding='utf-8')
    assert 'ingest_batch_id     String' in ch
    assert 'ingest_batch_id     VARCHAR' in sf
