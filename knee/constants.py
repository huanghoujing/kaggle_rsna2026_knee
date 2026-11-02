"""Names and paths that many modules use. The folders come from SETTINGS.json at the repository root."""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS = json.load(open(os.path.join(REPO, 'SETTINGS.json')))
DATA = os.environ.get('KNEE_DATA', os.path.join(REPO, SETTINGS['DATA_DIR']))   # env.sh: KNEE_DATA moves the data
RAW = os.path.join(DATA, SETTINGS['RAW_DATA_DIR'])      # the competition files (stream_unzip output)
META = os.path.join(DATA, SETTINGS['META_DIR'])         # DICOM header tables, knee sides, tissue tables, joint boxes
LABELS_DIR = os.path.join(DATA, SETTINGS['LABELS_DIR'])   # the label tables (docs/labels.md)

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
          'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
PLANES = ['Sagittal', 'Coronal', 'Axial']
