"""RSNA 2026 Knee Abnormality Detection."""
import os

# NumPy asks the kernel for huge pages for large arrays; on fragmented memory every such allocation waits on kernel
# compaction (we saw trainings and evaluations 8-20x slower). Small pages only; set before any `python -m knee...`
# imports NumPy.
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
