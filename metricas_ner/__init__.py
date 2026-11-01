"""metricas_ner: métricas de NER e keyphrases a partir de JSONL."""
from .avaliar import avaliar, imprimir_resumo
from .juiz import julgar
from .keyphrases import avaliar_keyphrases, ler_keyphrases, normalizar_padrao
from .leitura import ler_jsonl, ler_labels
from .metricas import (
    bpref, calcular_f1s, classificar_mismatches, f1_por_contagens, f1s_por_contagens,
    metricas_ranking, r_precision, reciprocal_rank,
)

__version__ = "0.2.0"
