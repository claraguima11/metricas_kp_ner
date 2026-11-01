"""python -m metricas_ner gold.jsonl pred.jsonl [--labels labels.json] [--juiz] [--salvar erros.jsonl]"""
import argparse

from .avaliar import avaliar


def main():
    ap = argparse.ArgumentParser(prog="metricas_ner",
                                 description="métricas a partir de dois JSONL do Doccano")
    ap.add_argument("gold", help="JSONL de referência")
    ap.add_argument("pred", help="JSONL a avaliar")
    ap.add_argument("--labels", help="lista de labels")
    ap.add_argument("--juiz", action="store_true",
                    help="learning-based e estendido")
    ap.add_argument("--salvar", metavar="ARQ.jsonl", help="salva os erros num JSONL")
    ap.add_argument("--sem-encaixe", action="store_true", help="não encaixa os spans em palavras inteiras")
    a = ap.parse_args()
    avaliar(a.gold, a.pred, labels=a.labels, juiz="bert" if a.juiz else None,
            encaixar=not a.sem_encaixe, salvar_detalhes=a.salvar)


if __name__ == "__main__":
    main()
