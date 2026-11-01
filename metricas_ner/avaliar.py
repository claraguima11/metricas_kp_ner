"""avaliar(gold.jsonl, pred.jsonl) -> todas as métricas."""
import json
from collections import Counter

from .leitura import ler_jsonl, ler_labels, parear, preparar_entidades
from .metricas import TIPOS, calcular_f1s, classificar_mismatches, metricas_ranking, montar_ranking


def avaliar(gold, pred, labels=None, juiz=None, encaixar=True, remover_sobrepostas=True,
            salvar_detalhes=None, imprimir=True):
    """comparação entre gold e pred
    juiz: None (estrito e relaxado), "bert" ou uma função texto -> label
    ranking só sai se todas as entidades do pred tiverem score"""
    docs_gold = ler_jsonl(gold) if isinstance(gold, str) else gold
    docs_pred = ler_jsonl(pred) if isinstance(pred, str) else pred
    pares = parear(docs_gold, docs_pred, imprimir)

    if labels is not None and imprimir:
        lista = set(ler_labels(labels) if isinstance(labels, str) else labels)
        fora = sorted({e["label"] for docs in (docs_gold, docs_pred) for d in docs for e in d["entidades"]} - lista)
        if fora:
            print("labels fora da lista:", fora)

    detalhes, rankings, golds_ranking, golds_preparados = [], [], [], []
    total_gold = total_pred = descartadas = 0
    tem_score = True

    for g, p in pares:
        texto = g["texto"]
        assert p["texto"] is None or p["texto"] == texto

        ents_g, dg = preparar_entidades(texto, g["entidades"], encaixar, remover_sobrepostas)
        ents_p, dp = preparar_entidades(texto, p["entidades"], encaixar, remover_sobrepostas)
        descartadas += dg + dp
        total_gold += len(ents_g)
        total_pred += len(ents_p)
        golds_preparados.append({"id": g["id"], "texto": texto, "entidades": ents_g})

        detalhes += classificar_mismatches(ents_g, ents_p, texto, g["id"])

        if any(e["score"] is None for e in ents_p):
            tem_score = False
        else:
            rankings.append(montar_ranking(texto, ents_p))
            golds_ranking.append({" ".join(texto[e["start"]:e["end"]].split()) for e in ents_g})

    n = Counter(d["tipo"] for d in detalhes)
    totais = {"documentos": len(pares), "gold": total_gold, "pred": total_pred,
              "exato": total_gold - (n["type-2"] + n["type-3"] + n["type-4"] + n["type-5"]),
              "descartadas_sobreposicao": descartadas}

    julgamentos = None
    if juiz is not None:
        from .juiz import JuizBert, julgar
        if juiz == "bert":
            juiz = JuizBert()
            juiz.treinar(golds_preparados)
        julgamentos = julgar(detalhes, juiz, imprimir)

    f1s = calcular_f1s(detalhes, total_gold, total_pred, julgamentos)
    ranking = metricas_ranking(rankings, golds_ranking) if tem_score and any(rankings) else None

    if salvar_detalhes:
        with open(salvar_detalhes, "w", encoding="utf-8") as f:
            for d in (julgamentos or detalhes):
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    resultado = {
        "f1": f1s, "ranking": ranking,
        "mismatches": {t: n[t] for t in TIPOS}, "totais": totais,
        "detalhes": detalhes, "julgamentos": julgamentos,
    }
    if imprimir:
        imprimir_resumo(resultado)
    return resultado


def imprimir_resumo(r):
    t = r["totais"]
    print(f"\n{t['documentos']} documentos | gold: {t['gold']} entidades | pred: {t['pred']} | "
          f"exatos: {t['exato']}" + (f" | {t['descartadas_sobreposicao']} descartadas por sobreposição"
                                     if t["descartadas_sobreposicao"] else ""))
    print("Mismatches:", "  ".join(f"{k}={v}" for k, v in r["mismatches"].items()))
    if r["julgamentos"] is not None:
        aceitos = Counter(j["tipo"] for j in r["julgamentos"] if j["decisao"] == "aceito")
        print("Aceitos pelo juiz:", "  ".join(
            f"{tp}={aceitos[tp]}/{r['mismatches'][tp]}" for tp in ("type-1", "type-3", "type-4", "type-5")),
            "(type-1 fica fora do F1)")
    print()
    for nome, m in r["f1"].items():
        print(f"F1 {nome:15} {m['f1']:.3f}  (P={m['precision']:.3f}  R={m['recall']:.3f})")
    if r["ranking"]:
        rk = r["ranking"]
        print(f"\nR-precision {rk['r_precision_medio']:.3f} | MRR {rk['mrr']:.3f} | "
              f"Bpref {rk['bpref_medio']:.3f}  ({rk['n_docs']} docs com gold)")
