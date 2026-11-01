"""Keyphrases como lista de termos por documento: P@k, R@k, F1@k, R-precision, MRR e Bpref."""
import json
import re
import unicodedata

from .metricas import bpref, r_precision, reciprocal_rank

_BORDA = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)


def normalizar_padrao(termo):
    """maiúscula e minúscula não fazem diferença, nem espaço extra ou pontuação nas bordas"""
    termo = unicodedata.normalize("NFC", str(termo)).lower()
    return _BORDA.sub("", " ".join(termo.split()))


def _id(d, n_linha):
    meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
    for v in (d.get("id"), meta.get("id_doc"), meta.get("id")):
        if v is not None and v != "":
            return str(v)
    return f"doc{n_linha}"


def ler_keyphrases(caminho, maior_score_melhor=True):
    """JSONL -> [(id, [termos em ordem])]; aceita lista, [termo, score] ou formato do Doccano"""
    docs = []
    with open(caminho, encoding="utf-8") as f:
        for n_linha, linha in enumerate(f, start=1):
            if not linha.strip():
                continue
            d = json.loads(linha)
            if "keyphrases" in d:
                pares = []
                for i, it in enumerate(d["keyphrases"] or []):
                    if isinstance(it, str):
                        pares.append((it, None, i))
                    elif isinstance(it, dict):
                        pares.append((it.get("text") or it.get("keyphrase"), it.get("score"), i))
                    else:
                        pares.append((it[0], it[1] if len(it) > 1 else None, i))
                if all(s is not None for _, s, _ in pares):
                    sinal = -1 if maior_score_melhor else 1
                    pares.sort(key=lambda p: (sinal * float(p[1]), p[2]))
                termos = [t for t, _, _ in pares if t]
            else:
                texto = d.get("text") or ""
                if "entities" in d:
                    spans = [(e["start_offset"], e["end_offset"]) for e in d["entities"]]
                else:
                    spans = [(e[0], e[1]) for e in d.get("label", [])]
                termos = [texto[a:b] for a, b in sorted(spans)]
            docs.append((_id(d, n_linha), termos))
    return docs


def _parear(gold, pred):
    """junta gold e pred pelo id; sem ids em comum, pela ordem das linhas"""
    por_id = dict(pred)
    if not any(g in por_id for g, _ in gold):
        assert len(gold) == len(pred)
        return [(g, t, p) for (g, t), (_, p) in zip(gold, pred)]
    return [(g, t, por_id.get(g, [])) for g, t in gold]


def _limpar(termos, normalizar):
    """normaliza e tira repetidos, mantendo a ordem"""
    return list(dict.fromkeys(n for n in (normalizar(t) for t in termos) if n))


def _prf_at_k(ranking, gold, k):
    if not gold:
        return None
    acertos = sum(1 for c in ranking[:k] if c in gold)
    p = acertos / k
    r = acertos / len(gold)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _media(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def avaliar_keyphrases(gold, pred, k=(5, 10), normalizar=normalizar_padrao, maior_score_melhor=True,
                       imprimir=True):
    """comparação entre gold e pred
    pred pode ser um dict {"metodo": caminho} pra comparar vários de uma vez
    maior_score_melhor=False pra métodos em que menor score é melhor (YAKE); aceita dict por método"""
    if isinstance(pred, dict):
        sentido = maior_score_melhor if isinstance(maior_score_melhor, dict) else {}
        resultados = {nome: avaliar_keyphrases(gold, c, k, normalizar,
                                               sentido.get(nome, True) if sentido else maior_score_melhor,
                                               imprimir=False)
                      for nome, c in pred.items()}
        if imprimir:
            _imprimir_tabela(resultados, k)
        return resultados

    ks = (k,) if isinstance(k, int) else tuple(k)
    trios = _parear(ler_keyphrases(gold), ler_keyphrases(pred, maior_score_melhor))

    rankings = [_limpar(p, normalizar) for _, _, p in trios]
    golds = [set(_limpar(g, normalizar)) for _, g, _ in trios]

    r = {"n_docs": sum(1 for g in golds if g)}
    for kk in ks:
        prf = [_prf_at_k(rk, g, kk) for rk, g in zip(rankings, golds)]
        r[f"P@{kk}"] = _media(x[0] if x else None for x in prf)
        r[f"R@{kk}"] = _media(x[1] if x else None for x in prf)
        r[f"F1@{kk}"] = _media(x[2] if x else None for x in prf)
    r["r_precision_medio"] = _media(r_precision(rk, g) for rk, g in zip(rankings, golds))
    r["mrr"] = _media(reciprocal_rank(rk, g) for rk, g in zip(rankings, golds))
    r["bpref_medio"] = _media(bpref(rk, g) for rk, g in zip(rankings, golds))
    r["por_documento"] = [{"id": i, "gold": sorted(g), "pred": rk, "acertos": [c for c in rk if c in g]}
                          for (i, _, _), rk, g in zip(trios, rankings, golds)]

    if imprimir:
        _imprimir_tabela({"pred": r}, ks)
    return r


def _imprimir_tabela(resultados, k):
    ks = (k,) if isinstance(k, int) else tuple(k)
    cols = [c for kk in ks for c in (f"P@{kk}", f"R@{kk}", f"F1@{kk}")] + ["r_precision_medio", "mrr", "bpref_medio"]
    nomes = {"r_precision_medio": "R-prec", "mrr": "MRR", "bpref_medio": "Bpref"}
    largura = max(len(n) for n in resultados) + 2
    print("".ljust(largura) + "".join(nomes.get(c, c).rjust(8) for c in cols))
    for nome, r in resultados.items():
        vals = "".join(("   -   " if r[c] is None else f"{r[c]:.3f}").rjust(8) for c in cols)
        print(nome.ljust(largura) + vals)
    n = next(iter(resultados.values()))["n_docs"]
    print(f"({n} documentos com gold)")
