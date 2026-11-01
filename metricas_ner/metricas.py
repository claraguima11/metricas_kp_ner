"""Mismatches (Type-1 a 5), F1 (4 variantes) e métricas de ranking."""
from collections import Counter

from .leitura import texto_do_span

TIPOS = ["type-1", "type-2", "type-3", "type-4", "type-5"]


# ======================= mismatches (Nejadgholi et al., 2020) =======================

def se_sobrepoem(a, b):
    return a["end"] > b["start"] and b["end"] > a["start"]


def _contexto(texto, inicio, fim, janela=80):
    a, b = max(0, inicio - janela), min(len(texto), fim + janela)
    return ("…" if a > 0 else "") + " ".join(texto[a:b].split()) + ("…" if b < len(texto) else "")


def classificar_mismatches(gold, pred, texto, id_doc=None):
    """comparação entre gold e pred de um documento; acerto exato não entra na lista"""
    detalhes = []
    gold_usados, pred_usadas = set(), set()

    def registro(tipo, g, p):
        ini = min(e["start"] for e in (g, p) if e)
        fim = max(e["end"] for e in (g, p) if e)
        return {
            "tipo": tipo, "doc": id_doc,
            "gold_texto": texto_do_span(texto, g) if g else None, "gold_label": g["label"] if g else None,
            "gold_span": (g["start"], g["end"]) if g else None,
            "pred_texto": texto_do_span(texto, p) if p else None, "pred_label": p["label"] if p else None,
            "pred_span": (p["start"], p["end"]) if p else None,
            "contexto": _contexto(texto, ini, fim),
        }

    for i_p, p in enumerate(pred):
        for i_g, g in enumerate(gold):
            if i_g in gold_usados or not se_sobrepoem(p, g):
                continue
            gold_usados.add(i_g)
            pred_usadas.add(i_p)
            if p["start"] == g["start"] and p["end"] == g["end"]:
                tipo = "type-3" if p["label"] != g["label"] else None
            else:
                tipo = "type-5" if p["label"] == g["label"] else "type-4"
            if tipo:
                detalhes.append(registro(tipo, g, p))
            break

    detalhes += [registro("type-2", g, None) for i, g in enumerate(gold) if i not in gold_usados]
    detalhes += [registro("type-1", None, p) for i, p in enumerate(pred) if i not in pred_usadas]
    return detalhes


# ============================ F1 ============================

def f1_por_contagens(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def f1s_por_contagens(tp_exato, n_tipos, aceitos=None):
    """estrito, relaxado e, com o juiz, learning-based e estendido"""
    t1, t2, t3, t4, t5 = (n_tipos.get(t, 0) for t in TIPOS)
    f1s = {
        "estrito": f1_por_contagens(tp_exato, t1 + t3 + t4 + t5, t2 + t3 + t4 + t5),
        "relaxado": f1_por_contagens(tp_exato + t5, t1 + t3 + t4, t2 + t3 + t4),
    }
    if aceitos is not None:
        a3, a4, a5 = (aceitos.get(t, 0) for t in ("type-3", "type-4", "type-5"))
        r3, r4, r5 = t3 - a3, t4 - a4, t5 - a5
        f1s["learning_based"] = f1_por_contagens(tp_exato + a5, t1 + t3 + t4 + r5, t2 + t3 + t4 + r5)
        f1s["estendido"] = f1_por_contagens(tp_exato + a3 + a4 + a5, t1 + r3 + r4 + r5, t2 + r3 + r4 + r5)
    return f1s


def calcular_f1s(detalhes, total_gold, total_pred, julgamentos=None):
    n = Counter(d["tipo"] for d in detalhes)
    tp_exato = total_gold - (n["type-2"] + n["type-3"] + n["type-4"] + n["type-5"])
    aceitos = None
    if julgamentos is not None:
        aceitos = Counter(j["tipo"] for j in julgamentos if j["decisao"] == "aceito")
    return f1s_por_contagens(tp_exato, n, aceitos)


# ======================= ranking (Hasan & Ng, 2014) =======================

def r_precision(ranking, gold):
    """acertos nas |gold| primeiras posições"""
    if not gold:
        return None
    n = len(gold)
    return sum(1 for c in ranking[:n] if c in gold) / n


def reciprocal_rank(ranking, gold):
    """1 / posição do primeiro acerto"""
    if not gold:
        return None
    for i, c in enumerate(ranking):
        if c in gold:
            return 1 / (i + 1)
    return 0.0


def bpref(ranking, gold):
    """nr limitado a R (Buckley & Voorhees, 2004); com N = 0 cada acerto vale 1"""
    if not gold:
        return None
    r = len(gold)
    n = sum(1 for c in ranking if c not in gold)
    soma, nr = 0.0, 0
    for c in ranking:
        if c in gold:
            soma += 1.0 if min(r, n) == 0 else 1 - min(nr, r) / min(r, n)
        else:
            nr += 1
    return soma / r


def metricas_ranking(rankings, golds):
    """média por documento; documento sem gold fica fora"""
    def media(f):
        vals = [v for v in (f(rk, g) for rk, g in zip(rankings, golds)) if v is not None]
        return sum(vals) / len(vals) if vals else None
    return {
        "r_precision_medio": media(r_precision),
        "mrr": media(reciprocal_rank),
        "bpref_medio": media(bpref),
        "n_docs": sum(1 for g in golds if g),
    }


def montar_ranking(texto, pred):
    """ordena por score; texto repetido conta uma vez"""
    ordenadas = sorted(pred, key=lambda e: -e["score"])
    return list(dict.fromkeys(texto_do_span(texto, e) for e in ordenadas))
