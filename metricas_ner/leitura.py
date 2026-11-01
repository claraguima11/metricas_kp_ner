"""Leitura dos JSONL e preparação das entidades."""
import json
import re

PALAVRA = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def ler_labels(caminho):
    """lista de labels: .json do Doccano, lista de strings ou .txt (um por linha)"""
    if str(caminho).lower().endswith(".json"):
        with open(caminho, encoding="utf-8") as f:
            dados = json.load(f)
        if isinstance(dados, dict):
            dados = dados.get("labels", list(dados))
        nomes = [d if isinstance(d, str) else d.get("text") or d.get("name") for d in dados]
    else:
        with open(caminho, encoding="utf-8") as f:
            nomes = [linha.strip() for linha in f]
    return [str(n).strip() for n in nomes if n and str(n).strip()]


def ler_jsonl(caminho):
    """JSONL do Doccano -> lista de {"id", "texto", "entidades"}"""
    docs = []
    with open(caminho, encoding="utf-8") as f:
        for n_linha, linha in enumerate(f, start=1):
            if not linha.strip():
                continue
            d = json.loads(linha)

            ents = []
            if "entities" in d:
                for e in d["entities"]:
                    ents.append({"start": int(e["start_offset"]), "end": int(e["end_offset"]),
                                 "label": str(e["label"]).strip(), "score": e.get("score")})
            else:
                for e in d.get("label", []):
                    ents.append({"start": int(e[0]), "end": int(e[1]), "label": str(e[2]).strip(),
                                 "score": e[3] if len(e) > 3 else None})
            for e in ents:
                if e["score"] is not None:
                    e["score"] = float(e["score"])

            meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
            id_doc = next((v for v in (meta.get("id_doc"), d.get("id"), meta.get("id"))
                           if v is not None and v != ""), f"doc{n_linha}")
            docs.append({"id": str(id_doc), "texto": d.get("text"), "entidades": ents})
    return docs


def parear(docs_gold, docs_pred, imprimir=True):
    """junta gold e pred pelo id; sem ids em comum, pela ordem das linhas"""
    por_id = {d["id"]: d for d in docs_pred}
    if not any(g["id"] in por_id for g in docs_gold):
        assert len(docs_gold) == len(docs_pred)
        return list(zip(docs_gold, docs_pred))

    vazio = lambda i: {"id": i, "texto": None, "entidades": []}
    return [(g, por_id.get(g["id"], vazio(g["id"]))) for g in docs_gold]


def palavras(texto):
    """(inicio, fim) de cada palavra"""
    return [(m.start(), m.end()) for m in PALAVRA.finditer(texto)]


def encaixar_em_palavras(ent, spans_palavras):
    """span cortado ou com espaço sobrando vira palavras inteiras"""
    tocadas = [(a, b) for a, b in spans_palavras if a < ent["end"] and ent["start"] < b]
    if not tocadas:
        return None
    return {**ent, "start": tocadas[0][0], "end": tocadas[-1][1]}


def sem_sobreposicao(ents):
    """entidades sobrepostas no mesmo arquivo: fica a mais longa"""
    mantidas = []
    for e in sorted(ents, key=lambda e: (-(e["end"] - e["start"]), e["start"])):
        if all(e["end"] <= m["start"] or m["end"] <= e["start"] for m in mantidas):
            mantidas.append(e)
    mantidas.sort(key=lambda e: (e["start"], e["end"]))
    return mantidas, len(ents) - len(mantidas)


def preparar_entidades(texto, ents, encaixar=True, remover_sobrepostas=True):
    descartadas = 0
    if remover_sobrepostas:
        ents, descartadas = sem_sobreposicao(ents)
    if encaixar:
        spans = palavras(texto)
        ents = [e2 for e2 in (encaixar_em_palavras(e, spans) for e in ents) if e2 is not None]
    if remover_sobrepostas:
        ents, d2 = sem_sobreposicao(ents)
        return ents, descartadas + d2
    return sorted(ents, key=lambda e: (e["start"], e["end"])), descartadas


def texto_do_span(texto, ent):
    return " ".join(texto[ent["start"]:ent["end"]].split())
