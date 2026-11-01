import json
import os

import pytest

from metricas_ner import (
    avaliar, bpref, classificar_mismatches, f1s_por_contagens, r_precision, reciprocal_rank,
)

AQUI = os.path.dirname(__file__)
EX = lambda nome: os.path.join(AQUI, "exemplos", nome)


def ent(texto, trecho, label, score=None):
    i = texto.index(trecho)
    return {"start": i, "end": i + len(trecho), "label": label, "score": score}


# ---------------- mismatches ----------------

def test_cada_tipo_de_mismatch():
    t = "O Banco do Brasil processou João Silva no Recife ontem"
    gold = [ent(t, "Banco do Brasil", "ORG"), ent(t, "João Silva", "PES"), ent(t, "Recife", "LOC")]
    pred = [ent(t, "Banco do", "ORG"),            # type-5: span menor, categoria certa
            ent(t, "João Silva", "ORG"),          # type-3: mesmo span, categoria errada
            ent(t, "ontem", "TEMPO")]             # type-1: não existe no gold
    tipos = sorted(d["tipo"] for d in classificar_mismatches(gold, pred, t))
    assert tipos == ["type-1", "type-2", "type-3", "type-5"]   # Recife -> type-2

    pred4 = [ent(t, "do Brasil processou", "LOC")]             # type-4: sobrepõe, categoria errada
    assert "type-4" in {d["tipo"] for d in classificar_mismatches(gold, pred4, t)}


def test_acerto_exato_nao_e_mismatch():
    t = "Banco do Brasil"
    assert classificar_mismatches([ent(t, t, "ORG")], [ent(t, t, "ORG")], t) == []


# ---------------- F1 ----------------

def test_exemplo_numerico_do_readme():
    # TPₑ=80, T1=5, T2=5, T3=4, T4=2, T5=9; juiz aceita 1 Type-3, 1 Type-4 e 6 Type-5
    n = {"type-1": 5, "type-2": 5, "type-3": 4, "type-4": 2, "type-5": 9}
    f = f1s_por_contagens(80, n, {"type-3": 1, "type-4": 1, "type-5": 6})
    assert round(f["estrito"]["f1"], 3) == 0.800
    assert round(f["learning_based"]["f1"], 3) == 0.860
    assert round(f["estendido"]["f1"], 3) == 0.880
    assert round(f["relaxado"]["f1"], 3) == 0.890


# ---------------- ranking ----------------

def test_ranking_exemplos_do_readme():
    assert r_precision(["a", "x", "b", "c", "y"], {"a", "b", "c"}) == pytest.approx(2 / 3)
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert bpref(["x", "a", "y", "b", "z"], {"a", "b"}) == pytest.approx(0.25)


def test_ranking_sem_gold_e_none():
    assert r_precision(["a"], set()) is None
    assert reciprocal_rank(["a"], set()) is None
    assert bpref(["a"], set()) is None


def test_bpref_protecoes():
    assert bpref(["a", "b"], {"a", "b"}) == 1.0                  # N = 0
    assert bpref(["x", "y", "z", "a"], {"a"}) == 0.0             # nr limitado a R


# ---------------- ponta a ponta ----------------

def test_exemplos_ponta_a_ponta():
    r = avaliar(EX("gold.jsonl"), EX("pred.jsonl"), labels=EX("labels.json"), imprimir=False)
    assert r["mismatches"] == {"type-1": 1, "type-2": 1, "type-3": 1, "type-4": 1, "type-5": 2}
    assert r["totais"]["exato"] == 6          # "Caixa Econômica Federal " (com espaço) conta como exato
    assert r["ranking"] is not None


def test_juiz_como_funcao():
    juiz_que_aceita_tudo = lambda texto: "ORGANIZACAO"
    r = avaliar(EX("gold.jsonl"), EX("pred.jsonl"), juiz=juiz_que_aceita_tudo, imprimir=False)
    f = r["f1"]
    assert f["estrito"]["f1"] <= f["learning_based"]["f1"] <= f["relaxado"]["f1"]
    assert f["learning_based"]["f1"] <= f["estendido"]["f1"]
    assert all(j["decisao"] == "nao_avaliavel" for j in r["julgamentos"] if j["tipo"] == "type-2")


def test_sem_score_pula_ranking(tmp_path):
    linhas = [json.loads(l) for l in open(EX("pred.jsonl"), encoding="utf-8")]
    for d in linhas:
        d["label"] = [e[:3] for e in d["label"]]
    p = tmp_path / "pred.jsonl"
    p.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in linhas), encoding="utf-8")
    assert avaliar(EX("gold.jsonl"), str(p), imprimir=False)["ranking"] is None


def test_formato_entities_e_pareamento_por_ordem(tmp_path):
    t = "Banco do Brasil em Recife"
    g = tmp_path / "g.jsonl"
    p = tmp_path / "p.jsonl"
    g.write_text(json.dumps({"id": 1, "text": t, "entities": [
        {"start_offset": 0, "end_offset": 15, "label": "ORG"}]}), encoding="utf-8")
    p.write_text(json.dumps({"id": 99, "text": t, "entities": [
        {"start_offset": 0, "end_offset": 15, "label": "ORG"}]}), encoding="utf-8")
    r = avaliar(str(g), str(p), imprimir=False)
    assert r["f1"]["estrito"]["f1"] == 1.0


def test_id_zero_no_meta(tmp_path):
    # meta.id = 0 é um id válido (não pode cair no fallback por ser "falso")
    from metricas_ner import ler_jsonl
    p = tmp_path / "a.jsonl"
    p.write_text(json.dumps({"text": "x", "label": [], "meta": {"id": 0}}), encoding="utf-8")
    assert ler_jsonl(str(p))[0]["id"] == "0"


# ---------------- keyphrases (lista de termos) ----------------

def _jsonl(tmp_path, nome, linhas):
    p = tmp_path / nome
    p.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in linhas), encoding="utf-8")
    return str(p)


def test_keyphrases_normaliza_e_ordena_por_score(tmp_path):
    from metricas_ner import avaliar_keyphrases
    g = _jsonl(tmp_path, "g.jsonl", [{"id": "a", "keyphrases": ["contrato de financiamento", "rescisão"]}])
    p = _jsonl(tmp_path, "p.jsonl", [{"id": "a", "keyphrases": [["banco", 0.2], ["Contrato  de Financiamento.", 0.9]]}])
    r = avaliar_keyphrases(g, p, k=(1, 2), imprimir=False)
    assert r["P@1"] == 1.0                     # maior score vem primeiro e casa depois de normalizar
    assert r["mrr"] == 1.0
    assert r["R@2"] == 0.5


def test_keyphrases_gold_no_formato_doccano(tmp_path):
    from metricas_ner import avaliar_keyphrases
    t = "O contrato de financiamento foi rescindido."
    g = _jsonl(tmp_path, "g.jsonl", [{"id": "a", "text": t, "label": [[2, 27, "KP"]]}])
    p = _jsonl(tmp_path, "p.jsonl", [{"id": "a", "keyphrases": ["rescindido", "contrato de financiamento"]}])
    r = avaliar_keyphrases(g, p, imprimir=False)
    assert r["mrr"] == 0.5


def test_keyphrases_varios_metodos():
    from metricas_ner import avaliar_keyphrases
    kp = lambda n: EX(os.path.join("keyphrases", n))
    r = avaliar_keyphrases(kp("gold.jsonl"), {"yake": kp("pred_yake.jsonl"), "keybert": kp("pred_keybert.jsonl")},
                           imprimir=False)
    assert r["yake"]["r_precision_medio"] == pytest.approx((2 / 3 + 1 / 2 + 1) / 3)
    assert r["yake"]["mrr"] == pytest.approx((1 + 1 / 2 + 1) / 3)


def test_keyphrases_score_menor_melhor(tmp_path):
    from metricas_ner import avaliar_keyphrases
    g = _jsonl(tmp_path, "g.jsonl", [{"id": "a", "keyphrases": ["rescisão"]}])
    p = _jsonl(tmp_path, "p.jsonl", [{"id": "a", "keyphrases": [["banco", 0.9], ["rescisão", 0.01]]}])
    assert avaliar_keyphrases(g, p, imprimir=False)["mrr"] == 0.5
    assert avaliar_keyphrases(g, p, maior_score_melhor=False, imprimir=False)["mrr"] == 1.0
    assert avaliar_keyphrases(g, {"yake": p}, maior_score_melhor={"yake": False}, imprimir=False)["yake"]["mrr"] == 1.0
