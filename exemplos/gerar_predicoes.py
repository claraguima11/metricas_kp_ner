"""Treina o NER e grava o pred.jsonl do test com score por entidade (fora da biblioteca)."""
import argparse
import json

import numpy as np

from metricas_ner.leitura import ler_jsonl, palavras


# ============================ texto -> palavras + BIO ============================

def doc_para_palavras(doc):
    """texto -> palavras + tags BIO"""
    spans = palavras(doc["texto"])
    tags = ["O"] * len(spans)
    for e in sorted(doc["entidades"], key=lambda e: (e["start"], e["end"])):
        dentro = [i for i, (a, b) in enumerate(spans) if a < e["end"] and e["start"] < b]
        for k, i in enumerate(dentro):
            tags[i] = ("B-" if k == 0 else "I-") + e["label"]
    return spans, tags


def bio_para_entidades(tags):
    """B- abre, I- continua, o resto fecha (igual ao pipeline)"""
    ents, atual = [], None
    for i, t in enumerate(tags):
        if t.startswith("B-"):
            if atual:
                ents.append(atual)
            atual = {"label": t[2:], "start": i, "end": i + 1}
        elif t.startswith("I-") and atual:
            atual["end"] = i + 1
        else:
            if atual:
                ents.append(atual)
            atual = None
    if atual:
        ents.append(atual)
    return ents


def cortar_em_trechos(textos_palavras, tokenizer, max_subtokens):
    """janelas de até max_subtokens, cortando de preferência depois de . ; :"""
    if not textos_palavras:
        return [(0, 0)]
    pesos = [max(1, len(ids)) for ids in tokenizer(textos_palavras, add_special_tokens=False)["input_ids"]]
    janelas, ini = [], 0
    while ini < len(pesos):
        fim, acumulado = ini, 0
        while fim < len(pesos) and acumulado + pesos[fim] <= max_subtokens:
            acumulado += pesos[fim]
            fim += 1
        fim = max(fim, ini + 1)
        if fim < len(pesos):
            for k in range(fim, ini + (fim - ini) // 2, -1):
                if textos_palavras[k - 1] in {".", ";", ":"}:
                    fim = k
                    break
        janelas.append((ini, fim))
        ini = fim
    return janelas


def preparar(docs, tokenizer, max_subtokens):
    """documentos -> trechos que cabem no BERT"""
    trechos, spans_por_doc = [], []
    for i_doc, doc in enumerate(docs):
        spans, tags = doc_para_palavras(doc)
        textos = [doc["texto"][a:b] for a, b in spans]
        spans_por_doc.append(spans)
        for a, b in cortar_em_trechos(textos, tokenizer, max_subtokens):
            trechos.append({"doc": i_doc, "ini": a, "palavras": textos[a:b], "tags": tags[a:b]})
    return trechos, spans_por_doc


# ============================ dataset do HuggingFace ============================

def tokenizar(trechos, tokenizer, tag2id):
    from datasets import Dataset

    def alinhar(word_ids, tags):
        out, anterior = [], None
        for w in word_ids:
            out.append(-100 if w is None or w == anterior else tag2id[tags[w]])
            anterior = w
        return out

    enc = tokenizer([t["palavras"] for t in trechos], is_split_into_words=True,
                    truncation=True, max_length=512)
    enc["labels"] = [alinhar(enc.word_ids(i), t["tags"]) for i, t in enumerate(trechos)]
    return Dataset.from_dict(dict(enc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--saida", default="pred_test.jsonl")
    ap.add_argument("--modelo", default="neuralmind/bert-base-portuguese-cased")
    ap.add_argument("--epocas", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-subtokens", type=int, default=400)
    ap.add_argument("--agregacao", choices=["media", "minimo"], default="media")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="./ner-checkpoints")
    a = ap.parse_args()
    executar(**{k.replace("-", "_"): v for k, v in vars(a).items()})


def executar(train, dev, test, saida="pred_test.jsonl", modelo="neuralmind/bert-base-portuguese-cased",
             epocas=3, lr=2e-5, batch=16, max_subtokens=400, agregacao="media", seed=42,
             output_dir="./ner-checkpoints"):
    import torch
    from scipy.special import softmax
    from seqeval.metrics import classification_report, f1_score
    from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                              DataCollatorForTokenClassification, Trainer, TrainingArguments, set_seed)

    set_seed(seed)
    docs = {nome: ler_jsonl(c) for nome, c in (("train", train), ("dev", dev), ("test", test))}
    labels = sorted({e["label"] for ds in docs.values() for d in ds for e in d["entidades"]})
    label_names = ["O"] + [f"{p}-{l}" for l in labels for p in ("B", "I")]
    tag2id = {t: i for i, t in enumerate(label_names)}
    print("Labels:", labels)

    tokenizer = AutoTokenizer.from_pretrained(modelo)
    trechos, spans, ds = {}, {}, {}
    for nome in docs:
        trechos[nome], spans[nome] = preparar(docs[nome], tokenizer, max_subtokens)
        ds[nome] = tokenizar(trechos[nome], tokenizer, tag2id)
        print(f"  {nome:5} {len(docs[nome])} docs -> {len(trechos[nome])} trechos")

    def decodificar(logits, label_ids):
        pred = np.argmax(logits, axis=-1)
        verd = [[label_names[l] for l in seq if l != -100] for seq in label_ids]
        prev = [[label_names[p] for p, l in zip(ps, ls) if l != -100] for ps, ls in zip(pred, label_ids)]
        return verd, prev

    def metricas(ev):
        verd, prev = decodificar(*ev)
        return {"f1": f1_score(verd, prev)}

    model = AutoModelForTokenClassification.from_pretrained(
        modelo, num_labels=len(label_names),
        id2label=dict(enumerate(label_names)), label2id=tag2id)
    args = TrainingArguments(
        output_dir=output_dir, eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        learning_rate=lr, per_device_train_batch_size=batch, per_device_eval_batch_size=batch,
        num_train_epochs=epocas, weight_decay=0.01, load_best_model_at_end=True,
        metric_for_best_model="f1", seed=seed, report_to=[], fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["dev"],
                      data_collator=DataCollatorForTokenClassification(tokenizer),
                      processing_class=tokenizer, compute_metrics=metricas)
    trainer.train()

    # ---------------- predição no test ----------------
    out = trainer.predict(ds["test"])
    probs = softmax(out.predictions.astype(np.float64), axis=-1)
    n_docs = len(docs["test"])
    tags_doc = [["O"] * len(s) for s in spans["test"]]
    conf_doc = [[0.0] * len(s) for s in spans["test"]]
    for i, t in enumerate(trechos["test"]):
        pos = [j for j, l in enumerate(out.label_ids[i]) if l != -100]   # 1º subtoken de cada palavra
        for k, j in enumerate(pos):
            c = int(np.argmax(probs[i, j]))
            tags_doc[t["doc"]][t["ini"] + k] = label_names[c]
            conf_doc[t["doc"]][t["ini"] + k] = float(probs[i, j, c])

    gold_tags = [doc_para_palavras(d)[1] for d in docs["test"]]
    print("\nseqeval no test (nível de palavra):")
    print(classification_report(gold_tags, tags_doc, digits=3))

    with open(saida, "w", encoding="utf-8") as f:
        for i in range(n_docs):
            d, sp, conf = docs["test"][i], spans["test"][i], conf_doc[i]
            ents = []
            for e in bio_para_entidades(tags_doc[i]):
                cs = conf[e["start"]:e["end"]]
                score = sum(cs) / len(cs) if agregacao == "media" else min(cs)
                ents.append([sp[e["start"]][0], sp[e["end"] - 1][1], e["label"], round(score, 6)])
            f.write(json.dumps({"id": d["id"], "text": d["texto"], "label": ents}, ensure_ascii=False) + "\n")
    print(f"Predições gravadas em {saida} ({n_docs} documentos)")
    return trainer


if __name__ == "__main__":
    main()
