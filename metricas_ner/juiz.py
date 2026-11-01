"""Automatic Judge (Nejadgholi et al., 2020)."""
import random
import re
from collections import Counter

from .leitura import texto_do_span
from .metricas import TIPOS


def julgar(detalhes, juiz, imprimir=True):
    """aceito se o juiz der ao texto previsto a mesma categoria do pred; type-2 não tem o que julgar"""
    resultados = []
    for d in detalhes:
        if d["tipo"] == "type-2":
            resultados.append({**d, "label_juiz": None, "confianca_juiz": None,
                               "decisao": "nao_avaliavel",
                               "motivo": "Não há span previsto para o juiz classificar."})
            continue
        saida = juiz(d["pred_texto"])
        label, conf = saida if isinstance(saida, tuple) else (saida, None)
        if label == d["pred_label"]:
            decisao, motivo = "aceito", "Juiz confirmou a classe prevista."
        elif label == d["gold_label"]:
            decisao, motivo = "rejeitado", "Juiz confirmou a classe gold, não a prevista."
        else:
            decisao, motivo = "rejeitado", "Juiz não confirmou a classe prevista."
        resultados.append({**d, "label_juiz": label, "confianca_juiz": conf,
                           "decisao": decisao, "motivo": motivo})
    if imprimir:
        for tipo in TIPOS:
            c = Counter(r["decisao"] for r in resultados if r["tipo"] == tipo)
            if c:
                print(f"  juiz {tipo}: {dict(c)}")
    return resultados

def _limpar_chunk(texto):
    texto = re.sub(r"^[^\wÀ-ÿ]+|[^\wÀ-ÿ]+$", "", texto.strip()).strip()
    if len(texto) < 3 or not any(c.isalpha() for c in texto):
        return None
    return texto


class JuizBert:
    """classificador de spans com classe "other", treinado nas entidades do gold"""

    def __init__(self, modelo="neuralmind/bert-base-portuguese-cased", spacy_model="pt_core_news_lg",
                 seed=42):
        import spacy
        from transformers import AutoTokenizer
        self.modelo_nome = modelo
        self.seed = seed
        self.nlp = spacy.load(spacy_model)
        self.tokenizer = AutoTokenizer.from_pretrained(modelo)
        self.modelo = None
        self.id2label = None

    def _pares(self, docs, rng):
        """(texto, label) do gold + noun chunks do spaCy como other"""
        positivos, other = [], []
        for d in docs:
            texto, ents = d["texto"], d["entidades"]
            positivos += [(texto_do_span(texto, e), e["label"]) for e in ents]
            for chunk in self.nlp(texto).noun_chunks:
                if any(chunk.start_char < e["end"] and e["start"] < chunk.end_char for e in ents):
                    continue
                limpo = _limpar_chunk(chunk.text)
                if limpo:
                    other.append(limpo)
        other = list(dict.fromkeys(other))
        contagem = Counter(l for _, l in positivos)
        limite = round(sum(contagem.values()) / len(contagem)) if contagem else 0
        pares = positivos + [(t, "other") for t in rng.sample(other, k=min(limite, len(other)))]
        rng.shuffle(pares)
        return pares

    def treinar(self, docs_gold, prop_val=0.2, epocas=3, lr=2e-5, batch_size=16,
                output_dir="./juiz-checkpoints"):
        import numpy as np
        from datasets import Dataset
        from sklearn.metrics import accuracy_score, f1_score
        from transformers import (AutoModelForSequenceClassification, DataCollatorWithPadding,
                                  Trainer, TrainingArguments, set_seed)

        set_seed(self.seed)
        rng = random.Random(self.seed)
        docs = [d for d in docs_gold if d["texto"]]
        rng.shuffle(docs)
        n_val = max(1, round(len(docs) * prop_val)) if len(docs) > 1 else 0
        pares_val, pares_treino = self._pares(docs[:n_val], rng), self._pares(docs[n_val:], rng)
        print("Juiz — treino:", dict(Counter(l for _, l in pares_treino)))

        classes = sorted({l for _, l in pares_treino})
        self.id2label = dict(enumerate(classes))
        label2id = {l: i for i, l in self.id2label.items()}
        pares_val = [p for p in pares_val if p[1] in label2id]

        def ds(pares):
            d = Dataset.from_dict({"text": [t for t, _ in pares], "label": [label2id[l] for _, l in pares]})
            return d.map(lambda ex: self.tokenizer(ex["text"], truncation=True), batched=True)

        def metricas(ev):
            logits, labels = ev
            p = np.argmax(logits, axis=-1)
            return {"accuracy": accuracy_score(labels, p), "f1": f1_score(labels, p, average="macro")}

        modelo = AutoModelForSequenceClassification.from_pretrained(
            self.modelo_nome, num_labels=len(classes), id2label=self.id2label, label2id=label2id)
        tem_val = len(pares_val) > 0
        args = TrainingArguments(
            output_dir=output_dir, learning_rate=lr, num_train_epochs=epocas, weight_decay=0.01,
            per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
            eval_strategy="epoch" if tem_val else "no", save_strategy="epoch" if tem_val else "no",
            load_best_model_at_end=tem_val, metric_for_best_model="f1" if tem_val else None,
            seed=self.seed, report_to=[],
        )
        trainer = Trainer(
            model=modelo, args=args, train_dataset=ds(pares_treino),
            eval_dataset=ds(pares_val) if tem_val else None,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            processing_class=self.tokenizer, compute_metrics=metricas if tem_val else None,
        )
        trainer.train()
        self.modelo = trainer.model
        return trainer

    def __call__(self, texto):
        import torch
        inputs = self.tokenizer(texto, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.modelo.device) for k, v in inputs.items()}
        self.modelo.eval()
        with torch.no_grad():
            probs = torch.softmax(self.modelo(**inputs).logits, dim=-1)[0]
        i = int(probs.argmax())
        return self.id2label[i], float(probs[i])
