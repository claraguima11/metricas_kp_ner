# metricas_ner

Biblioteca com as métricas do [pipeline_keyphrase_juridico](https://github.com/claraguima11/pipeline_keyphrase_juridico), separadas do modelo: entra dois JSONL (formato do Doccano), sai tipo de erro, 4 variantes de F1 e métricas de ranking. Serve pra avaliar a saída de qualquer modelo **ou** comparar a anotação de duas pessoas — pra biblioteca tanto faz de onde veio o `pred.jsonl`.

```python
from metricas_ner import avaliar

r = avaliar("gold.jsonl", "pred.jsonl", labels="labels.json")   # F1 estrito e relaxado (+ ranking se tiver score)
r = avaliar("gold.jsonl", "pred.jsonl", juiz="bert")              # + learning-based e estendido
r = avaliar("gold.jsonl", "pred.jsonl", juiz=meu_classificador)   # qualquer função texto -> label
```

O retorno é um dict com `f1`, `ranking`, `mismatches` (contagem por tipo), `totais`, `detalhes` (cada erro, com texto, spans e contexto) e `julgamentos` (decisão do juiz pra cada erro). Tem um exemplo pronto em `exemplos/`.

## Formato de entrada

Um documento por linha, offsets de caractere (`fim` exclusivo) — exatamente o que o Doccano exporta:

```json
{"text": "O Banco do Brasil ajuizou...", "label": [[2, 17, "ORGANIZACAO"]], "meta": {"id_doc": "doc1"}}
```

- também aceita `"entities": [{"start_offset", "end_offset", "label"}]` (export do Doccano com relações);
- no `pred`, uma entidade pode ter score como 4º elemento: `[2, 17, "ORGANIZACAO", 0.93]` (ou `"score"` no formato `entities`). Com score em todas, sai o ranking;
- os documentos dos dois arquivos são pareados por `meta.id_doc`, depois por `id`; se nenhum id bate, pela ordem das linhas. O texto tem que ser idêntico nos dois, senão os offsets não casam;
- labels: `.json` exportado do Doccano (lista de `{"text": "Tipo", ...}`), lista de strings, ou `.txt` com um por linha. É opcional — serve pra avisar quando aparece label que não está na lista.

## Keyphrases como lista de termos

Quando o modelo devolve só uma lista de termos por documento (sem posição no texto):

```json
{"id": "doc1", "keyphrases": ["contrato de financiamento", "alienação fiduciária"]}
{"id": "doc1", "keyphrases": [["contrato de financiamento", 0.91], ["alienação fiduciária", 0.60]]}
```

```python
from metricas_ner import avaliar_keyphrases

avaliar_keyphrases("gold.jsonl", "pred.jsonl")
avaliar_keyphrases("gold.jsonl", {"yake": "yake.jsonl", "keybert": "keybert.jsonl"},
                   maior_score_melhor={"yake": False})   # no YAKE, menor score = melhor
```

Sai P@k, R@k, F1@k, R-precision, MRR e Bpref. Maiúscula/minúscula, espaço extra e pontuação nas bordas não fazem diferença. O gold pode estar nesse formato ou no do Doccano. Exemplo em `exemplos/keyphrases/`.

## Arquivos

| Arquivo | O que tem |
|---|---|
| `metricas_ner/leitura.py` | ler JSONL e labels, parear documentos, encaixar spans, resolver sobreposição |
| `metricas_ner/metricas.py` | mismatches, os 4 F1, R-precision / MRR / Bpref |
| `metricas_ner/juiz.py` | `julgar` (genérico) e `JuizBert` (opcional) |
| `metricas_ner/keyphrases.py` | `avaliar_keyphrases` |
| `metricas_ner/avaliar.py` | `avaliar()`, que junta tudo, e o resumo impresso |
| `exemplos/gerar_predicoes.py` | treina um NER e grava o `pred.jsonl` (fora da biblioteca) |
| `test_metricas.py` | testes (`python -m pytest -q`) |
