# Training phase 3: Progressive Context Curriculum

## Estado

La fase de pretraining queda cerrada hasta 8K. El post-training (SFT, RL y
distillation) permanece fuera de alcance deliberadamente.

El curriculum es opcional y está desactivado por defecto. Al activarlo usa
umbrales de tokens vistos y la secuencia canónica:

```text
512 -> 1024 -> 2048 -> 4096 -> 8192
```

## Punto de entrada

El punto de entrada para CLI o Jupyter es exclusivamente:

```python
from training import train_kimiK3
```

Para PCC con reconstrucción segura del dataloader se pasa una factory:

```python
result = train_kimiK3(
    model=model,
    train_loader_factory=lambda max_seq_len: make_loader(
        max_seq_len=max_seq_len
    ),
    training_config=training_config,
    context_curriculum_config=context_curriculum_config,
)
```

Se pasa `train_loader` o `train_loader_factory`, nunca ambos.
La factory recibe siempre la longitud activa. Si usa sampler con cursor,
streaming, bucketing o workers persistentes, ella conserva/restaura su cursor;
el trainer conserva modelo, optimizer, scheduler, scaler, EMA, Muon y biases
de Quantile Balancing.

Cuando la duración del loader cambia entre etapas conviene pasar
`total_steps` explícitamente para que el cosine scheduler represente el
presupuesto completo y no sólo el loader inicial.

## Contratos de seguridad

- La transición ocurre únicamente después de un `optimizer.step` completo.
- Una actualización puede cruzar varios umbrales sin quedar en una etapa
  incoherente.
- El collator trunca y paddea sólo hasta el máximo necesario del batch.
- `input_ids`, labels, masks y campos secuenciales permanecen alineados.
- Sólo se admite padding derecho.
- Un batch packed con más de un documento por fila falla explícitamente:
  KDA/MLA todavía no implementan atención aislada por segmento.
- La truncación multimodal falla si separaría pixels/frames de sus
  placeholders.
- Con MTP se valida tanto la longitud mínima de la etapa como los tokens
  válidos de cada muestra.
- Resume valida la definición completa de stages y reconstruye el loader a la
  longitud restaurada.

## Checkpoint y observabilidad

El checkpoint guarda `context_curriculum` con enabled, stage, longitud activa,
tokens vistos, contador de transiciones, stages y límites operacionales.
También conserva la clave histórica `curriculum_state_dict` para lectores
anteriores.

La consola/Jupyter muestra un bloque separado `Progressive context` con stage,
longitud, tokens, padding, throughput, tiempo, memoria y detalle de cada
transición. Las métricas de visión están separadas bajo `vision/`.

## Auditoría del training checkpoint

Verificado para Stage A-C:

- core engine, accumulation, evaluación, AMP, clipping, EMA, RNG y resume;
- AdamW, Muon, Per-Head Muon, parameter groups y cosine warmup;
- NTP textual/multimodal, masking, visual tokens y MTP;
- Quantile Balancing y métricas MoE;
- diagnósticos KDA y paridad recurrent/chunkwise después de actualizar pesos;
- PCC por tokens, reconstrucción de loader y resume exacto;
- checkpoint integral y previews de next-token;
- overfit diminuto textual y multimodal.

Pendiente intencionalmente para otra fase:

- Stage D: SFT;
- Stage E: verifiable RL proxy;
- Stage F: multi-teacher distillation.

## Tests focales

Los tests de `tests/training/context_curriculum/` cubren configuración,
transiciones, acumulación, batching, multimodalidad, checkpoint/resume,
métricas y una ejecución real de Kimi K3 en CPU. No requieren dimensiones
canónicas ni GPU.
