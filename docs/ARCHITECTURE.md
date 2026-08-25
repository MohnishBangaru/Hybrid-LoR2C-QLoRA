# Architecture

`lor2c` follows a strict hexagonal (ports and adapters) layout. Dependencies point inward only:

```
CLI / YAML  ->  settings  ->  application  ->  domain
                              ^
infrastructure (torch hooks, quantization, Hugging Face, W&B, disk)
```

| Layer | Package | Role (one sentence) | May import |
|-------|---------|---------------------|------------|
| Domain | `lor2c.domain` | Low-rank adapter mathematics, residual routing rules, prompt/label/caption rules, early stopping. | `torch`, `pydantic`, stdlib |
| Application | `lor2c.application` | Use-cases (`CausalTrainingService`, `VisionTrainingService`) that orchestrate ports without making modelling decisions. | domain, settings |
| Settings | `lor2c.settings` | Validated, immutable run configuration crossing the CLI boundary. | domain constants |
| Infrastructure | `lor2c.infrastructure` | Port implementations: forward hooks, eager-mode QAT, disk persistence, JSON logging, W&B, Hugging Face loaders/trainers. | everything inward |
| CLI | `lor2c.cli` | Argument parsing and the composition root (`ServiceFactory`). | everything |

## Design decision: `torch` in the domain

The domain's subject matter *is* tensor algebra (`LowRankAdapter`, `ResidualRouter`). Treating
`torch` as the mathematical substrate (the way a pricing domain treats `decimal`) keeps the domain
honest and testable on CPU in milliseconds. What the domain must never import: `transformers`,
`peft`, `datasets`, `wandb`, `PIL`, file systems, environment variables, logging. Those live only
under `lor2c.infrastructure`.

## LoR2C mechanism

```
decoder layer i:   x_i ──► block_i ──► h_i ──(+)──► h'_i
                   │                          ▲
                   └── floor{i+1}: up(down(x_i)) * alpha/r
```

1. `ResidualSchedule.per_layer(depth)` produces one `ScheduleEntry` per decoder layer
   (`start == end == i`, name `floor{i+1}`); spanning entries (`start < end`) are supported.
2. `AdapterBank` holds one `LowRankAdapter` per entry.
3. `ResidualRouter.route(layer, hidden_in, hidden_out)` captures deltas at start layers and adds
   them at end layers; pending deltas are bounded by the schedule size.
4. `HookRouter` (infrastructure) registers a forward hook on every decoder layer found by a
   `LayerLocator` and delegates to the router; hooks are removed when the context exits.
5. Attention LoRA on `q_proj`/`v_proj` is applied by `peft` (`HubCausalModelPort.adapt`) and is
   independent of the residual bank.
6. `HookRouter.attach` also registers the bank as `model.lor2c` so any optimizer built from
   `model.parameters()` trains the residual adapters; the attribute is removed on exit.

## IMLoR2C (merge / inject)

- `FeatureSpaceShape` (domain) scores each adapter by the spectrum of `up @ down` (top-k share or
  mean singular value, as in the reference implementation).
- `AdaptationPlanner` (domain) picks the least concentrated adjacent pair to merge and the most
  concentrated single-layer adapter to inject; `AdaptationTimeline` spreads the events over the
  first quarter of training (`epochs / 4 / count` interval).
- `AdaptationController` (application) observes fractional epoch progress through the
  `Observer` port, mutates the `ResidualSchedule` (immutable value object, replaced via
  `ResidualRouter.reschedule`), re-keys the bank, and asks the `AttentionGate` port to release the
  attention LoRA of an injected layer. The Hugging Face adapter forwards `TrainerCallback.on_step_end`
  to the observer.
- Injection requires the attention LoRA to start frozen (`AttentionGate.freeze`), matching the
  reference `LoRAFreezeCallback`.
- Scoring direction: the paper writes SFS as `1 - top_k/total` and merges the pair with the
  minimum SFS sum; the reference code computes the proportion `top_k/total` and merges the pair
  with the minimum proportion sum (injecting the maximum). This package follows the reference
  code, which produced the published numbers.

## Ports

| Port | Implemented by | Notes |
|------|----------------|-------|
| `CausalModelPort` | `HubCausalModelPort` | `AutoModelForCausalLM` + `peft.LoraConfig` |
| `CausalDataPort` | `HubCausalDataPort` | `datasets` + `InstructionTokenizer` |
| `CausalTrainerPort` | `HubCausalTrainerPort` | `transformers.Trainer` |
| `Router` | `HookRouter` | forward hooks, context managed |
| `Quantizer` | `FakeQuantizer` | `torch.ao.quantization` eager QAT |
| `Repository` | `DiskRepository` | `model/` (peft or trainable state) + `residual/` |
| `Tracker` | `LogTracker`, `WandbTracker` | selected by `tracking.kind` |
| `Seeder` | `TorchSeeder` | Python + torch generators |
| `Observer` | `AdaptationController` (application) | fed by a `TrainerCallback` |
| `AttentionGate` | `PatternAttentionGate` | parameter-name based freeze/release |
| `VisionModelPort` | `HubVisionModelPort` | `AutoModelForVision2Seq` + `LinearInjector` |
| `VisionDataPort` | `HubVisionDataPort` | `CaptionDataset` |
| `VisionTrainerPort` | `TorchVisionTrainerPort` | AMP loop with warmup schedule |
| `Evaluator` | `BleuEvaluator` | beam search + smoothed corpus BLEU |

Every port is a `typing.Protocol`; tests exercise the services with in-memory fakes
(`tests/application/fakes.py`).

## Determinism and idempotency

- The seed is injected (`settings.seed`) and applied by `Seeder`; adapter initialisation takes an
  explicit `torch.Generator`.
- `DiskRepository.save` is idempotent: rerunning overwrites the same files.
- Training resumption is delegated to `transformers.Trainer` via `settings.resume`.

## Observability

Outer layers log JSON lines (`LoggingConfigurator`); fields prefixed `ctx_` in `extra` are
promoted to top-level keys. The domain never logs.

## Output layout

```
outputs/<run>/
  checkpoints/        transformers.Trainer checkpoints (causal only)
  model/              peft adapter (adapter_config.json, adapter_model.safetensors) or adapters.pt
  residual/           adapters.pt (AdapterBank state) + manifest.json
```

## Extending

- New host architecture: implement `LayerLocator` if `model.layers` is not a `ModuleList`.
- New tracker: implement `Tracker`, register it in `ServiceFactory`.
- New rank policy: implement `RankPolicy` (domain) and select it in `ServiceFactory`.
- New quantization scheme: implement `Quantizer` (must keep `Branch` semantics).
