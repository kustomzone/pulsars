# pulsar

An inference engine for giant Mixture-of-Experts models on hardware that
has no business running them. Successor to
[NeutronStar](https://github.com/giannisanni/neutronstar): same thesis
(the routed experts live on disk and stream per token; everything that
makes decisions stays resident), rebuilt as its own engine instead of a
fork, with no llama.cpp anywhere in the stack.

Working name. A pulsar is a neutron star that spins fast and emits beams.

## Why a new engine

NeutronStar proved the numbers on a single RTX 4060 Ti 16GB: Hy3 295B at
~2.2 tok/s decode and ~6 tok/s batch prefill, GLM-5.2 743B end to end.
Every hard bug on the way lived in one layer: concurrent I/O
orchestration - buffer ownership across fetch threads, cache lifetime
races, fire-and-forget ring completions. That layer is being rebuilt in
Rust, where those bug classes fail at compile time. The GPU kernels
(GQA attention, IQ2 expert tiles, dp4a MoE) stay CUDA C++ behind a thin
FFI - as they do in every engine - derived from the ds4 lineage with
attribution.

## Architecture (planned)

- `crates/gguf` - zero-copy GGUF reader: header, metadata, tensor table;
  tensor data is never touched at parse time. DONE, tested against the
  production Hy3 295B header.
- `crates/stream` - the expert streaming core: io_uring fetch engine,
  LFU host cache with persistent warm state, cross-layer speculative
  prefetch. The design is the measured-and-proven NeutronStar pipeline,
  with ownership made explicit. Fetch engine DONE and benched.
- `crates/kernels` - FFI to the CUDA kernel library (build-time nvcc).
- `crates/engine` - model graphs (GQA+MoE first: hy-v3, then
  deepseek/GLM-family MLA), scheduler, multi-GPU expert residency.
- `crates/serve` - CLI + OpenAI-compatible server.

## Milestones

1. GGUF reader against production headers. (done)
2. Disk path parity. (done) Rust io_uring fetcher vs a minimal C
   liburing reference, byte-identical plans (3000 random expert slabs,
   4.55 GiB, from the production Hy3 gguf), O_DIRECT, QD 64, matching
   checksums: C 4.83 GB/s, Rust 4.82 GB/s on a Gen4 NVMe. The language
   costs nothing on the path that is this engine's entire thesis. Bonus
   finding: the raw fetch pattern saturates the drive at ~4.8 GB/s while
   the C engine's in-decode effective feed measures 2.5-2.9 GB/s, so
   ~2 GB/s is currently lost to engine-side serialization - that gap is
   pulsar's headroom.
3. Single-GPU Hy3 decode parity on the 4060 Ti via FFI kernels. (done)
   Full hy-v3 forward graph in `crates/engine` over the ds4-lineage
   kernel set (GQA attention, dp4a q8_0 matmul with activation prequant,
   q8_K integer expert dots, sigmoid router) plus a from-gguf BPE
   tokenizer with ds4 gold-vector tests. Parity gate: teacher-forced
   along ds4's greedy path, 15/16 per-position argmax agreement - the
   one miss sits at ds4's 0.086-logit top1/top2 tie, and pulsar agrees
   at ds4's 0.013 and 0.002 ties. Decode: 2.01 tok/s streaming
   (three-tier expert path: VRAM hot-set cache with touch-count
   admission, LFU host cache, io_uring reads overlapping H2D uploads)
   vs ds4's 0.64-0.70 on the same box. Remaining niceties tracked
   separately: batched prefill, cross-layer prefetch.
   Decode-vs-prefill consistency holds by construction for now: prefill
   runs the same single-token forward as decode.
4. Multi-GPU expert residency on 2x RTX 5060 Ti (the reason this engine
   exists: ~48GB VRAM of resident experts, PCIe P2P where unlockable).
5. Own quantizer: BF16 -> uniform-slab expert quants without llama.cpp.
