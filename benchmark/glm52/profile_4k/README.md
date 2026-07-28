# GLM-5.2 4K Prefill profile

This directory records the exact four-Pod PD launch and profiling commands.

Pod mapping:

- `glm52-sglang-03`: Prefill rank 0, `192.168.4.3`
- `glm52-sglang-05`: Prefill rank 1, `192.168.4.5`
- `glm52-sglang-07`: Decode rank 0, `192.168.4.7`
- `glm52-sglang-10`: Decode rank 1, `192.168.4.10`

The launch scripts run inside their corresponding Pod from `/tmp/sglang`.
The router and benchmark run in `glm52-sglang-03`. Profile traces are written
to `/tmp/glm52-profile-4k-b1/traces` on both Prefill Pods because each node
owns eight profiler ranks.
