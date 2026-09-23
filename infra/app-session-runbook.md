# App session runbook: the revamped chat app and lab bench on the GPU

This session has one purpose: see the rebuilt app and lab bench working against the real
engine, with the owner in the browser. It measures nothing new and changes no code. Every step
is a command and a check; nothing here needs a judgement call. **If a step needs one, stop and
use the escalation section instead of improvising.**

## Escalate: stop, and tell the owner to switch effort back to high

Stop and hand over, with the exact output, on any of these:

- `ABORT` from `app-run.sh`, `LAUNCH_FAILED` or `LAUNCH_TIMEOUT` from `vllm-launch.sh`, or
  `smoke: N FAILURES`. Grep `infra/vllm-runbook.md`'s troubleshooting table first (incident 40)
  and report what it says; do not debug past that.
- Anything in the browser that needs a code change beyond a one-line typo.
- Any number or behaviour that contradicts what this file says to expect.
- Any failure whose cause is not already written down here or in the vLLM runbook.

The standing rules still bind: `CLAUDE.md` section 3 (the box), systemd units never foreground
ssh, `infra/sync.sh` never hand-typed ssh, and never quote a number without its configuration.

## 0. Bring the box up

```bash
aws ec2 describe-instances --instance-ids i-07d8b10bdcf39a099 \
    --query 'Reservations[0].Instances[0].State.Name' --output text      # expect: stopped
./infra/up.sh
```

**Known failure, not a bug:** `InsufficientInstanceCapacity` in us-east-1c. It cleared on the
second try on 2026-09-22. Retry once a minute, up to 15 times, in a loop that BRANCHES on the
result (incidents 30, 47: `&& break` is not a gate). `up.sh` also exits early with "no instance
found" if the box is still `pending`; wait for `running` with `aws ec2 wait instance-running`
and run `up.sh` again, which re-authorises the security group for the current IP.

Then, once `./infra/sync.sh run true` succeeds:

```bash
./infra/sync.sh run 'touch /opt/llm/.no-autoshutdown'     # REMOVE at the end (incident 23)
./infra/sync.sh push                                       # prints the deployed commit
```

The deployed commit must be the current `git log -1`, not marked dirty.

## 1. Start the stack

```bash
./infra/sync.sh run 'sudo systemd-run --unit=app-up --collect --working-directory=/opt/llm \
    /bin/bash /opt/llm/infra/app-run.sh up'
```

Poll `sudo journalctl -u app-up --no-pager -o cat` through `sync.sh run` every 15 s until it
prints `UP_OK` or `ABORT`, in a background loop with a bound. Model load is 60-105 s and that is
real, not a hang. Expected on the way:

- `GPU KV cache size: 138,528 tokens` (fp8 KV, pinned at 10213733807 bytes)
- `vLLM up: V2 runner off, bare </think>, reasoning arrives in delta.reasoning`
- `app deps: sse-starlette ...` (already installed on the box)
- smoke: off with `think_chars=0`; brief around 400-800 reasoning chars; full longer; `smoke: PASS`

The app's SQLite database on the box predates the revamp. The app upgrades it in place on
start (three added columns); the owner's old chats must still be listed afterwards.

Then the lab bench, on :8081 against the same vLLM:

```bash
./infra/sync.sh run 'sudo /bin/bash /opt/llm/infra/app-run.sh labbench'
```

Expect `LABBENCH_OK`. It takes seconds, so it may run directly rather than in a unit.

## 2. Hand the browser to the owner

Get the IP from `./infra/up.sh`'s output or `describe-instances`, and give the owner exactly:

```
ssh -i ~/.ssh/llm-inference.pem -L 8090:localhost:8090 -L 8081:localhost:8081 ubuntu@<ip>
```

Chat app at http://localhost:8090, lab bench at http://localhost:8081. Never open a security
group rule for these ports.

**Tell the owner: do not use the lab bench's backend switcher or quantization selector.** It
relaunches vLLM with its own flags and would drop `VLLM_USE_V2_MODEL_RUNNER=0`, after which the
three thinking levels silently behave identically. If it happens anyway, rerun step 1's `up`.

Checklist to give the owner, for the chat app:

1. A search-on question shows searching (with the query), source cards, thinking streaming then
   collapsing, then the answer.
2. Tables, bullet lists, bold and code render properly, including while streaming.
3. Off, Brief and Full behave differently: no thinking, short thinking, long thinking.
4. Stop mid-answer keeps the partial answer marked "Stopped early", and the next message works.
5. Edit a message: a branch appears with "1 / 2" arrows. Regenerate: a sibling answer appears.
6. New chat is one click, shows "#id New chat", and gets a title after the first answer.
7. Cream and dark themes; the old chats from 2026-09-22 are still there.

For the lab bench: the GPU panel shows real memory, utilisation, power and temperature (they
were "unavailable" on the laptop), the config header shows the model and KV budget, and TTFT
per turn charts as the conversation grows.

While the owner tests, the GPU is busy only when they send; the hold file keeps the guard off.

## 3. Collect and shut down

When the owner says they are done, collect their feedback verbatim into the reply, then:

```bash
./infra/sync.sh run 'sudo /bin/bash /opt/llm/infra/app-run.sh down'
./infra/sync.sh pull
./infra/sync.sh run 'rm -f /opt/llm/.no-autoshutdown && echo hold-removed'
./infra/down.sh
aws ec2 wait instance-stopped --instance-ids i-07d8b10bdcf39a099   # in the background; it can take minutes
```

Confirm `stopped`, and confirm `hold-removed` was printed BEFORE `down.sh`. Stop, never
terminate.

`sync.sh pull` brings back `results/app-gw.jsonl` and `results/app-smoke.jsonl`. Do not commit
them without asking: this session is not a measurement, and they contain the owner's chats'
timings.

## 4. Report

Short: did it come up, did smoke pass, what the owner said, anything that broke with its exact
output, and the box's final state. No analysis beyond what the owner asks for.
