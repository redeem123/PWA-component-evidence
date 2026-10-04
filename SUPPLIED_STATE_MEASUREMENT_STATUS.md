# Supplied-state measurement status

The [public package](https://github.com/redeem123/PWA-component-evidence/releases/tag/supplied-state-measurement-v1)
contains one frozen SAC checkpoint and a saved D30 prefix for five prescribed
routing arms. Anonymous inspection passed on Mac and fresh hosted Linux. The
fixed archive and manifest identities are in the release notes.

The [Linux preflight](https://github.com/redeem123/PWA-component-evidence/actions/runs/37170312490)
fails its unchanged exact-coefficient check. A separate
[arithmetic observation](https://github.com/redeem123/PWA-component-evidence/actions/runs/37171016380)
finds the same saved observation and tested keyed draws, but different float32
linear-layer results on Darwin arm64 and Linux x86_64. The maximum decoded
coefficient difference is approximately 1.19e-7. The particular low-level kernel
cause has not been isolated.

`measurement_arithmetic_diagnosis_v1.json` binds both retained observations and
reports the measured differences. Success of the diagnostic job means that the
observation was saved, not that the numerical replay gate passed. No fresh native
objective call, training transition or author reservation was made. External
measurement replay and reviewer closure remain incomplete. The package, source
pins and original numerical checks have not been changed to pass the gate.
