# RTSP Desktop Preparation Progress

## START

- Scope: prepare Desktop for a future Hikvision RTSP input without accessing the physical camera.
- No firewall, route, IP forwarding, NAT, VPN, public listener, model training, raw-data change or credential logging.

## SOURCE_ABSTRACTION

- Completed: `VideoSource` supports USB index, file path and RTSP/profile resolution.
- Completed: source states, redacted URL diagnostics, health counters, capped reconnect and optional low-latency buffer request.

## STREAM_HEALTH

- Completed: source health model and bounded latest-frame reader.
- Completed: source stale state is distinct from `POSE_LOST`.

## RUNTIME

- Completed: source-agnostic frozen V4 entry point and RTSP_DESKTOP/RTSP_BALANCED presets.
- Completed: configurable dual-resolution inference copy; original frame remains available to the runtime/UI.

## SECURITY_NETWORK

- Completed: `.env.example`, secret-free Hikvision profiles, network/gateway/relay documentation.
- Not performed: network configuration, relay installation or firewall changes.

## MULTI_PERSON_SEATS

- Completed: Desktop/Acer integration contract and seat-zone schema only.
- Not performed: tracker merge or real seat ROI calibration.

## DIAGNOSTICS

- Completed: bounded no-recording source diagnostic and bandwidth estimator.
- Camera result: not tested because the physical camera is not available in this run.

## TESTS

- Completed: mock tests for parsing, redaction, source states, reconnect, stale buffer, bandwidth, resizing, dashboard compatibility and old-session missing fields.
- Completed: full test suite `109 passed`.

## FINAL

- Completed: full test suite, raw/model integrity verification, documentation and final report.
- Physical camera/network result remains intentionally **NOT TESTED**.
- C: free space after preparation: approximately 21.28 GB.
- Versioned code-only Acer bundle v4 created and safety-checked; earlier bundles were not overwritten.
