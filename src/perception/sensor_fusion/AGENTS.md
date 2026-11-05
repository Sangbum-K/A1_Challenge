# Independent C++ sensor fusion project

The user explicitly requested this new independent folder on 2026-11-03,
overriding the parent rule that all new real-car work belongs in P5Brake.
Read README.md and PROJECT_CONTEXT.md before making changes.
Runtime implementations belong in cpp/. Python is for export, audit and tests.
reference/ and datasets/ are evidence: preserve original bytes. On 2026-11-04, user-authorized cleanup replaced verified duplicate large data directories with links to surviving originals (docs/cleanup/20261104.json). Do not recreate the large copies without a reason.
Do not send vehicle commands. Requests are shadow estimates with validity and expiry.
Do not invent calibrated transforms or treat raw IMU acceleration as velocity.
Keep code tests, real sensor reception, throughput and real-world accuracy separate.
