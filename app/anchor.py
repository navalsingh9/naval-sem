"""
app/anchor.py — Optional public Bitcoin timestamping for run fingerprints.

Why this exists
----------------
NAVAL-SEM computes a local SHA-256 "fingerprint" for every run (see
``_compute_fingerprint`` in app/main.py) covering the model syntax, a hash
of the data, the algorithm, environment info, and key fit results. That
fingerprint is useful on its own for local reproducibility checks, but it
carries no independent proof of *when* it was produced — which matters if
a result is later cited in a paper and someone wants to confirm it wasn't
generated or altered after the fact.

This module adds an OPT-IN way to get that proof via OpenTimestamps (OTS).
To be precise about what "anchored to Bitcoin" actually means here, since
that phrase can misleadingly suggest we run mining infrastructure: we
don't. What happens is closer to a free, decentralized version of RFC
3161 trusted timestamping —

  1. We send only the fingerprint hash (never any data) to a handful of
     free public OTS "calendar" servers.
  2. Those servers batch many users' hashes together into one Merkle tree
     and, periodically, embed that tree's root in a single Bitcoin
     transaction that they pay the fee for (nobody using NAVAL-SEM pays
     anything, and no wallet is needed).
  3. Once *that* transaction is mined — by ordinary miners, for ordinary
     reasons — your fingerprint is now permanently tied to that block's
     timestamp, and anyone can verify this independently with just the
     fingerprint hash and the ``.ots`` proof file, without trusting
     NAVAL-SEM, Anthropic, or the OTS operators.

Implementation note (important, Windows-specific): earlier versions of
this module shelled out to the ``ots`` CLI (``opentimestamps-client``).
That CLI's ``cmds.py`` unconditionally imports ``bitcoin.rpc`` and
``bitcoin.wallet`` — needed only for its local-Bitcoin-node mode, which we
never use — and those modules call ``ctypes.cdll.LoadLibrary`` for
OpenSSL at *import time*. On Windows, where there's no standard
``libssl``/``libcrypto`` on the DLL search path, that import crashes with
a cryptic ``TypeError`` before the CLI ever runs, even for the `stamp`
subcommand we needed. Calendar submission and verification only actually
require the ``opentimestamps`` core library (``opentimestamps.calendar``,
``opentimestamps.core.*``), which never touches ``bitcoin.rpc``/
``bitcoin.wallet``. This module talks to calendar servers directly via
that lower-level API instead, sidestepping the crash entirely — no
OpenSSL DLL installation required on any platform.

This is entirely optional and is skipped by default so the app keeps
working fully offline. It's only invoked when a caller explicitly asks
for it (``anchor=True`` on POST /run), and any failure (no internet, no
calendar servers reachable, timeout) is caught and reported as a
non-fatal status rather than breaking the underlying analysis run.
"""

from __future__ import annotations

import base64
from typing import List, Optional

try:
    from opentimestamps.calendar import RemoteCalendar
    from opentimestamps.core.notary import BitcoinBlockHeaderAttestation, PendingAttestation
    from opentimestamps.core.op import OpSHA256
    from opentimestamps.core.serialize import BytesDeserializationContext, BytesSerializationContext
    from opentimestamps.core.timestamp import DetachedTimestampFile, Timestamp

    _IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover — environment-dependent
    _IMPORT_ERROR = _e

# Same public calendar servers the official `ots` CLI defaults to.
_DEFAULT_CALENDARS: List[str] = [
    "https://a.pool.opentimestamps.org",
    "https://b.pool.opentimestamps.org",
    "https://a.pool.eternitywall.com",
    "https://ots.btc.catallaxy.com",
]

# Require at least this many calendar servers to accept the submission
# before calling it "pending" rather than "failed" — mirrors the CLI's
# own redundancy requirement so one flaky server doesn't sink a run.
_MIN_CALENDARS_FOR_PENDING = 2
_REQUEST_TIMEOUT_SECONDS = 15


def opentimestamps_available() -> bool:
    """Whether the pure-Python opentimestamps library is usable."""
    return _IMPORT_ERROR is None


def _fingerprint_digest(fingerprint_hex: str) -> bytes:
    """
    The exact bytes we timestamp: SHA-256 of the fingerprint's hex text
    (UTF-8), mirroring what `ots stamp <file>` would compute over a file
    containing that text. Keep this in sync everywhere it's used — a
    proof is only valid for exactly this digest.
    """
    return OpSHA256()(fingerprint_hex.encode("utf-8"))


def _serialize_dtf(dtf: "DetachedTimestampFile") -> bytes:
    ctx = BytesSerializationContext()
    dtf.serialize(ctx)
    return ctx.getbytes()


def _deserialize_dtf(data: bytes) -> "DetachedTimestampFile":
    ctx = BytesDeserializationContext(data)
    return DetachedTimestampFile.deserialize(ctx)


def _walk_pending(ts: "Timestamp"):
    """Recursively yield (subtree, PendingAttestation) pairs in a Timestamp tree."""
    for att in ts.attestations:
        if isinstance(att, PendingAttestation):
            yield (ts, att)
    for sub in ts.ops.values():
        yield from _walk_pending(sub)


def _has_bitcoin_attestation(ts: "Timestamp") -> bool:
    return any(isinstance(att, BitcoinBlockHeaderAttestation) for _msg, att in ts.all_attestations())


def stamp_fingerprint(fingerprint_hex: str, calendars: Optional[List[str]] = None) -> dict:
    """
    Submit a fingerprint hash to public OpenTimestamps calendar servers.

    Does NOT touch any wallet, does NOT spend any money, and does not run
    or wait for mining — it just asks free calendar servers to include
    this hash in their next batch. Returns a status dict; never raises
    (all failure modes — missing library, no network, calendar timeout —
    are captured in the returned dict).
    """
    if not opentimestamps_available():
        return {
            "status": "unavailable",
            "detail": f"opentimestamps library is not usable on this server: {_IMPORT_ERROR}",
        }

    calendars = calendars or _DEFAULT_CALENDARS
    digest = _fingerprint_digest(fingerprint_hex)
    file_hash_op = OpSHA256()
    dtf = DetachedTimestampFile(file_hash_op, Timestamp(digest))

    successes = 0
    errors = []
    for url in calendars:
        try:
            resp_ts = RemoteCalendar(url).submit(digest, timeout=_REQUEST_TIMEOUT_SECONDS)
            dtf.timestamp.merge(resp_ts)
            successes += 1
        except Exception as e:
            errors.append(f"{url}: {e}")

    if successes < _MIN_CALENDARS_FOR_PENDING:
        return {
            "status": "timeout" if successes == 0 else "failed",
            "detail": ("; ".join(errors)[-500:] or "No calendar servers responded.") +
                       (" This is expected if this machine has no internet access right now "
                        "— the analysis itself is unaffected." if successes == 0 else ""),
        }

    proof_bytes = _serialize_dtf(dtf)
    return {
        "status": "pending",
        "detail": (
            f"Submitted to {successes}/{len(calendars)} public OpenTimestamps calendar servers. "
            "Pending Bitcoin confirmation (usually a few hours to ~1 day); call the upgrade "
            "endpoint later to finalize a fully self-contained proof."
        ),
        "stamped_text": fingerprint_hex,
        "ots_proof_b64": base64.b64encode(proof_bytes).decode("ascii"),
    }


def upgrade_proof(fingerprint_hex: str, ots_proof_b64: str) -> dict:
    """
    Attempt to upgrade a pending proof to a complete, locally-verifiable
    one (i.e. check whether the Bitcoin transaction has confirmed yet).
    Safe to call repeatedly; a no-op reply if still pending.
    """
    if not opentimestamps_available():
        return {"status": "unavailable", "detail": f"opentimestamps library is not usable: {_IMPORT_ERROR}"}

    try:
        dtf = _deserialize_dtf(base64.b64decode(ots_proof_b64))
    except Exception as e:
        return {"status": "failed", "detail": f"Could not parse existing proof: {e}"}

    if dtf.file_digest != _fingerprint_digest(fingerprint_hex):
        return {"status": "failed", "detail": "Proof does not match this run's fingerprint."}

    pending = list(_walk_pending(dtf.timestamp))
    if not pending:
        confirmed = _has_bitcoin_attestation(dtf.timestamp)
        return {
            "status": "confirmed" if confirmed else "pending",
            "detail": "No pending calendar attestations remain." if confirmed
                       else "No pending attestations found, but no Bitcoin confirmation yet either.",
            "ots_proof_b64": ots_proof_b64,
        }

    upgraded_any = False
    errors = []
    for node, att in pending:
        try:
            url = att.uri if att.uri.startswith("http") else f"https://{att.uri}"
            new_ts = RemoteCalendar(url).get_timestamp(node.msg, timeout=_REQUEST_TIMEOUT_SECONDS)
            node.merge(new_ts)
            upgraded_any = True
        except Exception as e:
            errors.append(f"{att.uri}: {e}")

    proof_bytes = _serialize_dtf(dtf)
    confirmed = _has_bitcoin_attestation(dtf.timestamp)
    return {
        "status": "confirmed" if confirmed else "pending",
        "detail": ("Bitcoin confirmation found — proof is now complete and self-contained."
                    if confirmed else
                    ("Still pending Bitcoin confirmation." if upgraded_any
                     else ("; ".join(errors)[-500:] or "No update available yet."))),
        "ots_proof_b64": base64.b64encode(proof_bytes).decode("ascii"),
    }


def verify_proof(fingerprint_hex: str, ots_proof_b64: str) -> dict:
    """Verify a (possibly upgraded) proof against the original fingerprint hex."""
    if not opentimestamps_available():
        return {"status": "unavailable", "detail": f"opentimestamps library is not usable: {_IMPORT_ERROR}"}

    try:
        dtf = _deserialize_dtf(base64.b64decode(ots_proof_b64))
    except Exception as e:
        return {"status": "unverified", "detail": f"Could not parse proof: {e}"}

    if dtf.file_digest != _fingerprint_digest(fingerprint_hex):
        return {"status": "unverified", "detail": "Proof does not match this fingerprint."}

    confirmed = _has_bitcoin_attestation(dtf.timestamp)
    return {
        "status": "verified" if confirmed else "unverified",
        "detail": "Bitcoin attestation present and matches this fingerprint." if confirmed
                   else "No confirmed Bitcoin attestation yet — proof may still be pending.",
    }
