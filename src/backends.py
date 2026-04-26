from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from src.metrics import estimate_probs_from_counts

# BlueQubit mps.cpu documents a maximum two-qubit gate count (order ~10k).
_DEFAULT_MPS_MAX_CX = 10_000


def _count_cx(qc) -> int:
    return int(qc.count_ops().get("cx", 0))


def _transpile_for_bluequbit_mps(qc):
    level = int(os.getenv("BQ_MPS_OPT_LEVEL", "3"))
    seed = int(os.getenv("BQ_MPS_SEED", "42"))
    return transpile(
        qc,
        basis_gates=["u", "cx"],
        optimization_level=level,
        seed_transpiler=seed,
        coupling_map=None,
    )


def _select_mps_device(chi_max: int | None) -> str:
    """Default policy: always CPU unless explicitly overridden."""
    explicit = os.getenv("BQ_MPS_DEVICE")
    if explicit:
        return explicit
    return "mps.cpu"


@dataclass
class BackendResult:
    probs: np.ndarray | None
    counts: dict[str, int] | None
    backend_used: str
    shots: int
    requested_backend: str
    fell_back_to_aer: bool
    note: str


class BackendAdapter:
    """Unified adapter for local Aer + optional BlueQubit."""

    _shared_bq = None
    _shared_has_bq = None

    def __init__(self) -> None:
        if BackendAdapter._shared_has_bq is None:
            BackendAdapter._shared_bq = None
            BackendAdapter._shared_has_bq = False
            if os.getenv("BQ_DISABLE", "0") != "1":
                try:
                    import bluequbit  # type: ignore

                    endpoint = os.getenv("BLUEQUBIT_MAIN_ENDPOINT")
                    if endpoint:
                        os.environ["BLUEQUBIT_MAIN_ENDPOINT"] = endpoint
                    token = os.getenv("BQ_TOKEN") or os.getenv("BLUEQUBIT_API_TOKEN")
                    if token:
                        BackendAdapter._shared_bq = bluequbit.init(api_token=token)
                    else:
                        BackendAdapter._shared_bq = bluequbit.init()
                    BackendAdapter._shared_has_bq = True
                except Exception:
                    BackendAdapter._shared_has_bq = False
        self._bq = BackendAdapter._shared_bq
        self._has_bq = bool(BackendAdapter._shared_has_bq)

    def statevector_probs(self, circuit, prefer_bq: bool = False) -> BackendResult:
        requested = "bluequbit_sv" if prefer_bq else "aer_statevector"
        if prefer_bq and self._has_bq:
            try:
                result = self._bq.run(circuit, device="cpu", shots=0)
                sv = np.array(result.get_statevector())
                return BackendResult(
                    probs=np.abs(sv) ** 2,
                    counts=None,
                    backend_used="bluequbit_sv",
                    shots=0,
                    requested_backend=requested,
                    fell_back_to_aer=False,
                    note="",
                )
            except Exception as e:
                if os.getenv("BQ_STRICT", "0") == "1":
                    raise RuntimeError(f"BlueQubit SV requested but failed: {e}") from e
                note = f"BlueQubit SV failed, falling back to Aer: {e}"
                print(f"[BackendAdapter] WARNING: {note}")
        else:
            if prefer_bq and os.getenv("BQ_STRICT", "0") == "1":
                raise RuntimeError("BlueQubit SV requested in strict mode, but BQ is unavailable")
            note = "BlueQubit SV unavailable, using Aer"
            if prefer_bq:
                print(f"[BackendAdapter] WARNING: {note}")
        qc = circuit.copy()
        qc.save_statevector()
        sim = AerSimulator(method="statevector")
        sv = sim.run(transpile(qc, sim), shots=1).result().get_statevector()
        return BackendResult(
            probs=np.abs(np.array(sv)) ** 2,
            counts=None,
            backend_used="aer_statevector",
            shots=0,
            requested_backend=requested,
            fell_back_to_aer=prefer_bq,
            note=note if prefer_bq else "",
        )

    def sample_probs(
        self,
        circuit,
        shots: int = 100_000,
        prefer_mps: bool = False,
        chi_max: int | None = None,
        dense_output: bool = True,
    ) -> BackendResult:
        requested = "bluequbit_mps" if prefer_mps else "aer_qasm"
        if prefer_mps and self._has_bq:
            try:
                try:
                    submit = _transpile_for_bluequbit_mps(circuit)
                    cx = _count_cx(submit)
                except Exception:
                    # Some qiskit builds cannot synthesize `state_preparation` during transpile.
                    # In that case, submit the original circuit and let backend-side validation decide.
                    submit = circuit
                    cx = -1
                max_cx = int(os.getenv("BQ_MPS_MAX_CX", str(_DEFAULT_MPS_MAX_CX)))
                primary = _select_mps_device(chi_max)
                devices = [primary]
                if (
                    os.getenv("BQ_MPS_FALLBACK_GPU", "1") == "1"
                    and primary == "mps.cpu"
                    and "mps.gpu" not in devices
                ):
                    devices.append("mps.gpu")

                last_err: Exception | None = None
                for dev in devices:
                    if dev == "mps.cpu" and cx >= 0 and cx > max_cx:
                        last_err = RuntimeError(
                            f"MPS circuit has {cx} CX gates after transpile; "
                            f"BlueQubit {dev} allows at most {max_cx}. "
                            "Reduce n, lower BQ_MPS_OPT_LEVEL, or set BQ_MPS_DEVICE=mps.gpu "
                            "if your account supports it."
                        )
                        continue
                    kwargs: dict = {"device": dev, "shots": shots}
                    if chi_max is not None:
                        kwargs["options"] = {"mps_bond_dimension": int(chi_max)}
                    try:
                        result = self._bq.run(submit, **kwargs)
                        counts = result.get_counts()
                        probs = estimate_probs_from_counts(counts, circuit.num_qubits) if dense_output else None
                        note = f"device={dev}, cx={cx}" if dev != primary or cx > 0 else ""
                        return BackendResult(
                            probs=probs,
                            counts=counts,
                            backend_used="bluequbit_mps",
                            shots=shots,
                            requested_backend=requested,
                            fell_back_to_aer=False,
                            note=note,
                        )
                    except Exception as e:
                        last_err = e
                        err_txt = str(e).lower()
                        retriable = (
                            ("two-qubit" in err_txt or "10000" in err_txt or "too big" in err_txt)
                            and dev == "mps.cpu"
                            and len(devices) > 1
                        )
                        if retriable:
                            continue
                        break

                if last_err is not None:
                    raise last_err
                raise RuntimeError("BlueQubit MPS: no device attempted")
            except Exception as e:
                if os.getenv("BQ_STRICT", "0") == "1":
                    raise RuntimeError(f"BlueQubit MPS requested but failed: {e}") from e
                note = f"BlueQubit MPS failed, falling back to Aer qasm: {e}"
                print(f"[BackendAdapter] WARNING: {note}")
        else:
            if prefer_mps and os.getenv("BQ_STRICT", "0") == "1":
                raise RuntimeError("BlueQubit MPS requested in strict mode, but BQ is unavailable")
            note = "BlueQubit MPS unavailable, using Aer qasm"
            if prefer_mps:
                print(f"[BackendAdapter] WARNING: {note}")
        sim = AerSimulator()
        job = sim.run(transpile(circuit, sim), shots=shots)
        counts = job.result().get_counts()
        probs = estimate_probs_from_counts(counts, circuit.num_qubits) if dense_output else None
        return BackendResult(
            probs=probs,
            counts=counts,
            backend_used="aer_qasm",
            shots=shots,
            requested_backend=requested,
            fell_back_to_aer=prefer_mps,
            note=note if prefer_mps else "",
        )

