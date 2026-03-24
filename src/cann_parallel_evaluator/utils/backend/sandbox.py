# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Sandbox execution for CANN operators.

This module provides process isolation for NPU operations to prevent
environment pollution and segmentation faults from exec() calls.

Architecture:
- Compilation runs in the main process (subprocess.run for shell commands,
  no torch_npu needed, no segfault risk).
- Correctness verification and performance measurement run in sandboxed
  ``spawn`` subprocesses to isolate exec() and torch_npu state.
- A combined worker (_verify_and_measure_worker) handles both correctness
  and performance in a single subprocess, avoiding redundant NPU
  initialization (~10-20s saved per evaluation).
"""

import multiprocessing as mp
import time
from typing import Any, Dict, Optional, Callable


# ============================================================================
# Shared helpers (used inside subprocess workers)
# ============================================================================

def _setup_npu_environment(
    project_path: Optional[str],
    device_str: str = "npu:0",
) -> None:
    """Set up NPU environment inside a subprocess.

    MUST be called BEFORE ``import torch_npu`` because torch_npu reads
    these environment variables during import-time CANN runtime init.

    Sets:
    - ``ASCEND_RT_VISIBLE_DEVICES``: isolates the physical NPU so custom
      operators load correctly on any device (CANN custom OPP only works
      when the process sees a single device mapped to ``npu:0``).
    - ``ASCEND_CUSTOM_OPP_PATH``: points to the project-local OPP dir.
    - ``LD_LIBRARY_PATH``: adds libcust_opapi.so and pybind .so paths.

    Args:
        project_path: Root directory of the compiled operator project.
        device_str: Target device string (e.g. ``"npu:2"``). The physical
            device ID is extracted and set as ``ASCEND_RT_VISIBLE_DEVICES``.
    """
    import os
    from pathlib import Path

    # Isolate the physical NPU device so custom OPP loads correctly.
    # Without this, custom operators only work on npu:0 in multi-device
    # environments because CANN runtime binds custom OPP to device 0.
    if ":" in device_str:
        physical_id = device_str.split(":")[1]
    else:
        physical_id = "0"
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = physical_id

    if not project_path:
        return

    custom_opp_path = Path(project_path) / "opp" / "vendors" / "customize"
    if custom_opp_path.exists():
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(custom_opp_path)
        lib_path = custom_opp_path / "op_api" / "lib"
        if lib_path.exists():
            existing = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{existing}"

    extension_build = Path(project_path) / "CppExtension" / "build"
    if extension_build.exists():
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        if str(extension_build) not in existing:
            os.environ["LD_LIBRARY_PATH"] = f"{extension_build}:{existing}"


def _init_npu_context(
    context_data: Dict[str, Any],
    python_reference: str,
    device_str: str,
):
    """Common init sequence: import torch_npu, exec model & reference code.

    The ``device_str`` argument is ignored — ``npu:0`` is always used because
    ``_setup_npu_environment`` sets ``ASCEND_RT_VISIBLE_DEVICES`` to isolate
    the physical device, making it appear as ``npu:0`` to torch_npu.

    Returns:
        (context, device, torch_npu) tuple ready for correctness/performance.
    """
    import torch
    import torch_npu

    # Always use npu:0 — physical device isolation is done via
    # ASCEND_RT_VISIBLE_DEVICES in _setup_npu_environment.
    device = torch.device("npu:0")

    context = {}
    if "model_src" in context_data:
        exec(context_data["model_src"], context)
    exec(python_reference, context)

    return context, device, torch_npu


# ============================================================================
# Worker functions at module level to make them picklable
# ============================================================================

def _verify_correctness_worker(
    python_reference: str,
    context_data: Dict[str, Any],
    device_str: str,
    num_trials: int,
    seed: int,
    project_path: Optional[str],
    return_dict: dict,
    timing_dict: dict,
):
    """Worker for correctness verification in subprocess."""
    try:
        _setup_npu_environment(project_path, device_str)
        context, device, torch_npu = _init_npu_context(
            context_data, python_reference, device_str,
        )

        from cann_parallel_evaluator.utils.backend.correctness import execute_correctness_check

        passed, error_msg, info = execute_correctness_check(
            context=context,
            device=device,
            synchronize=torch_npu.npu.synchronize,
            num_trials=num_trials,
            seed=seed,
        )

        return_dict["result"] = {
            "pass": passed,
            "error": error_msg if not passed else None,
            **info,
        }
        timing_dict["completed"] = True

    except Exception as e:
        return_dict["result"] = {
            "pass": False,
            "error": f"Sandbox error: {str(e)}",
        }
        timing_dict["completed"] = True


def _measure_performance_worker(
    context_data: Dict[str, Any],
    python_reference: str,
    device_str: str,
    num_warmup: int,
    num_trials: int,
    project_path: Optional[str],
    return_dict: dict,
    timing_dict: dict,
):
    """Worker for performance measurement in subprocess."""
    try:
        _setup_npu_environment(project_path, device_str)
        context, device, torch_npu = _init_npu_context(
            context_data, python_reference, device_str,
        )

        from cann_parallel_evaluator.utils.backend.performance import measure_performance

        result = measure_performance(
            context=context,
            device=device,
            synchronize=torch_npu.npu.synchronize,
            event_class=torch_npu.npu.Event,
            num_warmup=num_warmup,
            num_trials=num_trials,
            measure_baseline=True,
        )

        return_dict["result"] = result
        timing_dict["completed"] = True

    except Exception as e:
        return_dict["result"] = {
            "runtime": None,
            "error": f"Sandbox error: {str(e)}",
        }
        timing_dict["completed"] = True


def _verify_and_measure_worker(
    python_reference: str,
    context_data: Dict[str, Any],
    device_str: str,
    num_correctness_trials: int,
    seed: int,
    num_warmup: int,
    num_perf_trials: int,
    skip_correctness: bool,
    skip_performance: bool,
    project_path: Optional[str],
    return_dict: dict,
    timing_dict: dict,
):
    """Combined worker: correctness + performance in a single subprocess.

    Avoids spawning two separate processes and re-initializing NPU (~10-20s
    saved). Runs correctness first; if it fails, skips performance.
    """
    try:
        _setup_npu_environment(project_path, device_str)
        context, device, torch_npu = _init_npu_context(
            context_data, python_reference, device_str,
        )

        # ── Correctness ──
        corr_result = None
        if not skip_correctness:
            from cann_parallel_evaluator.utils.backend.correctness import execute_correctness_check

            passed, error_msg, info = execute_correctness_check(
                context=context,
                device=device,
                synchronize=torch_npu.npu.synchronize,
                num_trials=num_correctness_trials,
                seed=seed,
            )
            corr_result = {
                "pass": passed,
                "error": error_msg if not passed else None,
                **info,
            }
            if not passed:
                return_dict["result"] = {
                    "correctness": corr_result,
                    "performance": None,
                }
                timing_dict["completed"] = True
                return

        # ── Performance ──
        perf_result = None
        if not skip_performance:
            from cann_parallel_evaluator.utils.backend.performance import measure_performance

            perf_result = measure_performance(
                context=context,
                device=device,
                synchronize=torch_npu.npu.synchronize,
                event_class=torch_npu.npu.Event,
                num_warmup=num_warmup,
                num_trials=num_perf_trials,
                measure_baseline=True,
            )

        return_dict["result"] = {
            "correctness": corr_result,
            "performance": perf_result,
        }
        timing_dict["completed"] = True

    except Exception as e:
        return_dict["result"] = {
            "correctness": None,
            "performance": None,
            "error": f"Sandbox error: {str(e)}",
        }
        timing_dict["completed"] = True


# ============================================================================
# Sandbox Executor Class
# ============================================================================

class CANNSandboxExecutor:
    """
    Sandbox executor for CANN operations using multiprocessing.

    Provides process isolation to prevent:
    - Environment pollution from exec() calls
    - Segmentation faults from NPU memory issues
    - Resource leaks between evaluations

    Usage:
        executor = CANNSandboxExecutor()

        # Combined correctness + performance (recommended, saves ~10-20s)
        result = executor.verify_and_measure_sandbox(
            python_reference=PYTHON_REF,
            context_data={"model_src": model_src},
            device="npu:0",
            project_path="/path/to/project",
        )

        # Correctness only
        result = executor.verify_correctness_sandbox(
            python_reference=PYTHON_REF,
            context_data={"model_src": model_src},
            device="npu:0",
        )
    """

    def __init__(self, default_timeout: int = 600):
        """
        Initialize sandbox executor.

        Args:
            default_timeout: Default timeout in seconds for operations
        """
        self.default_timeout = default_timeout

    @staticmethod
    def _monitor_process(
        process: mp.Process,
        timing_dict: dict,
        timeout: int,
    ) -> bool:
        """
        Monitor a process with timeout.

        Args:
            process: The subprocess to monitor
            timing_dict: Shared dict for timing info
            timeout: Timeout in seconds

        Returns:
            True if completed normally, False if timeout
        """
        start_time = time.time()

        while process.is_alive():
            if timing_dict.get("completed", False):
                process.join()
                return True

            elapsed = time.time() - start_time
            if elapsed > timeout:
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
                return False

            time.sleep(0.5)

        return True

    @staticmethod
    def _execute_in_sandbox(
        worker_func: Callable,
        worker_args: tuple,
        timeout: int,
        default_error: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a worker function in a sandboxed subprocess.

        Args:
            worker_func: The worker function to execute
            worker_args: Arguments for the worker function
            timeout: Timeout in seconds
            default_error: Default error result if timeout/failure

        Returns:
            Result dict from worker function
        """
        try:
            # Use spawn to ensure clean process
            ctx = mp.get_context("spawn")
            manager = ctx.Manager()
            return_dict = manager.dict()
            timing_dict = manager.dict()

            # Add shared dicts to args
            full_args = worker_args + (return_dict, timing_dict)

            process = ctx.Process(target=worker_func, args=full_args)
            process.start()

            if not CANNSandboxExecutor._monitor_process(process, timing_dict, timeout):
                error_result = default_error.copy()
                error_result["error"] = f"Operation timed out after {timeout}s"
                return error_result

            return dict(return_dict.get("result", default_error))

        except Exception as e:
            error_result = default_error.copy()
            error_result["error"] = f"Sandbox execution error: {str(e)}"
            return error_result

    def verify_correctness_sandbox(
        self,
        python_reference: str,
        context_data: Dict[str, Any],
        device: str = "npu:0",
        num_trials: int = 5,
        seed: int = 1024,
        project_path: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Verify correctness in a sandboxed subprocess.

        Args:
            python_reference: Python reference implementation
            context_data: Dict with model_src and other context data
            device: NPU device string
            num_trials: Number of verification trials
            seed: Random seed
            project_path: Path to compiled operator project (for environment setup)
            timeout: Timeout in seconds (uses default if None)

        Returns:
            {"pass": bool, "error": str or None, ...}
        """
        return self._execute_in_sandbox(
            worker_func=_verify_correctness_worker,
            worker_args=(python_reference, context_data, device, num_trials, seed, project_path),
            timeout=timeout or self.default_timeout,
            default_error={"pass": False, "error": "Unknown error"},
        )

    def measure_performance_sandbox(
        self,
        context_data: Dict[str, Any],
        python_reference: str,
        device: str = "npu:0",
        num_warmup: int = 3,
        num_trials: int = 100,
        project_path: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Measure performance in a sandboxed subprocess.

        Args:
            context_data: Dict with model_src and other context data
            python_reference: Python reference (for get_inputs, etc.)
            device: NPU device string
            num_warmup: Number of warmup iterations
            num_trials: Number of measurement trials
            project_path: Path to compiled operator project (for environment setup)
            timeout: Timeout in seconds (uses default if None)

        Returns:
            {"runtime": float, "std": float, ...}
        """
        return self._execute_in_sandbox(
            worker_func=_measure_performance_worker,
            worker_args=(context_data, python_reference, device, num_warmup, num_trials, project_path),
            timeout=timeout or self.default_timeout,
            default_error={"runtime": None, "error": "Unknown error"},
        )

    def verify_and_measure_sandbox(
        self,
        python_reference: str,
        context_data: Dict[str, Any],
        device: str = "npu:0",
        num_correctness_trials: int = 5,
        seed: int = 1024,
        num_warmup: int = 3,
        num_perf_trials: int = 100,
        skip_correctness: bool = False,
        skip_performance: bool = False,
        project_path: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Combined correctness + performance in a single sandboxed subprocess.

        Saves ~10-20s per evaluation by avoiding redundant NPU initialization.
        Runs correctness first; if it fails, skips performance.

        Args:
            python_reference: Python reference implementation
            context_data: Dict with model_src and other context data
            device: NPU device string
            num_correctness_trials: Number of correctness trials
            seed: Random seed
            num_warmup: Number of warmup iterations
            num_perf_trials: Number of performance trials
            skip_correctness: Skip correctness check
            skip_performance: Skip performance measurement
            project_path: Path to compiled operator project
            timeout: Timeout in seconds (uses default if None)

        Returns:
            {
                "correctness": {"pass": bool, "error": str, ...} or None,
                "performance": {"runtime": float, ...} or None,
                "error": str (only on unexpected failure),
            }
        """
        return self._execute_in_sandbox(
            worker_func=_verify_and_measure_worker,
            worker_args=(
                python_reference, context_data, device,
                num_correctness_trials, seed,
                num_warmup, num_perf_trials,
                skip_correctness, skip_performance,
                project_path,
            ),
            timeout=timeout or self.default_timeout,
            default_error={
                "correctness": None,
                "performance": None,
                "error": "Unknown error",
            },
        )
