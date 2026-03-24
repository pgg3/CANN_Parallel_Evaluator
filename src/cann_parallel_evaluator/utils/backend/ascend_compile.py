# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Ascend C compilation pipeline.

Adapted from MultiKernelBench/utils/ascend_compile_pipeline.py

Supports phased execution for parallel compilation:
- ascend_setup: msopgen + write files (thread-safe via cwd= isolation)
- ascend_build: build.sh + deploy + pybind (thread-safe via cwd= isolation)
- ascend_compile: full pipeline (setup + build)

Thread safety: All subprocess calls use cwd= instead of os.chdir(),
and no os.environ mutations in the main process. This allows multiple
evaluator threads to compile concurrently without races.
"""

import os
import re
import subprocess
import shutil
from typing import Dict, Any, List, Callable, Optional

from ..templates.pybind_templates import setup_pybind_directory


def _make_logger(verbose: bool) -> Callable[[str], None]:
    """Create a logger function based on verbose setting."""
    if verbose:
        return lambda msg: print(msg)
    else:
        return lambda msg: None


def _patch_build_sh_for_duplicate_fix(target_directory: str) -> None:
    """Patch build scripts to handle duplicate sections/definitions from CANN toolchain bug.

    The CANN build process sometimes generates duplicate sections in .ini files
    and duplicate REG_OP definitions in .h files. This function patches:
    1. ascendc_get_op_name.py to use a duplicate-tolerant configparser
    2. build.sh to run fix scripts after cmake generates autogen files
    """
    # Patch ascendc_get_op_name.py to tolerate duplicate sections
    get_op_name_path = os.path.join(target_directory, "cmake", "util", "ascendc_get_op_name.py")
    if os.path.exists(get_op_name_path):
        with open(get_op_name_path, "r") as f:
            content = f.read()

        # Replace configparser.ConfigParser() with a duplicate-tolerant version
        if "class DuplicateTolerantConfigParser" not in content:
            patched_content = content.replace(
                "import configparser",
                '''import configparser

class DuplicateTolerantConfigParser(configparser.ConfigParser):
    """ConfigParser that ignores duplicate sections instead of raising errors."""
    def _read(self, fp, fpname):
        """Override to skip duplicate sections silently."""
        import re
        seen_sections = set()
        lines = []
        current_section = None
        skip_section = False
        for line in fp:
            stripped = line.strip()
            if stripped.startswith('[') and stripped.endswith(']'):
                section = stripped[1:-1]
                if section in seen_sections:
                    skip_section = True
                    continue
                seen_sections.add(section)
                skip_section = False
                current_section = section
            if not skip_section:
                lines.append(line)
        from io import StringIO
        return super()._read(StringIO(''.join(lines)), fpname)'''
            )
            patched_content = patched_content.replace(
                "op_config = configparser.ConfigParser()",
                "op_config = DuplicateTolerantConfigParser()"
            )
            with open(get_op_name_path, "w") as f:
                f.write(patched_content)

    # Also write fix script and patch build.sh for .h file duplicates
    build_sh_path = os.path.join(target_directory, "build.sh")
    fix_script_path = os.path.join(target_directory, "fix_duplicates.py")

    fix_script_content = '''#!/usr/bin/env python3
"""Fix duplicate sections in .ini files and duplicate REG_OP in .h files."""
import os
import re
import glob

build_out_dir = os.path.join(os.path.dirname(__file__), "build_out")
autogen_dir = os.path.join(build_out_dir, "autogen")

if os.path.exists(autogen_dir):
    # Fix .ini files
    for ini_file in glob.glob(os.path.join(autogen_dir, "*.ini")):
        try:
            with open(ini_file, "r") as f:
                lines = f.readlines()
            seen_sections = set()
            result_lines = []
            skip_section = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("[") and stripped.endswith("]"):
                    section_name = stripped[1:-1]
                    if section_name in seen_sections:
                        skip_section = True
                    else:
                        seen_sections.add(section_name)
                        skip_section = False
                        result_lines.append(line)
                elif not skip_section:
                    result_lines.append(line)
            with open(ini_file, "w") as f:
                f.writelines(result_lines)
        except Exception:
            pass

    # Fix .h files (duplicate REG_OP)
    for header_file in glob.glob(os.path.join(autogen_dir, "*.h")):
        try:
            with open(header_file, "r") as f:
                content = f.read()
            reg_op_pattern = r"(REG_OP\\((\\w+)\\)[^;]*\\.OP_END_FACTORY_REG\\(\\2\\);)"
            seen_ops = set()
            def replace_dup(m):
                op_name = m.group(2)
                if op_name in seen_ops:
                    return ""
                seen_ops.add(op_name)
                return m.group(0)
            new_content = re.sub(reg_op_pattern, replace_dup, content, flags=re.DOTALL)
            new_content = re.sub(r"\\n{3,}", "\\n\\n", new_content)
            with open(header_file, "w") as f:
                f.write(new_content)
        except Exception:
            pass
'''
    with open(fix_script_path, "w") as f:
        f.write(fix_script_content)
    os.chmod(fix_script_path, 0o755)

    # Read and patch build.sh
    with open(build_sh_path, "r") as f:
        build_sh_content = f.read()

    # Insert fix script call before EVERY cmake --build line
    fix_call = 'python3 "$script_path/fix_duplicates.py"\n'

    if fix_call not in build_sh_content:
        build_sh_content = build_sh_content.replace(
            "cmake --build",
            fix_call + "  cmake --build"
        )
        with open(build_sh_path, "w") as f:
            f.write(build_sh_content)


def _patch_cmake_for_ascendc_includes(target_directory: str) -> None:
    """Patch CMake files to add ascendc highlevel_api include path for host-side compilation.

    The Ascend C high-level API headers (e.g. lib/matmul_intf.h, lib/tiling_api.h,
    tiling/platform/platform_ascendc.h) live under .../include/ascendc/highlevel_api/
    but the default CMake config only adds .../include/ to the search path.
    The kernel compiler (ccec) adds this path automatically, but g++ does not.
    """
    highlevel_api_path = "${ASCEND_CANN_PACKAGE_PATH}/include/ascendc/highlevel_api"

    # Patch intf.cmake - add to target_include_directories
    intf_cmake_path = os.path.join(target_directory, "cmake", "intf.cmake")
    if os.path.exists(intf_cmake_path):
        with open(intf_cmake_path, "r") as f:
            content = f.read()
        if "highlevel_api" not in content:
            content = content.replace(
                "target_include_directories(intf_pub INTERFACE ${ASCEND_CANN_PACKAGE_PATH}/include\n",
                f"target_include_directories(intf_pub INTERFACE ${{ASCEND_CANN_PACKAGE_PATH}}/include\n"
                f"    {highlevel_api_path}\n",
            )
            with open(intf_cmake_path, "w") as f:
                f.write(content)

    # Patch func.cmake - add -I flag to opbuild() compile command
    func_cmake_path = os.path.join(target_directory, "cmake", "func.cmake")
    if os.path.exists(func_cmake_path):
        with open(func_cmake_path, "r") as f:
            content = f.read()
        if "highlevel_api" not in content:
            content = content.replace(
                "-I ${ASCEND_CANN_PACKAGE_PATH}/include -I ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel",
                f"-I ${{ASCEND_CANN_PACKAGE_PATH}}/include -I ${{ASCEND_CANN_PACKAGE_PATH}}/include/ascendc/highlevel_api -I ${{CMAKE_CURRENT_SOURCE_DIR}}/../op_kernel",
            )
            with open(func_cmake_path, "w") as f:
                f.write(content)


def underscore_to_pascalcase(underscore_str: str) -> str:
    """Convert underscore-separated string to PascalCase."""
    if not underscore_str:
        return ""
    parts = underscore_str.split("_")
    return "".join(word.capitalize() for word in parts if word)


def _pascal_to_snake(pascal_str: str) -> str:
    """Convert PascalCase to snake_case (matches msopgen filename convention)."""
    return re.sub(r'(?<!^)(?=[A-Z])', '_', pascal_str).lower()


def write_project_files(
    full_code: Dict[str, str],
    op_name: str,
    project_path: str,
) -> Dict[str, Any]:
    """
    Write all project files without compiling (for fake mode).

    Creates the same directory structure as ascend_compile but skips:
    - msopgen (no project skeleton)
    - build.sh (no compilation)
    - deploy (no installation)
    - pybind build (no wheel)

    Args:
        full_code: Dictionary containing all code components
        op_name: Operator name (e.g., "add")
        project_path: Base directory for operator projects

    Returns:
        {"success": bool, "error": str or None, "files_written": list}
    """
    op = f"{op_name}_custom"
    op_capital = underscore_to_pascalcase(op)
    # msopgen derives filenames by converting PascalCase to snake_case
    # e.g., MatrixMultiplicationCustom -> matrix_multiplication_custom
    #        Avgpooling2dCustom -> avgpooling2d_custom
    op_file = _pascal_to_snake(op_capital)
    target_directory = os.path.join(project_path, op_capital)
    files_written: List[str] = []

    try:
        # Create directory structure
        os.makedirs(project_path, exist_ok=True)
        os.makedirs(os.path.join(target_directory, "op_host"), exist_ok=True)
        os.makedirs(os.path.join(target_directory, "op_kernel"), exist_ok=True)

        # Write project JSON
        json_path = os.path.join(project_path, f"{op}.json")
        with open(json_path, "w") as f:
            f.write(full_code.get("project_json_src", ""))
        files_written.append(json_path)

        # Write source files
        # NOTE: msopgen uses lowercase filenames (e.g., sdpa_custom_tiling.h, sdpa_custom.cpp)
        # and our generated code includes match this convention
        tiling_path = os.path.join(target_directory, "op_host", f"{op_file}_tiling.h")
        with open(tiling_path, "w") as f:
            f.write(full_code.get("host_tiling_src", ""))
        files_written.append(tiling_path)

        host_path = os.path.join(target_directory, "op_host", f"{op_file}.cpp")
        with open(host_path, "w") as f:
            f.write(full_code.get("host_operator_src", ""))
        files_written.append(host_path)

        kernel_path = os.path.join(target_directory, "op_kernel", f"{op_file}.cpp")
        with open(kernel_path, "w") as f:
            f.write(full_code.get("kernel_src", ""))
        files_written.append(kernel_path)

        # Set up Python binding directory
        cpp_ext_dir = setup_pybind_directory(project_path)
        csrc_dir = os.path.join(cpp_ext_dir, "csrc")

        pybind_path = os.path.join(csrc_dir, "op.cpp")
        with open(pybind_path, "w") as f:
            f.write(full_code.get("python_bind_src", ""))
        files_written.append(pybind_path)

        # Write model_src as reference
        model_path = os.path.join(project_path, "model_src.py")
        with open(model_path, "w") as f:
            f.write(full_code.get("model_src", ""))
        files_written.append(model_path)

        return {
            "success": True,
            "error": None,
            "files_written": files_written,
            "project_directory": target_directory,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to write files: {str(e)}",
            "files_written": files_written,
        }


def ascend_setup(
    full_code: Dict[str, str],
    op_name: str,
    project_path: str,
    device: str = "Ascend910B",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Setup phase: msopgen + write source files.

    This phase must run sequentially due to msopgen global resource conflicts.

    Args:
        full_code: Dictionary containing all code components
        op_name: Operator name (e.g., "add")
        project_path: Base directory for operator projects
        device: Target device (e.g., "Ascend910B")
        verbose: Whether to print progress messages

    Returns:
        {"success": bool, "error": str or None, "target_directory": str}
    """
    log = _make_logger(verbose)
    # Resolve to absolute path for cwd= arguments and --install-path.
    project_path = os.path.abspath(project_path)
    op = f"{op_name}_custom"
    op_capital = underscore_to_pascalcase(op)
    # msopgen derives filenames by converting PascalCase to snake_case
    # e.g., MatrixMultiplicationCustom -> matrix_multiplication_custom
    #        Avgpooling2dCustom -> avgpooling2d_custom
    op_file = _pascal_to_snake(op_capital)
    target_directory = os.path.join(project_path, op_capital)

    # Convert device name to msopgen compute unit format
    if device.lower().startswith("ascend"):
        if device[-1].isdigit() and device[-2].isalpha():
            compute_unit = f"ai_core-{device}"
        else:
            compute_unit = f"ai_core-{device}2"
    else:
        compute_unit = f"ai_core-{device}"

    try:
        # Step 1: Create operator project directory
        os.makedirs(project_path, exist_ok=True)

        if os.path.exists(target_directory):
            shutil.rmtree(target_directory)

        # Write project JSON
        json_path = os.path.join(project_path, f"{op}.json")
        with open(json_path, "w") as f:
            f.write(full_code.get("project_json_src", ""))

        # Step 2: Run msopgen to create project structure
        # NOTE: Use cwd= instead of os.chdir() for thread safety.
        log("[INFO] Creating operator project with msopgen...")

        try:
            subprocess.run(
                [
                    "msopgen", "gen",
                    "-i", f"{op}.json",
                    "-c", compute_unit,
                    "-lan", "cpp",
                    "-out", op_capital,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=project_path,
            )
            log("[INFO] Operator project created successfully")
        except subprocess.CalledProcessError as e:
            error_msg = f"msopgen failed:\nExit Code: {e.returncode}\nStdout:\n{e.stdout}\nStderr:\n{e.stderr}"
            return {"success": False, "error": error_msg, "target_directory": target_directory}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "msopgen timed out", "target_directory": target_directory}

        # Step 3: Write source files to project
        log("[INFO] Writing source files...")

        # NOTE: msopgen uses lowercase filenames (e.g., sdpa_custom_tiling.h, sdpa_custom.cpp)
        with open(os.path.join(target_directory, "op_host", f"{op_file}_tiling.h"), "w") as f:
            f.write(full_code.get("host_tiling_src", ""))

        with open(os.path.join(target_directory, "op_host", f"{op_file}.cpp"), "w") as f:
            f.write(full_code.get("host_operator_src", ""))

        with open(os.path.join(target_directory, "op_kernel", f"{op_file}.cpp"), "w") as f:
            f.write(full_code.get("kernel_src", ""))

        # Set up Python binding directory with built-in templates
        log("[INFO] Setting up Python binding environment...")
        cpp_ext_dir = setup_pybind_directory(project_path)
        csrc_dir = os.path.join(cpp_ext_dir, "csrc")
        with open(os.path.join(csrc_dir, "op.cpp"), "w") as f:
            f.write(full_code.get("python_bind_src", ""))

        # Write model_src for later use
        model_path = os.path.join(project_path, "model_src.py")
        with open(model_path, "w") as f:
            f.write(full_code.get("model_src", ""))

        log("[INFO] Setup phase completed successfully")
        return {"success": True, "error": None, "target_directory": target_directory}

    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}", "target_directory": target_directory}


def _cleanup_exception_dumps(target_directory: str, log, original_cwd: str = None) -> None:
    """Remove CANN exception dump files that can be 4GB+ each.

    The CANN compiler generates huge exception_info files in
    extra-info/data-dump/ when compilation fails. These quickly
    fill the disk in iterative compilation scenarios.
    CANN may also generate dumps in the Python launch directory.
    """
    dirs_to_clean = [target_directory]
    if original_cwd:
        dirs_to_clean.append(original_cwd)
    for base_dir in dirs_to_clean:
        extra_info_dir = os.path.join(base_dir, "extra-info")
        if os.path.exists(extra_info_dir):
            shutil.rmtree(extra_info_dir, ignore_errors=True)
            log(f"[INFO] Cleaned extra-info in {base_dir}")


def ascend_build(
    op_name: str,
    project_path: str,
    full_code: Dict[str, str],
    verbose: bool = True,
    skip_model_exec: bool = False,
) -> Dict[str, Any]:
    """
    Build phase: build.sh + deploy + pybind.

    This phase can run in parallel as each operates in isolated directories.

    Args:
        op_name: Operator name (e.g., "add")
        project_path: Base directory for operator projects
        full_code: Dictionary containing code (needed for model_src)
        verbose: Whether to print progress messages
        skip_model_exec: If True, skip Step 8 (exec model_src). Useful when
            the caller will handle model loading separately (e.g., sandbox
            workers that need to set env vars before importing torch_npu).

    Returns:
        {"success": bool, "error": str or None, "context": dict}
    """
    log = _make_logger(verbose)
    # Resolve to absolute path for --install-path and cwd= arguments.
    project_path = os.path.abspath(project_path)
    op = f"{op_name}_custom"
    op_capital = underscore_to_pascalcase(op)
    target_directory = os.path.join(project_path, op_capital)
    cpp_ext_dir = os.path.join(project_path, "CppExtension")
    context = {}

    try:
        # Step 4: Build the operator
        log("[INFO] Building operator...")

        # Clean build_out directory
        build_out_dir = os.path.join(target_directory, "build_out")
        if os.path.exists(build_out_dir):
            shutil.rmtree(build_out_dir)
            log("[INFO] Cleaned build_out directory")

        # Patch build.sh to fix CANN duplicate section bug
        _patch_build_sh_for_duplicate_fix(target_directory)
        log("[INFO] Patched build.sh for duplicate fix")

        # Patch CMake to add ascendc highlevel_api include path for Cube operators
        _patch_cmake_for_ascendc_includes(target_directory)
        log("[INFO] Patched CMake for ascendc includes")

        # NOTE: Use cwd= instead of os.chdir() for thread safety.
        # os.chdir() is process-global and causes races under parallel compilation.
        try:
            result = subprocess.run(
                ["bash", "./build.sh"],
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
                cwd=target_directory,
            )
            log("[INFO] Build succeeded")
        except subprocess.CalledProcessError as e:
            # Capture full output for debugging
            full_output = f"STDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"
            error_lines = []
            for line in (e.stdout + e.stderr).split("\n"):
                if "[ERROR]" in line or "error:" in line.lower() or "Error" in line:
                    error_lines.append(line)
            error_msg = f"Build failed:\nExit Code: {e.returncode}\nErrors:\n" + "\n".join(error_lines[:30])
            if not error_lines:
                error_msg += f"\n\nFull output:\n{full_output[:2000]}"
            _cleanup_exception_dumps(target_directory, log)
            return {"success": False, "error": error_msg, "context": context}
        except subprocess.TimeoutExpired:
            _cleanup_exception_dumps(target_directory, log)
            return {"success": False, "error": "Build timed out", "context": context}

        # Clean up after successful build too (may still generate dumps)
        _cleanup_exception_dumps(target_directory, log)

        # Step 5: Deploy the operator package to project-local opp directory
        # IMPORTANT: Use --install-path to avoid global OPP pollution
        # This enables parallel compilation without conflicts
        log("[INFO] Deploying operator package...")
        build_out_dir = os.path.join(target_directory, "build_out")

        # Create local opp directory for this project
        local_opp_path = os.path.join(project_path, "opp")
        os.makedirs(local_opp_path, exist_ok=True)

        # Find the .run installer file (name varies by platform)
        import glob as glob_module
        run_files = glob_module.glob(os.path.join(build_out_dir, "*.run"))
        if not run_files:
            return {"success": False, "error": "No .run installer found in build_out", "context": context}
        run_file = run_files[0]  # Use the first .run file found (absolute path)

        try:
            subprocess.run(
                [run_file, f"--install-path={local_opp_path}"],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=build_out_dir,
            )
            log(f"[INFO] Deploy succeeded to {local_opp_path}")
        except subprocess.CalledProcessError as e:
            error_msg = f"Deploy failed:\nExit Code: {e.returncode}\nOutput:\n{e.stdout}\nStderr:\n{e.stderr}"
            return {"success": False, "error": error_msg, "context": context}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Deploy timed out", "context": context}

        # Step 6: Build Python bindings
        log("[INFO] Building Python bindings...")

        try:
            subprocess.run(
                ["bash", "build_and_run.sh"],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=cpp_ext_dir,
            )
            log("[INFO] Python binding succeeded")
        except subprocess.CalledProcessError as e:
            error_msg = f"Python binding failed:\nExit Code: {e.returncode}\nOutput:\n{e.stdout}"
            return {"success": False, "error": error_msg, "context": context}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Python binding timed out", "context": context}

        # Step 7: Skip env var mutation — sandbox workers set ASCEND_CUSTOM_OPP_PATH
        # themselves via _setup_npu_environment(). Mutating os.environ here is
        # not thread-safe and not needed since compilation uses project-local paths.

        # Step 8: Load model code into context
        # Skip when caller handles model loading separately (e.g., sandbox workers
        # that must set ASCEND_CUSTOM_OPP_PATH before importing torch_npu)
        if not skip_model_exec:
            log("[INFO] Loading model code...")
            try:
                model_src = full_code.get("model_src", "")
                compile(model_src, "<string>", "exec")
                exec(model_src, context)
            except Exception as e:
                return {"success": False, "error": f"Failed to load model: {str(e)}", "context": context}

        log("[INFO] Build phase completed successfully")
        return {"success": True, "error": None, "context": context}

    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}", "context": context}


def ascend_compile(
    full_code: Dict[str, str],
    op_name: str,
    project_path: str,
    device: str = "Ascend910B",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compile Ascend C operator code (full pipeline).

    This function runs both setup and build phases sequentially.
    For parallel compilation, use ascend_setup + ascend_build separately.

    Args:
        full_code: Dictionary containing all code components:
            - project_json_src
            - host_tiling_src
            - host_operator_src
            - kernel_src
            - python_bind_src
            - model_src
        op_name: Operator name (e.g., "add")
        project_path: Base directory for operator projects
        device: Target device (e.g., "Ascend910B")
        verbose: Whether to print progress messages

    Returns:
        {"success": bool, "error": str or None, "context": dict}
    """
    # Phase 1: Setup
    setup_result = ascend_setup(full_code, op_name, project_path, device, verbose=verbose)
    if not setup_result["success"]:
        return {"success": False, "error": setup_result["error"], "context": {}}

    # Phase 2: Build
    build_result = ascend_build(op_name, project_path, full_code, verbose=verbose)
    return build_result
