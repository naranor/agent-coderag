"""Dependency sync use-cases (Maven/Gradle JAR path cache)."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess  # nosec
from pathlib import Path
from typing import Optional

from code_rag.core.interfaces import IStorage
from code_rag.core.utils import validate_path

logger = logging.getLogger(__name__)


async def _cache_jar_path(storage: IStorage, jar_path: str) -> None:
    jar_name = Path(jar_path).stem
    parts = jar_name.split("-")
    if len(parts) > 1 and parts[-1][0].isdigit():
        lib_name = "-".join(parts[:-1])
    else:
        lib_name = jar_name
    await storage.set_dependency_path(lib_name, jar_path)


async def _sync_maven(storage: IStorage, root: Path) -> None:
    mvn_bin: Optional[str] = shutil.which("mvn")
    if not mvn_bin:
        logger.warning("mvn not found, skipping dependency sync")
        return

    valid_root = validate_path(root)
    cp_file = valid_root / ".coderag_cp.txt"
    logger.info("Resolving Maven dependencies...")
    try:
        cmd = [
            mvn_bin,
            "dependency:build-classpath",
            f"-Dmdep.outputFile={cp_file.name}",
        ]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(valid_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )  # nosec
        _, stderr = await asyncio.wait_for(process.communicate(), timeout=60)

        if process.returncode != 0:
            logger.error("mvn failed: %s", stderr.decode())
            return

        if cp_file.exists():
            with open(cp_file, "r", encoding="utf-8") as f:
                classpath = f.read().strip()
            cp_file.unlink()

            for jar_path in classpath.split(os.pathsep):
                if not jar_path:
                    continue
                await _cache_jar_path(storage, jar_path)
            logger.info("Maven dependencies cached.")
    except Exception as e:
        logger.error("Failed to sync Maven dependencies: %s", e)


async def _execute_gradle_init(storage: IStorage, root: Path, gradle_bin: str) -> None:
    init_script = root / ".coderag_init.gradle"
    init_content = """
allprojects {
    tasks.register('printCodeRagCP') {
        doLast {
            def cp = []
            ['runtimeClasspath', 'compileClasspath', 'implementation'].each { cfgName ->
                def cfg = project.configurations.findByName(cfgName)
                if (cfg != null && cfg.canBeResolved) {
                    try {
                        cp.addAll(cfg.collect { it.absolutePath })
                    } catch (Exception e) {}
                }
            }
            if (cp) {
                println "CODERAG_CP:" + cp.unique().join(File.pathSeparator)
            }
        }
    }
}
"""
    logger.info("Resolving Gradle dependencies...")
    try:
        with open(init_script, "w", encoding="utf-8") as f:
            f.write(init_content)

        process = await asyncio.create_subprocess_exec(
            gradle_bin,
            "-q",
            "--init-script",
            str(init_script),
            "printCodeRagCP",
            cwd=str(root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)

        if process.returncode != 0:
            logger.error("gradle failed: %s", stderr.decode())
            return

        output = stdout.decode()
        for line in output.splitlines():
            if line.startswith("CODERAG_CP:"):
                classpath = line[len("CODERAG_CP:") :].strip()
                for jar_path in classpath.split(os.pathsep):
                    if not jar_path or not jar_path.endswith(".jar"):
                        continue
                    await _cache_jar_path(storage, jar_path)

        logger.info("Gradle dependencies cached.")
    except Exception as e:
        logger.error("Failed to sync Gradle dependencies: %s", e)
    finally:
        if init_script.exists():
            init_script.unlink()


async def _sync_gradle(storage: IStorage, root: Path) -> None:
    gradle_bin: Optional[str] = shutil.which("gradle")
    if not gradle_bin:
        logger.warning(
            "System 'gradle' executable not found in PATH. Skipping dependency "
            "sync to avoid executing untrusted repository wrappers."
        )
        return
    valid_root = validate_path(root)
    await _execute_gradle_init(storage, valid_root, gradle_bin)


async def sync_dependencies(
    storage: IStorage,
    project_path: str,
    *,
    allow_build_execution: bool = False,
) -> None:
    """Resolve Maven/Gradle deps and cache JAR paths into ``storage``."""
    root = Path(project_path)
    pom_xml = root / "pom.xml"
    gradle_files = list(root.glob("build.gradle*"))

    if not pom_xml.exists() and not gradle_files:
        return

    if not allow_build_execution:
        logger.warning(
            "Dependency sync is disabled by default because Maven/Gradle "
            "build files are executable repository code. Use the explicit "
            "allow_build_execution option only for trusted projects."
        )
        return

    if pom_xml.exists():
        await _sync_maven(storage, root)
    elif gradle_files:
        await _sync_gradle(storage, root)
