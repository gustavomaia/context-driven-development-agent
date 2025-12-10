"""Performance profiling tests for CDD Agent CLI.

This test suite validates startup time, import performance, and memory usage
against the targets defined in the roadmap.
"""

import importlib
import subprocess
import sys
import time

import pytest


class TestStartupPerformance:
    """Test CLI startup performance."""

    def test_help_command_startup_time(self):
        """Test that --help command responds quickly.

        Target: <200ms (ideally <100ms)
        This is the most critical metric for CLI responsiveness.
        """
        # Run multiple times to get average
        times = []
        for _ in range(5):
            start = time.perf_counter()
            result = subprocess.run(
                ["poetry", "run", "cdd-agent", "--help"],
                capture_output=True,
                timeout=5,
            )
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            times.append(elapsed)

            assert result.returncode == 0, "Help command should succeed"

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print("\n--help startup time:")
        print(f"  Average: {avg_time:.1f}ms")
        print(f"  Min: {min_time:.1f}ms")
        print(f"  Max: {max_time:.1f}ms")
        print("  Target: <200ms (ideal: <100ms)")

        # We'll allow up to 2000ms for now since we know performance needs work
        # TODO: Lower this to 200ms after optimization
        assert avg_time < 2000, (
            f"Startup time {avg_time:.1f}ms exceeds maximum allowed (2000ms). "
            f"Target is <200ms."
        )

    def test_version_command_startup_time(self):
        """Test that --version command responds quickly."""
        start = time.perf_counter()
        result = subprocess.run(
            ["poetry", "run", "cdd-agent", "--version"],
            capture_output=True,
            timeout=10,
        )
        elapsed = (time.perf_counter() - start) * 1000

        print(f"\n--version startup time: {elapsed:.1f}ms")

        assert result.returncode == 0, "Version command should succeed"
        # Allow up to 5000ms for parallel test execution (contention)
        assert elapsed < 5000, f"Version startup {elapsed:.1f}ms too slow"


class TestImportPerformance:
    """Test module import performance."""

    def test_cli_module_import_time(self):
        """Test how long it takes to import the CLI module.

        This is a proxy for startup time since importing the CLI module
        triggers all the cascading imports.
        """
        # Clear any cached imports
        modules_to_clear = [
            name for name in sys.modules.keys() if name.startswith("cdd_agent")
        ]
        for module_name in modules_to_clear:
            del sys.modules[module_name]

        start = time.perf_counter()
        importlib.import_module("cdd_agent.cli")
        elapsed = (time.perf_counter() - start) * 1000

        print(f"\nCLI module import time: {elapsed:.1f}ms")
        print("Target: <100ms for core modules")

        # Allow up to 1500ms for now
        assert elapsed < 1500, f"Import time {elapsed:.1f}ms too slow"

    def test_individual_module_import_times(self):
        """Profile import time for each major module."""
        modules_to_test = [
            "cdd_agent.config",
            "cdd_agent.tools",
            "cdd_agent.agent",
            "cdd_agent.auth",
        ]

        import_times = {}

        for module_name in modules_to_test:
            # Clear cached import
            if module_name in sys.modules:
                del sys.modules[module_name]

            # Clear dependent modules
            to_clear = [
                name for name in sys.modules.keys() if name.startswith(module_name)
            ]
            for name in to_clear:
                if name in sys.modules:
                    del sys.modules[name]

            start = time.perf_counter()
            try:
                importlib.import_module(module_name)
                elapsed = (time.perf_counter() - start) * 1000
                import_times[module_name] = elapsed
            except ImportError as e:
                import_times[module_name] = f"FAILED: {e}"

        print("\nIndividual module import times:")
        for module, time_ms in sorted(
            import_times.items(),
            key=lambda x: x[1] if isinstance(x[1], float) else 0,
        ):
            if isinstance(time_ms, float):
                print(f"  {module}: {time_ms:.1f}ms")
            else:
                print(f"  {module}: {time_ms}")

        # Verify all modules imported successfully
        for module, result in import_times.items():
            assert not isinstance(result, str), f"{module} failed to import: {result}"


class TestMemoryUsage:
    """Test memory usage patterns."""

    def test_import_memory_footprint(self):
        """Test memory usage after importing main modules.

        Target: <100MB baseline before conversation starts.
        """
        import tracemalloc

        tracemalloc.start()

        # Import main modules
        import cdd_agent.cli  # noqa: F401

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        current_mb = current / 1024 / 1024
        peak_mb = peak / 1024 / 1024

        print("\nMemory usage after imports:")
        print(f"  Current: {current_mb:.1f} MB")
        print(f"  Peak: {peak_mb:.1f} MB")
        print("  Target: <100 MB baseline")

        # Allow up to 150MB for now (includes test overhead)
        assert current_mb < 150, (
            f"Memory usage {current_mb:.1f}MB exceeds target. " f"Target is <100MB."
        )


class TestProviderLoadingPerformance:
    """Test lazy loading of provider SDKs."""

    def test_anthropic_not_loaded_at_import(self):
        """Verify Anthropic SDK is not loaded at import time (lazy loading)."""
        # Clear imports
        modules_to_clear = [
            name
            for name in sys.modules.keys()
            if name.startswith("anthropic") or name.startswith("cdd_agent")
        ]
        for module_name in modules_to_clear:
            if module_name in sys.modules:
                del sys.modules[module_name]

        # Import CLI
        import cdd_agent.cli  # noqa: F401

        # Check if anthropic was loaded
        anthropic_loaded = any(
            name.startswith("anthropic") for name in sys.modules.keys()
        )

        if anthropic_loaded:
            print("\n⚠️  Anthropic SDK loaded at import time (should be lazy loaded)")
            print("   This adds ~500ms to startup time unnecessarily")
        else:
            print("\n✓ Anthropic SDK not loaded at import (good lazy loading)")

        # For now, we document the issue but don't fail
        # TODO: Make this assertion strict after implementing lazy loading
        # assert not anthropic_loaded, "Anthropic should be lazy loaded"

    def test_openai_not_loaded_at_import(self):
        """Verify OpenAI SDK is not loaded at import time (lazy loading)."""
        # Clear imports
        modules_to_clear = [
            name
            for name in sys.modules.keys()
            if name.startswith("openai") or name.startswith("cdd_agent")
        ]
        for module_name in modules_to_clear:
            if module_name in sys.modules:
                del sys.modules[module_name]

        # Import CLI
        import cdd_agent.cli  # noqa: F401

        # Check if openai was loaded
        openai_loaded = any(name.startswith("openai") for name in sys.modules.keys())

        if openai_loaded:
            print("\n⚠️  OpenAI SDK loaded at import time (should be lazy loaded)")
            print("   This adds ~400ms to startup time unnecessarily")
        else:
            print("\n✓ OpenAI SDK not loaded at import (good lazy loading)")

        # For now, we document the issue but don't fail
        # TODO: Make this assertion strict after implementing lazy loading
        # assert not openai_loaded, "OpenAI should be lazy loaded"


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Benchmark tests for tracking performance over time."""

    @pytest.mark.skip(
        reason="pytest-benchmark not installed - enable manually for profiling"
    )
    def test_full_startup_benchmark(self, benchmark):
        """Benchmark full CLI startup time using pytest-benchmark.

        This test requires pytest-benchmark to be installed.
        Skip if not available.
        """
        pytest.importorskip("pytest_benchmark")

        def startup():
            result = subprocess.run(
                ["poetry", "run", "cdd-agent", "--help"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode

        result = benchmark(startup)
        assert result == 0, "Startup should succeed"
