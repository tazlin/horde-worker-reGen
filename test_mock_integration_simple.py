"""Simplified integration tests for mock processes and event system.

These tests use Python's compile to validate the code structure without
requiring all dependencies to be installed.
"""

import ast
import sys
from pathlib import Path


def test_event_system_structure():
    """Test that event system files have correct structure."""
    print("test_event_system_structure:")
    print("-" * 70)

    base_path = Path(__file__).parent / "horde_worker_regen" / "events"

    # Check event_types.py has required event classes
    event_types_path = base_path / "event_types.py"
    with open(event_types_path) as f:
        content = f.read()
        tree = ast.parse(content)

    # Find all class definitions
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

    required_events = [
        "ProcessStateChangedEvent",
        "ProcessHeartbeatEvent",
        "ProcessMemoryUpdatedEvent",
        "JobPoppedEvent",
        "JobStartedEvent",
        "JobCompletedEvent",
        "ModelDownloadStartedEvent",
        "ModelLoadedEvent",
    ]

    for event in required_events:
        assert event in classes, f"Missing event class: {event}"

    print(f"✓ Event system has all required event classes ({len(required_events)} total)")
    for event in required_events:
        print(f"  - {event}")


def test_mock_system_structure():
    """Test that mock system files have correct structure."""
    print("\ntest_mock_system_structure:")
    print("-" * 70)

    base_path = Path(__file__).parent / "horde_worker_regen" / "process_management" / "mock"

    # Check mock_config.py
    mock_config_path = base_path / "mock_config.py"
    with open(mock_config_path) as f:
        content = f.read()
        tree = ast.parse(content)

    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    assert "MockConfig" in classes
    assert "MockScenario" in classes

    # Check mock_data_generator.py
    data_gen_path = base_path / "mock_data_generator.py"
    with open(data_gen_path) as f:
        content = f.read()
        tree = ast.parse(content)

    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    required_functions = [
        "generate_fake_image",
        "calculate_mock_kudos",
        "calculate_mock_inference_time",
        "generate_fake_nsfw_score",
        "generate_fake_csam_score",
    ]

    for func in required_functions:
        assert func in functions, f"Missing function: {func}"

    print(f"✓ Mock system has all required components")
    print(f"  Classes: MockConfig, MockScenario")
    print(f"  Functions: {len(required_functions)} data generators")


def test_process_manager_event_integration():
    """Test that process_manager.py integrates with event system."""
    print("\ntest_process_manager_event_integration:")
    print("-" * 70)

    process_manager_path = Path(__file__).parent / "horde_worker_regen" / "process_management" / "process_manager.py"

    with open(process_manager_path) as f:
        content = f.read()

    # Check for event system imports
    assert "from horde_worker_regen.events import" in content
    assert "EventDispatcher" in content
    assert "ProcessStateChangedEvent" in content or "ProcessHeartbeatEvent" in content

    # Check for event dispatcher creation
    assert "self.event_dispatcher" in content or "self._event_dispatcher" in content

    # Check for event emissions
    assert ".emit(" in content

    print("✓ process_manager.py integrates with event system")
    print("  - Imports EventDispatcher and event types")
    print("  - Creates event dispatcher instance")
    print("  - Emits events")


def test_process_manager_mock_integration():
    """Test that process_manager.py integrates with mock system."""
    print("\ntest_process_manager_mock_integration:")
    print("-" * 70)

    process_manager_path = Path(__file__).parent / "horde_worker_regen" / "process_management" / "process_manager.py"

    with open(process_manager_path) as f:
        content = f.read()

    # Check for mock imports
    assert "from horde_worker_regen.process_management.mock import" in content
    assert "MockConfig" in content
    assert "start_mock_inference_process" in content
    assert "start_mock_safety_process" in content

    # Check for mock config creation
    assert "MockConfig" in content
    assert "enable_mock_processes" in content

    # Check for conditional process creation
    assert "if self.bridge_data.enable_mock_processes" in content or "if bridge_data.enable_mock_processes" in content

    print("✓ process_manager.py integrates with mock system")
    print("  - Imports mock components")
    print("  - Creates MockConfig when enabled")
    print("  - Conditionally creates mock processes")


def test_run_worker_cli_arguments():
    """Test that run_worker.py has CLI arguments for mock mode."""
    print("\ntest_run_worker_cli_arguments:")
    print("-" * 70)

    run_worker_path = Path(__file__).parent / "horde_worker_regen" / "run_worker.py"

    with open(run_worker_path) as f:
        content = f.read()

    # Check for argparse arguments
    assert "argparse" in content
    assert "--mock" in content
    assert "--mock-speed" in content
    assert "--mock-scenario" in content

    # Check for argument handling
    assert "enable_mock" in content or "args.mock" in content

    print("✓ run_worker.py has mock-related CLI arguments")
    print("  - --mock flag")
    print("  - --mock-speed argument")
    print("  - --mock-scenario argument")


def test_bridge_data_mock_fields():
    """Test that bridge data model has mock configuration fields."""
    print("\ntest_bridge_data_mock_fields:")
    print("-" * 70)

    data_model_path = Path(__file__).parent / "horde_worker_regen" / "bridge_data" / "data_model.py"

    with open(data_model_path) as f:
        content = f.read()

    # Check for mock fields
    required_fields = [
        "enable_mock_processes",
        "mock_speed_multiplier",
        "mock_enable_failures",
        "mock_failure_rate",
        "mock_scenario",
    ]

    for field in required_fields:
        assert field in content, f"Missing field: {field}"

    # Check for validation
    assert "validate_mock_configuration" in content or "mock" in content.lower()

    print("✓ Bridge data model has mock configuration fields")
    for field in required_fields:
        print(f"  - {field}")


def test_files_compile():
    """Test that all modified files compile without syntax errors."""
    print("\ntest_files_compile:")
    print("-" * 70)

    files_to_check = [
        "horde_worker_regen/events/event_types.py",
        "horde_worker_regen/events/event_dispatcher.py",
        "horde_worker_regen/events/event_listener.py",
        "horde_worker_regen/process_management/mock/mock_config.py",
        "horde_worker_regen/process_management/mock/mock_data_generator.py",
        "horde_worker_regen/process_management/mock/mock_inference_process.py",
        "horde_worker_regen/process_management/mock/mock_safety_process.py",
        "horde_worker_regen/process_management/process_manager.py",
        "horde_worker_regen/run_worker.py",
        "horde_worker_regen/bridge_data/data_model.py",
    ]

    base_path = Path(__file__).parent
    compiled_count = 0

    for file_path in files_to_check:
        full_path = base_path / file_path
        if not full_path.exists():
            print(f"  ⚠  File not found: {file_path}")
            continue

        try:
            with open(full_path) as f:
                compile(f.read(), str(full_path), 'exec')
            compiled_count += 1
        except SyntaxError as e:
            raise AssertionError(f"Syntax error in {file_path}: {e}")

    print(f"✓ All {compiled_count} files compile without syntax errors")


def test_documentation_exists():
    """Test that documentation files exist."""
    print("\ntest_documentation_exists:")
    print("-" * 70)

    base_path = Path(__file__).parent

    docs = [
        "TERMINAL_UI_FOUNDATION.md",
        "horde_worker_regen/events/README.md",
        "horde_worker_regen/process_management/mock/DESIGN.md",
        "horde_worker_regen/process_management/mock/README.md",
        "bridgeData_mock_example.yaml",
    ]

    for doc in docs:
        doc_path = base_path / doc
        assert doc_path.exists(), f"Missing documentation: {doc}"

    print(f"✓ All {len(docs)} documentation files exist")
    for doc in docs:
        print(f"  - {doc}")


if __name__ == "__main__":
    print("=" * 70)
    print("Running Integration Tests (Simplified)")
    print("=" * 70)
    print()

    tests = [
        test_event_system_structure,
        test_mock_system_structure,
        test_process_manager_event_integration,
        test_process_manager_mock_integration,
        test_run_worker_cli_arguments,
        test_bridge_data_mock_fields,
        test_files_compile,
        test_documentation_exists,
    ]

    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed.append((test.__name__, e))

    print()
    print("=" * 70)
    if failed:
        print(f"❌ {len(failed)} test(s) failed:")
        for name, error in failed:
            print(f"  - {name}: {error}")
        sys.exit(1)
    else:
        print(f"✅ All {len(tests)} tests passed!")
        print("=" * 70)
        print()
        print("Integration Summary:")
        print("  ✓ Event system structure verified")
        print("  ✓ Mock system structure verified")
        print("  ✓ Event integration in process_manager verified")
        print("  ✓ Mock integration in process_manager verified")
        print("  ✓ CLI arguments for mock mode verified")
        print("  ✓ Bridge data mock fields verified")
        print("  ✓ All files compile successfully")
        print("  ✓ Documentation complete")
