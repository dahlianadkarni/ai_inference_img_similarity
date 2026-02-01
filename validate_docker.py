#!/usr/bin/env python3
"""
Validate Docker configuration for Step 2.

This checks that all Docker files are properly configured without requiring Docker to be installed.
"""

import sys
from pathlib import Path

def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists."""
    path = Path(filepath)
    if path.exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} NOT FOUND")
        return False

def check_dockerfile_syntax(dockerfile: str) -> bool:
    """Basic validation of Dockerfile syntax."""
    path = Path(dockerfile)
    if not path.exists():
        return False
    
    content = path.read_text()
    required_instructions = ["FROM", "WORKDIR", "COPY", "RUN", "EXPOSE", "CMD"]
    
    missing = []
    for instruction in required_instructions:
        if instruction not in content:
            missing.append(instruction)
    
    if missing:
        print(f"✗ {dockerfile} missing instructions: {', '.join(missing)}")
        return False
    
    # Check for health check
    if "HEALTHCHECK" in content:
        print(f"✓ {dockerfile} has HEALTHCHECK")
    
    # Check for environment variables
    if "ENV MODEL_NAME" in content or "MODEL_NAME" in content:
        print(f"✓ {dockerfile} uses environment variables")
    
    return True

def check_dockerignore() -> bool:
    """Validate .dockerignore exists and has content."""
    path = Path(".dockerignore")
    if not path.exists():
        print("✗ .dockerignore not found")
        return False
    
    content = path.read_text()
    essential_patterns = ["__pycache__", "*.pyc", ".git", "venv"]
    
    found = sum(1 for pattern in essential_patterns if pattern in content)
    if found >= 3:
        print(f"✓ .dockerignore has essential patterns ({found}/{len(essential_patterns)})")
        return True
    else:
        print(f"⚠ .dockerignore may be incomplete ({found}/{len(essential_patterns)} patterns)")
        return False

def check_docker_compose() -> bool:
    """Validate docker-compose.yml."""
    path = Path("docker-compose.yml")
    if not path.exists():
        print("✗ docker-compose.yml not found")
        return False
    
    content = path.read_text()
    required = ["services:", "inference-service:", "ports:", "environment:", "healthcheck:"]
    
    missing = [r for r in required if r not in content]
    if missing:
        print(f"✗ docker-compose.yml missing: {', '.join(missing)}")
        return False
    
    print("✓ docker-compose.yml structure looks good")
    return True

def check_server_config() -> bool:
    """Check that server.py accepts environment variables."""
    server_path = Path("src/inference_service/server.py")
    if not server_path.exists():
        print("✗ server.py not found")
        return False
    
    content = server_path.read_text()
    
    # Check for argparse or environment variable handling
    has_argparse = "argparse" in content or "ArgumentParser" in content
    has_env = "os.getenv" in content or "os.environ" in content
    
    if has_argparse and has_env:
        print("✓ server.py accepts CLI args and environment variables")
        return True
    elif has_argparse:
        print("✓ server.py accepts CLI args")
        return True
    elif has_env:
        print("✓ server.py reads environment variables")
        return True
    else:
        print("⚠ server.py may not be configurable via env vars")
        return False

def main():
    """Run all validation checks."""
    print("="*60)
    print("Docker Configuration Validation (Step 2)")
    print("="*60)
    print()
    
    results = {}
    
    # Check file existence
    print("1. Checking required files...")
    results["dockerfile"] = check_file_exists("Dockerfile", "Main Dockerfile")
    results["dockerfile_gpu"] = check_file_exists("Dockerfile.gpu", "GPU Dockerfile")
    results["dockerignore"] = check_file_exists(".dockerignore", ".dockerignore")
    results["compose"] = check_file_exists("docker-compose.yml", "docker-compose.yml")
    results["run_script"] = check_file_exists("docker-run.sh", "Run script")
    results["compose_script"] = check_file_exists("docker-compose.sh", "Compose script")
    print()
    
    # Validate Dockerfile syntax
    print("2. Validating Dockerfile syntax...")
    if results["dockerfile"]:
        results["dockerfile_valid"] = check_dockerfile_syntax("Dockerfile")
    if results["dockerfile_gpu"]:
        results["dockerfile_gpu_valid"] = check_dockerfile_syntax("Dockerfile.gpu")
    print()
    
    # Validate .dockerignore
    print("3. Validating .dockerignore...")
    if results["dockerignore"]:
        results["dockerignore_valid"] = check_dockerignore()
    print()
    
    # Validate docker-compose.yml
    print("4. Validating docker-compose.yml...")
    if results["compose"]:
        results["compose_valid"] = check_docker_compose()
    print()
    
    # Check server configuration
    print("5. Checking server.py configuration...")
    results["server_config"] = check_server_config()
    print()
    
    # Summary
    print("="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"Passed: {passed}/{total} checks")
    print()
    
    if passed == total:
        print("✅ Step 2 configuration is COMPLETE and VALID!")
        print()
        print("Next steps:")
        print("1. Install Docker Desktop (if not already installed)")
        print("2. Run: ./docker-run.sh")
        print("3. Test: curl http://localhost:8002/health")
        print("4. View docs: http://localhost:8002/docs")
        return 0
    else:
        print("⚠ Some checks failed. Review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
