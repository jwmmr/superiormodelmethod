# install_requirements.py
import subprocess
import sys

def ensure_pip():
    """Ensure pip is installed, install it via ensurepip if not available."""
    try:
        import pip
        print("✔ pip is already installed.")
    except ImportError:
        print("❌ pip not found. Attempting to install pip...")
        try:
            import ensurepip
            ensurepip.bootstrap()
            print("✔ Successfully installed pip via ensurepip.")
        except Exception as e:
            print(f"❌ Failed to install pip. Error: {e}")
            input("Press Enter to exit...")
            sys.exit(1)

def install_packages():
    """Install required packages (Pillow and numpy) if missing."""
    required_packages = {
        'Pillow': 'PIL',  # Pillow is the package name, PIL is the module you import
        'numpy': 'numpy'
    }
    
    installed_packages = []
    missing_packages = []
    
    # Check which packages are already installed
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
            installed_packages.append(package)
        except ImportError:
            missing_packages.append(package)
    
    if not missing_packages:
        print("\n✔ All required packages are already installed:")
        for package in installed_packages:
            print(f"- {package}")
        return
    
    print("\n⚠ The following packages will be installed:")
    for package in missing_packages:
        print(f"- {package}")
    
    # Install missing packages using pip
    for package in missing_packages:
        try:
            print(f"\n⌛ Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✔ Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}. Error: {e}")
            input("Press Enter to exit...")
            sys.exit(1)
    
    print("\n✔ All required packages have been installed successfully!")

if __name__ == "__main__":
    print("=== Checking for pip ===")
    ensure_pip()
    
    print("\n=== Checking for required packages ===")
    install_packages()
    
    # Keep console open
    print("\n✅ Setup completed. Review output above for any errors.")
    input("Press Enter to exit...")
