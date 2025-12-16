# Elessar

## Downloading Zig

This project includes scripts to automatically download and install Zig 0.15.2.

### Using the Cross-Platform Script (Recommended)

The `zig/download.ps1` script works on macOS, Linux, and Windows:

```bash
# On macOS or Linux
./zig/download.ps1

# On Windows (PowerShell)
.\zig\download.ps1
```

This script automatically detects your operating system and runs the appropriate download script.

### Platform-Specific Scripts

Alternatively, you can run the platform-specific scripts directly:

**macOS/Linux:**
```bash
./zig/download.sh
```

**Windows:**
```powershell
.\zig\download.win.ps1
```

### What the Scripts Do

The download scripts will:
- Download Zig 0.15.2 for your platform (aarch64 or x86_64)
- Verify the download using SHA-256 checksums
- Extract the archive
- Install Zig to the `zig/` directory

After downloading, the Zig compiler will be available at:
- **macOS/Linux:** `./zig/zig`
- **Windows:** `.\zig\zig.exe`

You can add this to your PATH if desired, or run it directly from the project directory.
