# flake.nix
{
  description = "Python project with uv, LSP, and formatter";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            # Python & uv
            pkgs.python313
            pkgs.uv

            # LSP (Language Server Protocol)
            pkgs.pyright        # Type checker & LSP
            pkgs.python313Packages.python-lsp-server  # Alternative LSP

            # Formatter & Linter
            pkgs.ruff           # Fast formatter + linter (recommended)
            pkgs.black          # Alternative formatter
            pkgs.isort          # Import sorting

            pkgs.python313Packages.debugpy

            pkgs.py-spy
          ];

          shellHook = ''
            echo "=========================================="
            echo "🐍 Python Development Environment Ready!"
            echo "=========================================="
            echo "Python: $(python --version)"
            echo "uv:     ${pkgs.uv.version}"
            echo "ruff:   ${pkgs.ruff.version}"
            echo "pyright:${pkgs.pyright.version}"
            echo "=========================================="
            
            # Auto-create venv if missing
            if [ ! -d ".venv" ]; then
              echo "Creating virtual environment..."
              uv venv
            fi
            
            # Activate the venv
            source .venv/bin/activate
            echo "✅ Virtual environment activated"
          '';
        };
      }
    );
}

