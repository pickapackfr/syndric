This folder contains configuration for VS Code Dev Containers.

Quick start

- In VS Code: open the Command Palette and choose `Remote-Containers: Reopen in Container`.
- The container will be built from the included `Dockerfile` and `devcontainer.json`.
- After the container is created, the `postCreateCommand` will attempt to run:
  - `python -m pip install --upgrade pip`
  - `pip install -e .` (installs the project editable if a `pyproject.toml` exists)
  - `pip install pytest`

Notes / Troubleshooting

- If you already have a local `.venv` and don't want it mounted, remove the `mounts` entry in `devcontainer.json`.
- If your project uses Poetry or other tools, adjust `postCreateCommand` accordingly.
- Use the `Remote - Containers` extension to control rebuilds and attach to the container.
