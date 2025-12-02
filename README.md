# IFC 3D Geometry Auto-Decoder

This repository contains the implementation of the paper **"Enabling AI-Driven Modular Building Design: An Auto-Decoder Approach for IFC 3D Geometry Representation"**. The project provides an auto-decoder framework for learning and generating 3D geometry representations from Industry Foundation Classes (IFC) data.

## Installation

This package was developed and tested on **Ubuntu 24.04**. We use [uv](https://github.com/astral-sh/uv) for package management.

### Prerequisites

First, install `uv` by following the instructions at: https://docs.astral.sh/uv/getting-started/installation/

### Setup

1. Clone this repository and navigate to the project folder:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Synchronize dependencies:
```bash
uv sync
```

3. Run the main script:
```bash
uv run start.py
```

> **Note:** On first run, the program will automatically create the necessary data directories.

