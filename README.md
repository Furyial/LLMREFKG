# LLM-Reflex-GeoKG

LLM-Reflex-GeoKG (A Reflexion-Enhanced LLM Framework for Automated Geographic Knowledge Graph Construction) offers the open-source scaffolding of our pipeline: in-house prompts, datasets, and credentials remain private while this repo exposes the orchestration scripts that connect LLM-driven NER, relation extraction, and entity alignment into a reproducible GeoKG workflow.

## Prerequisites
- Python 3.10+
- Neo4j 5.x (running and accessible)
- MySQL 8.x (optional, only if you plan to store auxiliary metadata)
- Recommended: virtual environment (`venv` or `conda`)

## Installation
```bash
pip install -r requirements.txt
```
## Usage
1. **Prepare Neo4j data**  
   ```bash
   python dataloaderneo4j.py
   ```

2. **Run NER pipeline**  
   ```bash
   python ApiGPT.py
   ```

3. **Extract relations with DeepSeek**  
   ```bash
   python deepskre.py
   ```

4. **Align entities across sources**  
   ```bash
   python EntityAlignment.py
   ```

5. **Full Reflxion workflow **  
   ```bash
   python reflxion.py
   ```



