import yaml
from pathlib import Path

with open('_data/publications.yml', 'r', encoding='utf-8') as f:
    publications = yaml.safe_load(f)

output_dir = Path('_publications')
output_dir.mkdir(exist_ok=True)

for pub in publications:
    id = pub['id']
    filename = output_dir / f"{id}.md"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('---\n')
        for key, value in pub.items():
            # Convert list to YAML array
            if isinstance(value, list):
                f.write(f"{key}:\n")
                for item in value:
                    f.write(f"  - {item}\n")
            else:
                f.write(f"{key}: \"{value}\"\n")
        f.write("layout: publication\n")
        f.write('---\n')
