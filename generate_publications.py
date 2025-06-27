import yaml
from pathlib import Path

def write_yaml_value(f, key, value, indent=0):
    prefix = '  ' * indent
    if isinstance(value, dict):
        f.write(f"{prefix}{key}:\n")
        for sub_key, sub_val in value.items():
            write_yaml_value(f, sub_key, sub_val, indent + 1)
    elif isinstance(value, list):
        f.write(f"{prefix}{key}:\n")
        for item in value:
            f.write(f"{prefix}  - {item}\n")
    else:
        # Quote only strings
        if isinstance(value, str):
            value = value.replace('"', '\\"')  # Escape quotes
            f.write(f'{prefix}{key}: "{value}"\n')
        else:
            f.write(f"{prefix}{key}: {value}\n")

# Load the YAML
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
            write_yaml_value(f, key, value)
        f.write("layout: publication\n")
        f.write('---\n')
