import os
import sys

try:
    import gcsfs
    import yaml
except ImportError:
    print("Please install required packages:")
    print("pip install gcsfs pyyaml")
    sys.exit(1)

def update_config():
    # Setup GCS
    print("Fetching datasets from GCS (sunflower-data/speech)...")
    
    try:
        gcs = gcsfs.GCSFileSystem(project='sb-gcp-project-01', token='gcs-data-viewer-key.json')
        bucket_path = "sunflower-data/speech"
        all_datasets = gcs.ls(path=bucket_path, detail=False)
    except Exception as e:
        print(f"Error accessing GCS: {e}")
        return

    # Extract dataset names (e.g., 'ach_salt' from 'sunflower-data/speech/ach_salt')
    dataset_names = []
    for path in all_datasets:
        name = path.split('/')[-1]
        if name:
            dataset_names.append(name)
            
    dataset_names = sorted(dataset_names)

    if not dataset_names:
        print("No datasets found or error in listing paths. Exiting.")
        return

    # Extract unique language codes (e.g., 'ach' from 'ach_salt')
    languages = sorted(list(set(name.split('_')[0] for name in dataset_names if '_' in name)))

    # Construct the dataset lists for train and dev
    train_datasets = [{'path': f'gcs://sunflower-data/speech/{name}/train/*.parquet'} for name in dataset_names]
    dev_datasets = [{'path': f'gcs://sunflower-data/speech/{name}/dev/*.parquet'} for name in dataset_names]

    config_path = 'configs/whisper_finetuning_gcs.yaml'

    print(f"Loading {config_path}...")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML: {e}")
        return

    # Function to create flow style lists (for inline lists like [ach, afr, ...])
    class FlowList(list):
        pass

    def flow_list_representer(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    yaml.add_representer(FlowList, flow_list_representer)

    # Update train configuration
    if 'train' in config:
        config['train']['datasets'] = train_datasets
        if 'source' in config['train'] and 'language' in config['train']['source']:
            config['train']['source']['language'] = FlowList(languages)
        if 'target' in config['train'] and 'language' in config['train']['target']:
            config['train']['target']['language'] = FlowList(languages)

    # Update validation configuration
    if 'validation' in config:
        config['validation']['datasets'] = dev_datasets
        if 'source' in config['validation'] and 'language' in config['validation']['source']:
            config['validation']['source']['language'] = FlowList(languages)
        if 'target' in config['validation'] and 'language' in config['validation']['target']:
            config['validation']['target']['language'] = FlowList(languages)

    print(f"Updating {config_path}...")
    try:
        with open(config_path, 'w') as f:
            # default_flow_style=False keeps mappings in block style
            # sort_keys=False preserves the insertion order of keys for python 3.7+ dictionaries
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Successfully updated {config_path}")
        print(f"- Added {len(dataset_names)} datasets with matching train/dev paths.")
        print(f"- Updated languages list with {len(languages)} languages: {', '.join(languages)}")
    except Exception as e:
        print(f"Error writing YAML: {e}")

if __name__ == "__main__":
    update_config()
