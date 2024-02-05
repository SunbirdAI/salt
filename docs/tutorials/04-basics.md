### one-to-multiple translation: English text to Luganda and Acholi text

```python

import sys
sys.path.append('../..')
import leb.dataset
import leb.utils
import yaml

```

set up the configs

```python

yaml_config = '''
huggingface_load:   
  path: Sunbird/salt
  split: train
  name: text-all
source:
  type: text
  language: eng
  preprocessing:
      - prefix_target_language
target:
  type: text
  language: [lug, ach]
'''

config = yaml.safe_load(yaml_config)
ds = leb.dataset.create(config)
list(ds.take(5))

```
output

```
[{'source': '>>lug<< Eggplants always grow best under warm conditions.',
  'target': 'Bbiringanya lubeerera  asinga kukulira mu mbeera ya bugumu'},
 {'source': '>>ach<< Eggplants always grow best under warm conditions.',
  'target': 'Bilinyanya pol kare dongo maber ka lyeto tye'},
 {'source': '>>lug<< Farmland is sometimes a challenge to farmers.',
  'target': "Ettaka ly'okulimirako n'okulundirako ebiseera ebimu kisoomooza abalimi"},
 {'source': '>>ach<< Farmland is sometimes a challenge to farmers.',
  'target': 'Ngom me pur i kare mukene obedo peko madit bot lupur'},
 {'source': '>>lug<< Farmers should be encouraged to grow more coffee.',
  'target': 'Abalimi balina okukubirizibwa okwongera okulima emmwanyi'}]

```

This is how a basic data loader works