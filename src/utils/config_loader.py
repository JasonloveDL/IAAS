import json
import os.path

config_root = "../config"
NASConfig = None
modulesConfig = None
modulesList = None


def load_config(nasConfig_file, modules_file):
    global NASConfig
    global modulesConfig
    global modulesList

    with open(os.path.join(config_root, nasConfig_file)) as f:
        if NASConfig is None:
            NASConfig = json.load(f)
        else:
            tmp = json.load(f)
            for k in tmp:
                NASConfig[k] = tmp[k]
        if not os.path.exists(NASConfig['OUT_DIR']):
            os.makedirs(NASConfig['OUT_DIR'])
    with open(os.path.join(config_root, modules_file)) as f:
        if modulesConfig is None:
            modulesConfig = json.load(f)
        else:
            tmp = json.load(f)
            for k in tmp:
                modulesConfig[k] = tmp[k]

    NASConfig['editable'] = []
    for k in modulesConfig.keys():
        if modulesConfig[k]['editable']:
            NASConfig['editable'].append(k)
    if modulesList is None:
        modulesList = list(modulesConfig.keys())
    else:
        modulesList.clear()
        modulesList.extend(list(modulesConfig.keys()))


load_config('NASConfig.json', 'modules_rnn.json')
