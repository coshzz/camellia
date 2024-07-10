import requests
import io
import pickle
import bz2

############################################################################
def dump_pickle_bz2(data, name, directory='./'):
    fhnd = open(f'{directory}/{name}.pickle.bz2', 'wb')
    fhnd.write(bz2.compress(pickle.dumps(data)))
    fhnd.seek(0)
    fhnd.close()

def load_pickle_bz2(name, directory='./'):
    fhnd = open(f'{directory}/{name}.pickle.bz2', 'rb')
    dataz = fhnd.read()
    fhnd.close()
    return pickle.loads(bz2.decompress(dataz))
    
############################################################################
def dump_to_linode_pickle_bz2(data, name, directory='test/kaggle'):
    pkbz2fhnd = io.BytesIO()
    pkbz2fhnd.write(bz2.compress(pickle.dumps(data)))
    pkbz2fhnd.seek(0)

    url = 'http://172.104.36.130:8080/uploadfile'
    files = {'file': (name+'.pickle.bz2', pkbz2fhnd, 'application/octet-stream')}
    params = {'directory': directory}
    r = requests.post(url, files=files, params=params)

    return r.json()

def load_from_linode_pickle_bz2(name, directory='test/kaggle'):
    url = 'http://172.104.36.130:8080/data/' + directory + '/' + name + '.pickle.bz2'
    r = requests.get(url)
    return pickle.loads(bz2.decompress(r.content))
