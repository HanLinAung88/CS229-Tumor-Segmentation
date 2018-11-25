import requests
url = 'http://peipa.essex.ac.uk/pix/mias/all-mias.tar.gz'
r = requests.get(url, allow_redirects=True)
open('all-mias.tar.gz','wb').write(r.content)
