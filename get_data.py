import requests
url = 'http://peipa.essex.ac.uk/pix/mias/all-mias.tar.gz'
r = requests.get(url, allow_redirects=True)
open('all-mias.tar.gz','wb').write(r.content)

#folow this with in the shell:
# sudo apt-get update
# sudo apt-get p7zip-full
# 7z e all-mias.tar.gz  && 7z x all-mias.tar
# rm all-mias.tar.gz
# rm all-mias.tar
