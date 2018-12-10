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
# follow this to install anaconda? : https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04
# install tensor flow: https://anaconda.org/conda-forge/tensorflow
# update: http://fizzylogic.nl/2017/05/08/monitor-progress-of-your-keras-based-neural-network-using-tensorboard/ Juypter Notebook
# run a tensorboard --logdir=gs://model_output --port=8080

# https://stackoverflow.com/questions/17903859/why-cant-i-ping-the-external-ip-address-of-my-google-compute-engine-instance
# https://www.google.com/search?q=Error+loading+server+extension+jupyter_tensorboard&oq=Error+loading+server+extension+jupyter_tensorboard&aqs=chrome..69i57j0.553j0j1&sourceid=chrome&ie=UTF-8
#https://github.com/lspvic/jupyter_tensorboard

# tensorboard --logdir=logs --port=8008

#https://piazza.com/class/jme0l06lytg2pz?cid=436
