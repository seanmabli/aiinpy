import subprocess
import firebase_admin
from firebase_admin import firestore

cred = firebase_admin.credentials.Certificate('adminkey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()
docs = db.collection('documentation').stream()

for doc in docs:
    doc.reference.delete()

versions = str(subprocess.Popen('pip index versions aiinpy', stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0])
versions = versions[versions.find('Available versions') + 20: versions.find('0.0.11') + 6].replace(" ", "").split(",")

for version in versions:
    subprocess.run('pip install aiinpy==' + version)
    subprocess.run('python3 extractdata.py')
