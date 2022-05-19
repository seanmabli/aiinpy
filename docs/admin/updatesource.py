import firebase_admin
from firebase_admin import firestore

cred = firebase_admin.credentials.Certificate('adminkey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

for doc in db.collection('documentation').stream():
  if doc.to_dict()['type'] == 'activation':
    doc.reference.update({'sourcecode': 'https://github.com/seanmabli/aiinpy/blob/' + doc.to_dict()['version'] + '/aiinpy/activation.py'})