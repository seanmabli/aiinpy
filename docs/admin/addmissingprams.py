import firebase_admin
from firebase_admin import firestore
from getadminkey import getadminkey

adminkey = getadminkey()
cred = firebase_admin.credentials.Certificate(adminkey)
firebase_admin.initialize_app(cred)
db = firestore.client()

for doc in db.collection('documentation').stream():
  if doc.to_dict()['version'] == '0.0.18':
    doc.reference.update({'equation' : 'y = x^2', 'equationderivative' : 'y = 2x', 'examples' : 'placeholder', 'graphx' : [], 'graphy' : [], 'parameters' : 'placeholder'})
    