import firebase_admin
from firebase_admin import firestore
from getadminkey import getadminkey

adminkey = getadminkey()
cred = firebase_admin.credentials.Certificate(adminkey)
firebase_admin.initialize_app(cred)
db = firestore.client()

for doc in db.collection('documentation').stream():
  if doc.to_dict()['type'] == 'activation':
    doc.reference.update({'equation': 'y = x^2'})
    doc.reference.update({'equationderivative': 'y = 2x'})