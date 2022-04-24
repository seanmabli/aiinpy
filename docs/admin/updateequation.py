import firebase_admin
from firebase_admin import firestore

cred = firebase_admin.credentials.Certificate('adminkey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

for doc in db.collection('documentation').stream():
  if doc.to_dict()['type'] == 'activation':
    if doc.to_dict()['equation'] == 'y = x^2':
      doc.reference.update({'equation': ''})
    if doc.to_dict()['equationderivative'] == 'y = 2x':
      doc.reference.update({'equationderivative': ''})
  else:
    doc.reference.update({'equation' : firestore.DELETE_FIELD})
    doc.reference.update({'equationderivative' : firestore.DELETE_FIELD})