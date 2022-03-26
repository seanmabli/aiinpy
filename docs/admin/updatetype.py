import firebase_admin
from firebase_admin import firestore

cred = firebase_admin.credentials.Certificate('adminkey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

for doc in db.collection('documentation').stream():

  inx = input(doc.to_dict()['title'] + ': ')
  if inx == 'a':
    out = 'activation'
  elif inx == 'c':
    out = 'computation'

  db.collection('documentation').document(doc.id).update({'type' : out})