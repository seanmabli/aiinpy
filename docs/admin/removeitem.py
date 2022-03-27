import firebase_admin
from firebase_admin import firestore

cred = firebase_admin.credentials.Certificate('adminkey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

itemname = input('item name: ')

for doc in db.collection('documentation').stream():
  db.collection('documentation').document(doc.id).update({itemname : firestore.DELETE_FIELD})