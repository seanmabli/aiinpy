import firebase_admin
from firebase_admin import firestore

cred = firebase_admin.credentials.Certificate('adminkey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

for doc in db.collection('documentation').stream():
  doc.reference.update({'description' : 'this is a description placeholder because there is currently no description in place.  a discription in this feild will discribe the function.'})