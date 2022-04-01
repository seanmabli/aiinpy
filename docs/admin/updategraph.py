import firebase_admin
from firebase_admin import firestore
import aiinpy

cred = firebase_admin.credentials.Certificate('adminkey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

activation = aiinpy.sigmoid().forward

x = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
y = [activation(i) for i in x]

for doc in db.collection('documentation').stream():
  if doc.to_dict()['type'] == 'activation':
    doc.reference.update({'graphx': x})
    doc.reference.update({'graphy': y})