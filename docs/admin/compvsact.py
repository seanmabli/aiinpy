import firebase_admin
from firebase_admin import firestore
from getadminkey import getadminkey
import numpy as np

adminkey = getadminkey()
cred = firebase_admin.credentials.Certificate(adminkey)
firebase_admin.initialize_app(cred)
db = firestore.client()

known = []
unknown = []

for doc in db.collection('documentation').stream():
  if doc.to_dict()['version'] != '0.0.18':
    known.append([doc.to_dict()['function'].lower(), doc.to_dict()['type']])
  else:
    unknown.append(doc.to_dict()['function'].lower())

print(len(known), len(unknown))

new = []

for a in unknown:
  for b in known:
    if a == b[0]:
      new.append([a, b[1]])

for doc in db.collection('documentation').stream():
  for a in new:
    if doc.to_dict()['version'] == '0.0.18' and doc.to_dict()['function'].lower() == a[0]:
      doc.reference.update({'type': a[1]})