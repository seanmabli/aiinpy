import firebase_admin
from firebase_admin import firestore
import aiinpy
import numpy as np

cred = firebase_admin.credentials.Certificate('adminkey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

for doc in db.collection('documentation').stream():
  print(doc.to_dict()['function'])
  if doc.to_dict()['version'] == '0.0.18' and doc.to_dict()['type'] == 'activation' and doc.to_dict()['function'] != 'softplus' and doc.to_dict()['function'] != 'leakyrelu' and doc.to_dict()['function'] != 'elu' and doc.to_dict()['function'] != 'prelu' and doc.to_dict()['function'] != 'mish':
    activation = getattr(aiinpy, doc.to_dict()['function'])()
    x = list(np.linspace(-4, 5, 1000))
    y = list(np.round([activation.forward(i) for i in x], 3))
    doc.reference.update({'graphx': x})
    doc.reference.update({'graphy': y})