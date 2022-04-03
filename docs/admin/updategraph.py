import firebase_admin
from firebase_admin import firestore
import aiinpy

cred = firebase_admin.credentials.Certificate('adminkey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

for doc in db.collection('documentation').stream():
  if doc.to_dict()['version'] == '0.0.17' and doc.to_dict()['type'] == 'activation':
    try:
      activation = getattr(aiinpy, doc.to_dict()['function'])()

      x = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
      y = [activation.forward(i) for i in x]
      doc.reference.update({'graphx': x})
      doc.reference.update({'graphy': y})
    except:
      print(doc.to_dict()['function'])
      doc.reference.update({'graphx': []})
      doc.reference.update({'graphy': []})
  if doc.to_dict()['version'] != '0.0.17' or doc.to_dict()['function'] == ('stablesoftmax' or 'softmax'):
    doc.reference.update({'graphx': []})
    doc.reference.update({'graphy': []})