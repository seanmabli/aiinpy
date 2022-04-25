import firebase_admin
from firebase_admin import firestore
import aiinpy

cred = firebase_admin.credentials.Certificate('adminkey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

for doc in db.collection('documentation').stream():
  # if doc.to_dict()['version'] == '0.0.17' and doc.to_dict()['type'] == 'activation' and doc.to_dict()['function'] == 'leakyrelu':
  #     activation = getattr(aiinpy, doc.to_dict()['function'])()
  #   x = range(-100, 100)
  #   y = [activation.forward(i) for i in x]
  #   doc.reference.update({'graphx': x})
    # doc.reference.update({'graphy': y})


  if doc.to_dict()['version'] == '0.0.18' and doc.to_dict()['type'] == 'activation':
    # try:
    print(doc.to_dict()['function'])
    if 'soft' not in doc.to_dict()['function'] and doc.to_dict()['graphx'] == [] and doc.to_dict()['function'] != 'mish':
      activation = getattr(aiinpy, doc.to_dict()['function'])()
      x = list(range(-4, 5))
      y = list(map(int, [activation.forward(i) for i in x]))
      print('a')
      doc.reference.update({'graphx': x})
      print('b')
      doc.reference.update({'graphy': y})
      print('c')
    # except:
    #   print('failed', doc.to_dict()['function'], x, y)
    #   doc.reference.update({'graphx': []})
    #   doc.reference.update({'graphy': []})
  # if doc.to_dict()['version'] != ('0.0.17' or '0.0.18') or doc.to_dict()['function'] == ('stablesoftmax' or 'softmax'):
  #   doc.reference.update({'graphx': []})
  #   doc.reference.update({'graphy': []})