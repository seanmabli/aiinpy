import paramiko, getpass
import firebase_admin
from firebase_admin import firestore
import os

try:
    ssh_client=paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname='api.seanmabli.com',username='sean',password=getpass.getpass('password: '))
    open('adminkey.json', 'w').write(' '.join(ssh_client.open_sftp().open('aiinpy/docs/admin/adminkey.json').readlines()))
    cred = firebase_admin.credentials.Certificate('adminkey.json')
    firebase_admin.initialize_app(cred)
    db = firestore.client()
except:
    os.system('rm adminkey.json')
else:
    os.system('rm adminkey.json')