import paramiko, getpass, json

def getadminkey():
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname='api.seanmabli.com', username='sean', password=getpass.getpass('password: '))
    return json.loads(' '.join(ssh_client.open_sftp().open('aiinpy/docs/admin/adminkey.json').readlines()))