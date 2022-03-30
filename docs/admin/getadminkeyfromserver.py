import paramiko
import os

password = input('password: ')
os.system('clear')

ssh_client=paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh_client.connect(hostname='api.seanmabli.com',username='sean',password=password)
stdin,stdout,stderr=ssh_client.exec_command('cd aiinpy/docs/admin ; ls')
print(stdout.readlines())
sftp_client = ssh_client.open_sftp()
remote_file = sftp_client.open('aiinpy/docs/admin/additem.py')
try:
    for line in remote_file:
        print(line)
finally:
    remote_file.close()