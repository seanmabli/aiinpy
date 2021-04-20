import os
from datetime import date
os.system("cd agi")
GitStatusInfo = os.popen("git status").read()

# New Files
while(GitStatusInfo.find('Untracked files') != -1):
  GitCommand = "git add " + str(GitStatusInfo[GitStatusInfo.find('Untracked files') + 83:GitStatusInfo[GitStatusInfo.find('Untracked files') + 83:].find("\n") + GitStatusInfo.find('Untracked files') + 83])
  os.system(GitCommand)
  GitStatusInfo = os.popen("git status").read()
  
GitStatusInfo = GitStatusInfo[GitStatusInfo.find('Changes not staged for commit'):]

# Modified Files
while(GitStatusInfo.find('Changes not staged for commit') != -1):
  if(GitStatusInfo.find('mod') != -1):
    GitCommand = "git add " + str(GitStatusInfo[GitStatusInfo.find('mod') + 12:GitStatusInfo[GitStatusInfo.find('mod') + 12:].find("\n") + GitStatusInfo.find('mod') + 12])
  if(GitStatusInfo.find('del') != -1):
    GitCommand = "git rm " + str(GitStatusInfo[GitStatusInfo.find('del') + 12:GitStatusInfo[GitStatusInfo.find('del') + 12:].find("\n") + GitStatusInfo.find('del') + 12])
  os.system(GitCommand)
  GitStatusInfo = os.popen("git status").read()[GitStatusInfo.find('Changes not staged for commit'):]
  
# Commit Everything
os.system("git commit -m '" + str(date.today()) + "'")
os.system("git remote add origin git@github.com:Sean-Mabli/agi.git")
os.system("git branch -M main")
os.system("git push -u origin main")