from bs4 import BeautifulSoup
import urllib
import re
r=urllib.urlopen('https://www.indeed.com/r/542fd3b35a00acae?sp=0').read()
soup=BeautifulSoup(r)
c = soup.find_all(["p","span","h2"])
print type(c[0])
f=open('resume/testfile.txt','a')
cl=re.compile('<.*?>')
for j in range(7,len(c)):
    i=str(c[j])
    cc=cl.sub('',i)
    #print cc
    f.write(cc)
    f.write('\n')
