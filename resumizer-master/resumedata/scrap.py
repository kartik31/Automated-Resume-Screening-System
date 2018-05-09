from bs4 import BeautifulSoup
import urllib
import re,os

#change link to resume search page of indeed.com 
links = urllib.urlopen("https://www.indeed.com/resumes?q=anytitle%3A%28software+engineer%29&l=")

soup = BeautifulSoup(links)
arr=[]
for link in soup.findAll('a', attrs={'href': re.compile("/r/")}):
    arr.append(str(link.get('href')))
#print(arr)
a="https://www.indeed.com"
#change position here an
position = 'software_engineer'
if not os.path.exists(position):
    os.makedirs(position)
path = position+'/resume'
ext='.txt'
for i in range(0,len(arr)):
    r=urllib.urlopen(a+arr[i]).read()
    soup=BeautifulSoup(r)
    c = soup.find_all(["p","span","h2"])
    name=path+str(i+1)+ext
    print name
    f=open(name,'a')
    cl=re.compile('<.*?>')
    for j in range(7,len(c)):
        i=str(c[j])
        cc=cl.sub('',i)
        #print cc
        f.write(cc)
        f.write('\n')