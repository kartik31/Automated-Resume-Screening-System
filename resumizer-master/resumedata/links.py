from bs4 import BeautifulSoup
import urllib
import re
 
html_page = urllib.urlopen("https://www.indeed.com/resumes?q=anytitle%3A%28software+engineer%29&l=")
soup = BeautifulSoup(html_page)
i=1
for link in soup.findAll('a', attrs={'href': re.compile("/r/")}):
    print link.get('href')
    print(i)
    i+=1