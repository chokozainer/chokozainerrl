import os

class Test:
	def sayStr(self, str):
		print(str)
 
def makeDir(str):
	if not os.path.exists(str):
  		os.makedirs(str)

def sayStr(str):
	print(str)
