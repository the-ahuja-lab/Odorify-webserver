def multi():
	print('printed from crontab')
	f = open('crontab-gen-file','w')
	print("yo", f)
	f.close()