from uuid import uuid4
from htmlmin.decorators import minified_response
from django.http import request
from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.http import JsonResponse
import pandas as pd
import os
import shutil
from .models import queuedisp,result,disp,disp4,disp2,disp3
from zipfile import ZipFile 
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import traceback
import uuid

root = os.path.abspath('./')

def get_id(request):
	if 'id' in request.session:
		return str(request.session['id'])[:10]
	else:
		id = uuid.uuid4()
		request.session['id'] = str(id.int)[:10]
		return str(id.int)[:10]

def fastaformat(s):
	t = []
	temp = ""
	for i in s[1:]:	
		if i.startswith(">"):
			t.append(temp)
			temp = ""
			continue	
		temp = temp+i
	t.append(temp)
	return t
	
def writeresult(a,job_name):
	temp = f'{job_name}/temp.txt'
	result = f"{job_name}/result.txt"
	with open(result,'r') as f:
		with open(temp,'w') as f2: 
			f2.write(f"{a.job_name}\n")
			f2.write(f"{a.count}\n")
			f2.write(f"{a.model}\n")
			f2.write(f"{a.id}\n")
			f2.write(f.read())
	try:
		os.rename(temp, result)
	except WindowsError:
		os.remove(result)
		os.rename(temp, result)

def readresult(user):
	with open(f"{user}/result.txt",'r') as f: # the current directory is generated
		s = result()
		s.job_name = f.readline().replace("\n","")
		if len(s.job_name) == 0:
			return
		else:
			s.count = int(f.readline().replace("\n",""))
			s.model = int(f.readline().replace("\n",""))
			s.id = (f.readline().replace("\n",""))
			return s		

def check_user(request):
	os.chdir(root)
	id = get_id(request)
	generated = "olfy/static/olfy/generated"
	if not os.path.isdir(f"{generated}/{id}"): 
		os.makedirs(f"{generated}/{id}/m1")
		os.makedirs(f"{generated}/{id}/m2")
		os.makedirs(f"{generated}/{id}/m3")
		os.makedirs(f"{generated}/{id}/m4")	
		f = open(f"{generated}/{id}/result.txt", 'w')
		f.close()
	return id

def loadingpage(request):
	os.chdir(root)
	return render(request, "olfy/Ahuja labs website/loading.html", {'hide': 'd-none'})

@minified_response
def home(request):
	check_user(request)	
	os.chdir(root)
	context= {
		'hide': 'd-none'
	}
	return render(request, "olfy/Ahuja labs website/index.html", context)

@minified_response
def displaymodels(request):
	check_user(request)	
	return render(request, "olfy/Ahuja labs website/modelsList.html")

@minified_response
def about(request):
	os.chdir(root)
	context= {
		'team': [
			{
				'name': 'Dr. Gaurav Ahuja',
				'post': 'Principal Investigator',
				'email': 'gaurav.ahuja@iiitd.ac.in',
				'image': 'Gaurav.jpg'
			},
			{
				'name': 'Dr. Tripti Mishra',
				'post': 'Intellectual Contribution',
				'email': 'mistripti01@gmail.com',
				'image': 'Tripti.png'
			},
			{
				'name': 'Vishesh Agrawal',
				'post': 'Deep Learning & Interpretability',
				'email': 'vishesh18420@iiitd.ac.in',
				'image': 'Vishesh.png'
			},
			{
				'name': 'Ria Gupta',
				'post': 'Deep Learning & Interpretability',
				'email': 'ria18405@iiitd.ac.in',
				'image': 'Ria.png'
			},
			{
				'name': 'Rishi Raj Jain',
				'post': 'Lead Design & Development',
				'email': 'rishi18304@iiitd.ac.in',
				'image': 'Rishi.jpg'
			},
			{
				'name': 'Sushant Gupta',
				'post': 'Back-End Development ',
				'email': 'sushant19450@iiitd.ac.in',
				'image': 'Sushant.jpg'
			},
			{
				'name': 'Aayushi Mittal',
				'post': 'Data Collection & Design',
				'email': 'aayushim@iiitd.ac.in',
				'image': 'Aayushi.jpg'
			},
			{
				'name': 'Krishan Gupta',
				'post': 'Deep Learning Testing',
				'email': 'krishang@iiitd.ac.in',
				'image': 'Krishan.jpg'
			},
			{
				'name': 'Prakriti',
				'post': 'Testing',
				'email': 'prakriti19439@iiitd.ac.in',
				'image': 'Prakriti.jpg'
			},
			{
				'name': 'Riya Sogani',
				'post': 'Web Development & Testing',
				'email': 'riya19442@iiitd.ac.in',
				'image': 'Riya.jpg'
			}
		]
	}
	return render(request, "olfy/Ahuja labs website/about.html", context)

@minified_response
def help(request):
	os.chdir(root)
	check_user(request)	
	context={
		'tabledata': [
			{
				'name': 'Odorant Predictor',
				'columnames': 'SMILES',
				'extension': '*.csv'
			},
			{
				'name': 'OR Finder',
				'columnames': 'SMILES',
				'extension': '*.csv'
			},
			{
				'name': 'Odor Finder',
				'columnames': 'Header of FASTA file, Receptor Sequence',
				'extension': '*.csv'
			},
			{
				'name': 'Odorant-OR Pair Analysis',
				'columnames': 'SMILES, Header of FASTA file, Receptor Sequence',
				'extension': '*.csv'
			}
		],
		'troubleshoot':  [
			{
				'ques': 'Where can I see the input format?',
				'ans': 'See input format here and the sample CSV for each prediction engine here properly to understand and replicate the format.'
			},
			{
				'ques': 'If I log out of my browser, would my history remain saved?',
				'ans': 'Yes, your history will remain saved up to 7 days, or if you choose to clear your cookies in the browser cache.'
			},
			{
				'ques': 'Can I run 2 prediction models from different tabs of the same browser?',
				'ans': 'No.'
			},
			{
				'ques': 'Can I navigate away from the loading screen?',
				'ans': 'We understand that it can be a little time consuming, considering the high computations. Please try to be patient, and do not navigate away from the loading screen to get your results. You could, however, add your email address to receive your results.'
			},
			{
				'ques': 'What if I add more than 25 entries?',
				'ans': 'We only select the first 25 entries as input.'
			},
			{
				'ques': 'The result page shows a table of only ‘NA’ entries. What does this mean?',
				'ans': "Don't worry, NA stands for Not Applicable, which indicates that for the given input, there are no ligands/receptors which can bind to the input receptor/ligand with the given input parameters."
			},
			{
				'ques': 'How do I interpret my results?',
				'ans': 'We have used 3 ways to interpret results, receptor sequence-based, smile based. structure-based. Atoms highlighted in green symbolizes positive contribution towards the binding, and atoms highlighted in red would symbolize Negative contribution towards binding.'
			},
			{
				'ques': 'I set the counter value of Top-k to be ‘x’, but I receive ‘y’ output records (y<x)?',
				'ans': "The value of ‘k’ only sets an upper bound of the number of outputs you can get. It is possible to have fewer receptors binding a given input smile than k, or vice versa."
			},
			{
				'ques': 'What does the threshold mean?',
				'ans': "In OR finder, we have used a Tanimoto similarity threshold to find SMILES similar within that threshold. Setting a lower threshold would produce more output records."
			},
			{
				'ques': 'How to set a job title? Can I have special characters in my title?',
				'ans': "Yes, all characters are fit for job titles. We recommend using meaningful job names to keep track of the job. You can see the sample input for more information."
			},
			{
				'ques': 'What is the prediction based on?',
				'ans': "The prediction is based on Deep Learning Models."
			}
		]
	}
	return render(request, "olfy/Ahuja labs website/help.html", context)

def results(request):
	if request.method == "GET":
		os.chdir(root)
		id = check_user(request)
		user = f"olfy/static/olfy/generated/{id}"
		a = readresult(user)
		if a is None: 
			return render(request, "olfy/Ahuja labs website/results.html",{"z": False})
		elif a.model == 1:
			s = f'{user}/m1/{a.job_name}/predicted_output.csv'
			data = pd.read_csv(s)
			print(data)
			number_of_rows = len(data)
			display = []
			for i in range(number_of_rows):
				b = disp()
				b.smiles = data["smiles"][i]
				b.prob = str(data["prob"][i])[0:5]
				b.sno = i+1
				temp = data["pred_odor"][i]
				if temp == 1:
					odor = "odorant"
				else:
					odor = "odorless"
				b.odor = odor
				display.append(b)
			return render(request, "olfy/Ahuja labs website/results.html",{"result": a, "z":True, "display": [{"row": display}]})
		elif a.model == 2:
			display = []
			for i in range(a.count):
				data = pd.read_csv(f'{user}/m2/{a.job_name}/{i+1}/output.csv')
				number_of_rows = len(data["smiles"])
				temp = {}
				temp["smiles"] = data["smiles"][0]
				temp1 = []
				for j in range(number_of_rows):
					b = disp2()
					b.sno = j+1
					if "Empty" == data["Probability"][j]: 
						b.seq = "NA"
						b.receptorname = "NA"
						b.prob = "NA"
						b.noresult = True
					else:						
						b.seq = data["Final_Sequence"][j]
						b.receptorname = data["Receptor"][j]
						b.prob = str(data["Probability"][j])[0:5]
					b.tableno = i+1
					temp1.append(b)
				temp["row"] = temp1
				print(temp)
				display.append(temp)
			return render(request, "olfy/Ahuja labs website/results.html",{"result":a,"z":True,"display":display})
		elif a.model == 3:
			display = []
			for i in range(a.count):
				data = pd.read_csv(f'{user}/m3/{a.job_name}/{i+1}/output.csv')
				number_of_rows = len(data["Smiles"])
				temp = {}
				temp["seq"] = (data["header"][0])
				temp1 = []
				for j in range(number_of_rows):
					b = disp3()
					b.sno = j+1
					if "Empty" == data["Probability"][j]: 
						b.smiles = "NA"
						b.prob = "NA"
						b.noresult = True
					else:						
						b.smiles = data["Smiles"][j]
						b.prob = str(data["Probability"][j])[0:5]
					b.tableno = i+1
					temp1.append(b)
				temp["row"] = temp1
				display.append(temp)
			return render(request, "olfy/Ahuja labs website/results.html",{"result":a,"z":True,"display":display})
		elif a.model == 4:
			s = f'{user}/m4/{a.job_name}/output.csv'
			data = pd.read_csv(s)
			index = data.index
			number_of_rows = len(index)
			display = []
			for i in range(number_of_rows):
				b = disp4()
				b.smiles = data["smiles"][i]
				b.prob = str(data["prob"][i])[:5]
				b.sno = i+1
				b.seq = data["seq"][i]
				if data["status"][i] == 0:
					b.status = "non binding"
				else:
					b.status = "binding"
				display.append(b)
			return render(request, "olfy/Ahuja labs website/results.html",{"result":a,"z":True, "display": [{"row": display}]})

def result_queue(request,job_name,model,count):
	if request.method == "GET":
		os.chdir(root)
		id = check_user(request)
		user = f"olfy/static/olfy/generated/{id}"
		a = result()
		a.job_name = job_name
		a.model = int(model)
		a.count = int(count)
		a.id = id
		if a is None: 
			return render(request, "olfy/Ahuja labs website/results.html",{"z": False})
		elif a.model == 1:
			s = f'{user}/m1/{a.job_name}/predicted_output.csv'
			data = pd.read_csv(s)
			print(data)
			number_of_rows = len(data)
			display = []
			for i in range(number_of_rows):
				b = disp()
				b.smiles = data["smiles"][i]
				b.prob = str(data["prob"][i])[0:5]
				b.sno = i+1
				temp = data["pred_odor"][i]
				if temp == 1:
					odor = "odorant"
				else:
					odor = "odorless"
				b.odor = odor
				display.append(b)
			return render(request, "olfy/Ahuja labs website/results.html",{"result": a, "z":True, "display": [{"row": display}]})
		elif a.model == 2:
			display = []
			for i in range(a.count):
				data = pd.read_csv(f'{user}/m2/{a.job_name}/{i+1}/output.csv')
				number_of_rows = len(data["smiles"])
				temp = {}
				temp["smiles"] = data["smiles"][0]
				temp1 = []
				for j in range(number_of_rows):
					b = disp2()
					b.sno = j+1
					if "Empty" == data["Probability"][j]: 
						b.seq = "NA"
						b.receptorname = "NA"
						b.prob = "NA"
						b.noresult = True
					else:						
						b.seq = data["Final_Sequence"][j]
						b.receptorname = data["Receptor"][j]
						b.prob = str(data["Probability"][j])[0:5]
					b.tableno = i+1
					temp1.append(b)
				temp["row"] = temp1
				print(temp)
				display.append(temp)
			return render(request, "olfy/Ahuja labs website/results.html",{"result":a,"z":True,"display":display})
		elif a.model == 3:
			display = []
			for i in range(a.count):
				data = pd.read_csv(f'{user}/m3/{a.job_name}/{i+1}/output.csv')
				number_of_rows = len(data["Smiles"])
				temp = {}
				temp["seq"] = (data["seq"][0])
				temp1 = []
				for j in range(number_of_rows):
					b = disp3()
					b.sno = j+1
					if "Empty" == data["Probability"][j]: 
						b.smiles = "NA"
						b.prob = "NA"
						b.noresult = True
					else:						
						b.smiles = data["Smiles"][j]
						b.prob = str(data["Probability"][j])[0:5]
					b.tableno = i+1
					temp1.append(b)
				temp["row"] = temp1
				display.append(temp)
			return render(request, "olfy/Ahuja labs website/results.html",{"result":a,"z":True,"display":display})
		elif a.model == 4:
			s = f'{user}/m4/{a.job_name}/output.csv'
			data = pd.read_csv(s)
			index = data.index
			number_of_rows = len(index)
			display = []
			for i in range(number_of_rows):
				b = disp4()
				b.smiles = data["smiles"][i]
				b.prob = str(data["prob"][i])[:5]
				b.sno = i+1
				b.seq = data["seq"][i]
				if data["status"][i] == 0:
					b.status = "non binding"
				else:
					b.status = "binding"
				display.append(b)
			return render(request, "olfy/Ahuja labs website/results.html",{"result":a,"z":True, "display": [{"row": display}]})

def odor(request):
	if "GET" == request.method:
		os.chdir(root)
		check_user(request)		
		return render(request, "olfy/Ahuja labs website/odor.html")
	else:
		try:
			os.chdir(root)
			a = result()
			id = check_user(request)
			a.id = id
			userm1 = f"olfy/static/olfy/generated/{id}/m1"
			job_name = request.POST["job_name"]
			if len(job_name) == 0:
				job_name = "untitled"
			smiles = request.POST["smiles"]
			email = request.POST["email"]
			s = smiles.replace('\r',"").split('\n')
			if "" in s:
				s.remove("")
			temp = {"smiles":s}
			data = pd.DataFrame(temp)
			data = data.head(25)
			count = 1
			while os.path.isdir(f"{userm1}/{job_name}"):
				job_name = f"{job_name}1"
			os.mkdir(f"{userm1}/{job_name}")
			a.job_name = job_name
			job_name = f"{userm1}/{job_name}"
			path = os.path.abspath(job_name)
			data.to_csv(f"{path}/input.csv",index=False)
			a.model = 1
			os.chdir("olfy/static/olfy/generated/m1")
			shutil.copyfile("olfy_model_v1.tar",f'{path}/olfy_model_v1.tar')
			os.system(f"python transformer-cnn.py {path}")
			f = pd.read_csv(f"{path}/input.csv")
			smiles = f["smiles"]
			for i in smiles:
				smile_path = f"{path}/{count}"
				os.makedirs(smile_path)			
				cmd = f"python ochem.py detectodor.pickle "+ f'"{i}" ' f"{smile_path}"
				print(cmd)
				os.system(cmd)
				print("done")
				os.system(f"gnuplot " + f'"{path}/map.txt"')
				count+=1
			print("relax pdf")
			os.system(f"python generate_table.py {path}")
			os.remove(f"{path}/map.txt")
			os.remove(f"{path}/olfy_model_v1.tar")
			os.remove(f"{path}/results.csv")
			os.remove(f"{path}/input.csv")
			a.count = count-1
			os.chdir("../")
			writeresult(a,id)
			os.chdir("../../../../")
			if len(email)!=0:
				send_attachment(a,email,request)
			return JsonResponse({'code': 1})
		except Exception as e:
			traceback.print_exc()
			os.chdir(root)
			return JsonResponse({'code': 0})

# def getEmail(request):
# 	if "POST" == request.method:
# 		os.chdir(root)
# 		return JsonResponse({'code': 1})
# 		# registerEmail(request.POST['email'])

def odor_Or(request):
	if "GET" == request.method:
		os.chdir(root)
		check_user(request)			
		return render(request, "olfy/Ahuja labs website/odorOR.html")
	else:
		try:
			os.chdir(root)			
			a = result()
			job_name = request.POST["job_name"]
			if len(job_name) == 0:
				job_name = "untitled"
			smiles = request.POST["smiles"]
			fasta = request.POST["fasta"]
			email = request.POST["email"]
			id = check_user(request)
			a.id = id
			s = smiles.replace('\r',"").split('\n')
			if "" in s:
				s.remove("")
			t = fasta.replace('\r',"").split('\n')
			if "" in t:
				t.remove("")
			seq = []
			header = []
			for i in range(0,len(t),2):
				header.append(t[i][1:].strip())
				seq.append(t[i+1].strip())
			userm4 = f"olfy/static/olfy/generated/{id}/m4"
			temp = {"smiles":s,"seq":seq,"header":header}
			data = pd.DataFrame(temp)
			data = data.head(25)
			while os.path.isdir(f"{userm4}/{job_name}"):
				job_name = f"{job_name}1"
			a.job_name = job_name
			job_name = f"{userm4}/{job_name}"
			os.mkdir(job_name)
			path = os.path.abspath(job_name)
			data.to_csv(f"{path}/input.csv",index=False)
			os.chdir("olfy/static/olfy/generated/m4")
			os.system(f"python M4_final.py {path}")
			a.model = 4
			data = pd.read_csv(f"{path}/output.csv")
			number_of_rows = len(data)
			os.remove(f"{path}/input.csv")
			a.count = number_of_rows
			os.chdir("../")
			writeresult(a,id)
			for i in range(4):
				os.chdir("../")
			if len(email)!=0:
				send_attachment(a,email,request)			
			return JsonResponse({'code': 1})
		except Exception as e:
			os.chdir(root)
			traceback.print_exc()
			return JsonResponse({'code': 0})	

def Or(request):
	if "GET" == request.method:
		os.chdir(root)
		check_user(request)	
		return render(request, "olfy/Ahuja labs website/or.html")		
	else:
		try:
			os.chdir(root)
			a = result()
			id = check_user(request)
			a.id = id
			job_name = request.POST["job_name"]
			if len(job_name) == 0:
				job_name = "untitled"
			fasta = request.POST["fasta"]
			email = request.POST["email"]
			counter = request.POST["normal_counter"]
			t = fasta.replace('\r',"").split('\n')
			if "" in t:
				t.remove("")
			seq = []
			header = []
			for i in range(0,len(t),2):
				header.append(t[i][1:].strip())
				seq.append(t[i+1].strip())
			temp = {"seq":seq,"header":header}
			data = pd.DataFrame(temp)
			while os.path.isdir(f"olfy/static/olfy/generated/{id}/m3/{job_name}"):
				job_name = f"{job_name}1"
			a.job_name = job_name
			job_name = f"olfy/static/olfy/generated/{id}/m3/{job_name}"
			os.mkdir(job_name)
			path = os.path.abspath(job_name)
			data = data.head(25)
			data.to_csv(f"{path}/input.csv",index=False)
			a.model = 3
			f = pd.read_csv(f"{path}/input.csv")
			print(f)
			os.chdir('olfy/static/olfy/generated/m3')
			a.count = len(f["seq"])
			for i in range(len(f["seq"])):
				dic = {"seq":[f["seq"][i]],"k":int(counter)}
				df = pd.DataFrame(dic)
				os.makedirs(f"{path}/{i+1}")
				df.to_csv(f"{path}/{i+1}/temp.csv",index=False)
				os.system(f"python M3.py {path}/{i+1}")
				os.remove(f"{path}/{i+1}/temp.csv")
				df = pd.read_csv(f"{path}/{i+1}/output.csv")
				j = []
				for k in range(len(df["Probability"])):
					j.append(f["seq"][i])
				df["seq"] = j
				j = []
				for k in range(len(df["Probability"])):
					j.append(f["header"][i])				
				df["header"] = j
				df.to_csv(f"{path}/{i+1}/output.csv",index=False)
			os.remove(f"{path}/input.csv")
			os.chdir("../")
			writeresult(a,id)
			for i in range(4):
				os.chdir("../")
			if len(email)!=0:
				send_attachment(a,email,request)
			return JsonResponse({'code': 1})
		except Exception as e:
			traceback.print_exc()
			os.chdir(root)
			return JsonResponse({'code': 0})	
		
def odor2(request):
	if "GET" == request.method:
		os.chdir(root)
		check_user(request)	
		return render(request, "olfy/Ahuja labs website/odor2.html")
	else:
		try:
			os.chdir(root)
			a = result()
			id = check_user(request)
			a.id = id
			job_name = request.POST["job_name"]
			if len(job_name) == 0:
				job_name = "untitled"
			smiles = request.POST["smiles"]
			email = request.POST["email"]
			slider = request.POST["slider_value"]
			counter = request.POST["normal_counter"]
			switch = request.POST["typeOfTesting"]
			t = smiles.replace('\r',"").split('\n')
			if "" in t:
				t.remove("")
			temp = {"smiles":t}
			data = pd.DataFrame(temp)
			data = data.head(25)
			a.model = 2
			userm2 = f"olfy/static/olfy/generated/{id}/m2"
			while os.path.isdir(f"{userm2}/{job_name}"):
				job_name = f"{job_name}1"
			a.job_name = job_name
			job_name = f"{userm2}/{job_name}"
			path = os.path.abspath(job_name)
			os.mkdir(job_name)
			data.to_csv(f"{path}/input.csv",index=False)
			os.chdir("olfy/static/olfy/generated/m2")
			f = pd.read_csv(f"{path}/input.csv")
			a.count = len(f["smiles"])
			for i in range(len(f["smiles"])):
				if switch=="Rapid":
					dic = {"smiles":[f["smiles"][i]],"threshhold":float(slider)}
					df = pd.DataFrame(dic)
					os.makedirs(f"{path}/{i+1}")
					df.to_csv(f"{path}/{i+1}/temp.csv",index=False)
					os.system(f"python M2.py {path}/{i+1}")
					os.remove(f"{path}/{i+1}/temp.csv")
				elif switch=="Normal":
					dic = {"smiles":[f["smiles"][i]],"k":int(counter)}
					df = pd.DataFrame(dic)
					os.makedirs(f"{path}/{i+1}")
					df.to_csv(f"{path}/{i+1}/temp.csv",index=False)
					os.system(f"python M2-brute-force.py {path}/{i+1}")
					os.remove(f"{path}/{i+1}/temp.csv")					
				else:
					print("bruteforce",slider,type(slider))
					dic = {"smiles":[f["smiles"][i]],"threshhold":float(slider)}
					df = pd.DataFrame(dic)
					os.makedirs(f"{path}/{i+1}")
					df.to_csv(f"{path}/{i+1}/temp.csv",index=False)
					os.system(f"python M2.py {path}/{i+1}")
					os.remove(f"{path}/{i+1}/temp.csv")
				df = pd.read_csv(f"{path}/{i+1}/output.csv")
				j = []
				for k in range(len(df["Probability"])):
					j.append(f["smiles"][i])
				df["smiles"] = j
				df.to_csv(f"{path}/{i+1}/output.csv",index=False)
			os.remove(f"{path}/input.csv")
			os.chdir("../")
			writeresult(a,id)
			for i in range(4):
				os.chdir("../")
			if len(email)!=0:
				send_attachment(a,email,request)			
			return JsonResponse({'code': 1})
		except Exception as e:
			traceback.print_exc()
			os.chdir(root)
			return JsonResponse({'code': 0})

@minified_response
def contactus(request):
	if "GET" == request.method:
		os.chdir(root)
		check_user(request)	
		return render(request, "olfy/Ahuja labs website/contact.html")
	else:
		try:
			os.chdir(root)
			email= request.POST["email"]
			subject= request.POST["title"]
			message= request.POST["message"]
			nameUser= request.POST["name"]
			print(email, subject, message, nameUser)
			sender= "odorify.ahujalab@iiitd.ac.in"
			msg= MIMEMultipart()
			msg['From']= sender
			msg['To']= sender
			msg['Subject']= subject
			msg.attach(MIMEText(message, 'plain'))
			text= msg.as_string()
			s= smtplib.SMTP('smtp.gmail.com', 587)
			s.starttls()
			s.login(sender, "odorify123")
			s.sendmail(sender, sender, text)
			msg = MIMEMultipart()
			msg['From']= sender
			msg['To']= email
			msg['Subject']= f"Odorify Query: {subject}"
			message= f"Hi {nameUser},\nWe appreciate your interest in OdoriFy. We've received your query and we'll get back to you with a (human) response as soon as possible.\n\nCheers,\nOdoriFy Bot"
			msg.attach(MIMEText(message, 'plain'))
			text= msg.as_string() 
			s.sendmail(sender, email, text)
			s.quit()
			return JsonResponse({'code': 1})
		except:
			return JsonResponse({'code': 0})

def queue(request):
	if "GET" == request.method:
		os.chdir(root)
		id = check_user(request)
		f = open(f"olfy/static/olfy/generated/{id}/result.txt")
		data = f.read().splitlines()
		length = len(data)
		queue = []
		for i in range(0,length,4):
			temp = queuedisp()
			temp.count = data[i+1]
			temp.job_name = data[i]
			temp.sno = (i//4)+1
			temp.model = data[i+2]
			if temp.model == '1':
				temp.model_name = "Odorant Predictor"
			elif temp.model == '2':
				temp.model_name = "OR Finder"
			elif temp.model == '3':
				temp.model_name = "Odor Finder"
			elif temp.model == '4':
				temp.model_name = "Odorant-OR Pair Analysis"
			queue.append(temp)
		return render(request, "olfy/Ahuja labs website/queue.html",{"queue":queue})

def makezip(a,request):
	os.chdir(root)
	id = check_user(request)
	file_path = []
	os.chdir(f"olfy/static/olfy/generated/{id}/m1")
	for i in range(a.count):
		print(f"{a.job_name}/{i+1}/lrp.pdf")
		file_path.append(f"{a.job_name}/{i+1}/lrp.pdf")
		file_path.append(f"{a.job_name}/{i+1}/mol.svg") 
	file_path.append(f"{a.job_name}/predicted_output.csv")
	zip = ZipFile(f"{a.job_name}/data.zip",'w') 
	for file in file_path: 
		zip.write(file)
	zip.close()
	zip = open(f"{a.job_name}/data.zip","rb")
	for i in range(6):
		os.chdir("../")
	return zip

def makezip2(a,request):
	os.chdir(root)
	id = check_user(request)
	file_path = []
	os.chdir(f"olfy/static/olfy/generated/{id}/m2")
	for i in range(a.count):
		print(os.getcwd())
		f = pd.read_csv(f"{a.job_name}/{i+1}/output.csv")
		print(f)
		count = len(f)
		for j in range(count):
			if "Empty" != f["Probability"][0]:
				file_path.append(f"{a.job_name}/{i+1}/{j+1}_SmileInterpretability.png")
				file_path.append(f"{a.job_name}/{i+1}/{j+1}_SequenceInterpretability.pdf") 
				file_path.append(f"{a.job_name}/{i+1}/{j+1}_mol.svg") 
		file_path.append(f"{a.job_name}/{i+1}/output.csv")
	zip = ZipFile(f"{a.job_name}/data.zip",'w') 
	for file in file_path: 
		zip.write(file)
	zip.close()
	zip = open(f"{a.job_name}/data.zip","rb")
	for i in range(6):
		os.chdir("../")
	return zip

def makezip3(a,request):
	os.chdir(root)
	id = check_user(request)
	file_path = []
	os.chdir(f"olfy/static/olfy/generated/{id}/m3")
	for i in range(a.count):
		f = pd.read_csv(f"{a.job_name}/{i+1}/output.csv")
		count = len(f)
		for j in range(count):
			if "Empty" != f["Probability"][0]:
				file_path.append(f"{a.job_name}/{i+1}/{j+1}_SmileInterpretability.png")
				file_path.append(f"{a.job_name}/{i+1}/{j+1}_SequenceInterpretability.pdf") 
				file_path.append(f"{a.job_name}/{i+1}/{j+1}_mol.svg") 
		file_path.append(f"{a.job_name}/{i+1}/output.csv")
	zip = ZipFile(f"{a.job_name}/data.zip",'w') 
	for file in file_path: 
		zip.write(file)
	zip.close()
	zip = open(f"{a.job_name}/data.zip","rb")
	for i in range(6):
		os.chdir("../")	
	return zip

def makezip4(a,request):
	os.chdir(root)
	id = check_user(request)
	file_path = []
	os.chdir(f"olfy/static/olfy/generated/{id}/m4")
	for i in range(a.count):
		file_path.append(f"{a.job_name}/{i+1}_SmileInterpretability.png")
		file_path.append(f"{a.job_name}/{i+1}_SequenceInterpretability.pdf") 
		file_path.append(f"{a.job_name}/{i+1}_mol.svg") 
	file_path.append(f"{a.job_name}/output.csv")
	zip = ZipFile(f"{a.job_name}/data.zip",'w') 
	for file in file_path: 
		zip.write(file)
	zip.close()
	zip = open(f"{a.job_name}/data.zip","rb")
	for i in range(6):
		os.chdir("../")
	return zip

def download(request,job_name,model,count):
	os.chdir(root)
	a = result()
	a.job_name = job_name
	a.model = int(model)
	a.count = int(count)
	zip = ''
	if a.model==1:
		zip = makezip(a,request)
	if a.model==2:
		zip = makezip2(a,request)
	if a.model==3:
		zip = makezip3(a,request)
	if a.model==4:
		zip = makezip4(a,request)
	response = HttpResponse(zip,content_type='application/zip')
	response['Content-Disposition'] = 'attachment; filename=data.zip'
	return response

def send_attachment(a,email,request):
	os.chdir(root)
	attachment = ""
	sender = "odorify.ahujalab@iiitd.ac.in"
	if a.model==1:
		attachment = makezip(a,request)
	if a.model==2:
		attachment = makezip2(a,request)
	if a.model==3:
		attachment = makezip3(a,request)
	if a.model==4:
		attachment = makezip4(a,request)

	msg = MIMEMultipart()
	msg['From'] = sender
	msg['To'] = email
	msg['Subject'] = "Results"
	filename = "data.zip"
	p = MIMEBase('application', 'octet-stream') 
	p.set_payload((attachment).read()) 
	encoders.encode_base64(p) 
	p.add_header('Content-Disposition', "attachment; filename= %s" % filename) 
	msg.attach(p) 
	message = "Dear User,\n Thank you for using OdoriFy.\n Please find attached your combined results in a zip file. In case of any queries, please contact us <a href='www.odorify.iiitd.edu.in/contact'>here</a>."
	msg.attach(MIMEText(message, 'plain'))	
	text = msg.as_string() 
	s = smtplib.SMTP('smtp.gmail.com', 587)
	s.starttls()
	s.login(sender, "odorify123")
	s.sendmail(sender, email, text)
	s.quit()
